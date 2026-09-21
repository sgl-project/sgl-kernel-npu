"""Model-specific PLE convolution: native accuracy, graph replay and rejection."""
import importlib

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from sgl_kernel_npu.qwen3_8_flash_next.short_conv import launch_info, short_conv

pytestmark = pytest.mark.skipif(not torch_npu.npu.is_available(), reason="NPU is required")
TOLERANCES = {torch.bfloat16: 0.02, torch.float16: 0.003}


def native(x, weight, dilation, *, single_token=False):
    width = x.shape[-1] - 9
    if not x.shape[0]:
        return x.new_empty((0, 10240) if single_token else (0, width, 10240))
    out = F.conv1d(x, weight, dilation=dilation, groups=10240)
    return out.squeeze(-1) if single_token else out.transpose(1, 2)

DTYPES = [torch.bfloat16, torch.float16]
MODES = [(1, True), (1, False), (2, False), (3, False), (4, False)]


def make_case(case, dtype=torch.bfloat16):
    _, batch, channels, k, d, width, _ = case
    torch.manual_seed(42)
    x = torch.randn(batch, channels, (k-1)*d+width, device='npu', dtype=dtype)
    w = torch.randn(channels, 1, k, device='npu', dtype=dtype)
    return x, w


def check_output(label, out, x, w, single):
    # Native convolution is intentionally outside all graph capture contexts.
    ref = native(x, w, 3, single_token=single)
    tol = TOLERANCES[x.dtype]
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
    width = x.shape[-1]-9
    assert tuple(out.shape) == ((x.shape[0],10240) if single else (x.shape[0],width,10240))
    assert out.dtype == x.dtype and out.device == x.device and out.is_contiguous()


def assert_unchanged(x, w, before_x, before_w):
    torch.testing.assert_close(x, before_x, atol=0, rtol=0)
    torch.testing.assert_close(w, before_w, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width,single', MODES)
@pytest.mark.parametrize('batch', [0,1,2,3,5,8,17,32,65])
def test_model_contract(dtype, width, single, batch):
    case = ('model',batch,10240,4,3,width,single)
    x, w = make_case(case, dtype)
    before_x, before_w = x.clone(), w.clone()
    out = short_conv(x, w, 3, single_token=single)
    check_output(f'model/B{batch}/W{width}/single{single}', out, x, w, single)
    assert_unchanged(x, w, before_x, before_w)
    info = launch_info(x, w, 3, single_token=single)
    assert info['branch'] == ('empty' if batch == 0 else 'triton')
    if batch:
        block = (512 if batch == 1 else 1024) if width == 1 else (256 if batch == 1 else 512)
        assert info['block_size'] == block
        assert info['tasks'] == batch * (10240 // block)
        assert 0 < info['grid'] <= info['tasks']


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width', [1,2,3,4])
def test_contiguous_offset_and_padding(dtype, width):
    x, w = make_case(('offset',3,10240,4,3,width,False), dtype)
    backing_x = torch.empty(x.numel()+3, device=x.device, dtype=dtype)
    backing_w = torch.empty(w.numel()+5, device=w.device, dtype=dtype)
    backing_x.fill_(7); backing_w.fill_(-3)
    sx = backing_x[3:].view_as(x); sx.copy_(x); x = sx
    sw = backing_w[5:].view_as(w); sw.copy_(w); w = sw
    # Graph padding is input data: preserve history and compute padded rows too.
    x[1,:,9:] = 0
    x[2] = 0
    bx, bw = backing_x.clone(), backing_w.clone()
    check_output('offset/padded_rows', short_conv(x,w,3), x,w,False)
    torch.testing.assert_close(backing_x,bx,atol=0,rtol=0)
    torch.testing.assert_close(backing_w,bw,atol=0,rtol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width,single', MODES)
@pytest.mark.parametrize('batch,grid_mode', [(0,'native'),(1,'native'),(8,'native'),
                                           (32,'native'),(3,'reuse'),(5,'nondivisible')])
def test_graph_updates(dtype,width,single,batch,grid_mode,monkeypatch):
    module = importlib.import_module('sgl_kernel_npu.qwen3_8_flash_next.short_conv')
    block = (512 if batch == 1 else 1024) if width == 1 else (256 if batch == 1 else 512)
    if grid_mode != 'native':
        cores = 10240//block if grid_mode == 'reuse' else 3
        monkeypatch.setattr(module, '_vector_cores', lambda device_index: cores)
    x,w = make_case(('graph',batch,10240,4,3,width,single),dtype)
    for _ in range(2):
        short_conv(x,w,3,single_token=single)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    bx,bw=x.clone(),w.clone()
    with torch.npu.graph(graph):
        out=short_conv(x,w,3,single_token=single)
    assert_unchanged(x,w,bx,bw)
    for label, changed in [('initial',None),('input',x),('weight',w)]:
        if changed is not None:
            changed.mul_(-0.7)
        bx,bw=x.clone(),w.clone()
        graph.replay()
        torch.npu.synchronize()
        check_output(f'graph/B{batch}/W{width}/single{single}/{grid_mode}/{label}',out,x,w,single)
        assert_unchanged(x,w,bx,bw)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width', [1,2,3,4])
@pytest.mark.parametrize('delta', [-1,0,1])
def test_grid_boundary(dtype,width,delta,monkeypatch):
    # B changes task count in channel-tile units. Put the physical-grid limit
    # immediately below/at/above the total to exercise launch and loop tails.
    module=importlib.import_module('sgl_kernel_npu.qwen3_8_flash_next.short_conv')
    x,w=make_case(('grid_boundary',2,10240,4,3,width,False),dtype)
    tasks=2*(10240//(1024 if width==1 else 512))
    monkeypatch.setattr(module,'_vector_cores',lambda device_index:tasks+delta)
    info=launch_info(x,w,3)
    assert info['grid']==min(tasks,tasks+delta)
    check_output(f'grid_boundary/W{width}/{delta}',short_conv(x,w,3),x,w,False)


class Metadata:
    """Host metadata double: address/device rejection needs no huge allocation."""
    def __init__(self,shape,dtype=torch.bfloat16,device=None,contiguous=True):
        self.shape=shape; self.ndim=len(shape); self.dtype=dtype
        self.device=device or torch.device('npu',torch.npu.current_device())
        self.contiguous=contiguous
    def is_contiguous(self):
        return self.contiguous


REJECTIONS = [
    ('input_rank',dict(xshape=(1,10240))),
    ('weight_rank',dict(wshape=(10240,4))),
    ('channels',dict(xshape=(1,17,13),wshape=(17,1,4))),
    ('channel_mismatch',dict(wshape=(10239,1,4))),
    ('kernel',dict(wshape=(10240,1,1))),
    ('weight_axis',dict(wshape=(10240,2,4))),
    *[(f'dilation_{v}',dict(dilation=v)) for v in (0,-1,1,2,4,3.0,True)],
    *[(f'width_{v}',dict(xshape=(1,10240,9+v))) for v in (0,5,1024,4096,4097)],
    ('single_width',dict(single=True)),
    ('single_type',dict(single=1)),
    *[(f'dtype_{v}',dict(dtype=v)) for v in (torch.float32,torch.float64,torch.int32)],
    ('mixed_dtype',dict(wdtype=torch.float16)),
    ('input_layout',dict(xcontig=False)),
    ('weight_layout',dict(wcontig=False)),
    ('cpu',dict(device=torch.device('cpu'))),
    ('empty_out_of_scope',dict(xshape=(0,17,13),wshape=(17,1,4))),
]


@pytest.mark.parametrize('label,settings',REJECTIONS,ids=[x[0] for x in REJECTIONS])
def test_reject_metadata(label,settings):
    s=settings
    x=Metadata(s.get('xshape',(1,10240,13)),s.get('dtype',torch.bfloat16),
               s.get('device'),s.get('xcontig',True))
    w=Metadata(s.get('wshape',(10240,1,4)),s.get('wdtype',x.dtype),
               x.device,s.get('wcontig',True))
    with pytest.raises(ValueError):
        short_conv(x,w,s.get('dilation',3),single_token=s.get('single',False))


def test_reject_device_mismatch():
    x=Metadata((1,10240,13))
    # No operation is sent to another device; only torch.device metadata exists.
    w=Metadata((10240,1,4),device=torch.device('npu',(x.device.index+1)%16))
    with pytest.raises(ValueError,match='same NPU'):
        short_conv(x,w,3)


@pytest.mark.parametrize('width',[1,2,3,4])
@pytest.mark.parametrize('delta',[0,1])
def test_int32_boundary(width,delta,monkeypatch):
    import triton
    module=importlib.import_module('sgl_kernel_npu.qwen3_8_flash_next.short_conv')
    monkeypatch.setattr(module,'_vector_cores',lambda device_index:40)
    block=1024 if width==1 else 512
    margin=triton.next_power_of_2(block*(width+9))
    max_batch=(2**31-1-margin)//(10240*(width+9))
    x=Metadata((max_batch+delta,10240,width+9));w=Metadata((10240,1,4))
    if delta:
        with pytest.raises(ValueError,match='int32'):
            short_conv(x,w,3)
    else:
        assert launch_info(x,w,3)['branch']=='triton'


@pytest.mark.parametrize('which',['input','weight'])
def test_reject_real_strided_tensor(which):
    x,w=make_case(('strided',1,10240,4,3,4,False))
    v=x if which=='input' else w
    backing=torch.empty((*v.shape[:-1],v.shape[-1]*2),device=v.device,dtype=v.dtype)
    backing[...,::2]=v
    if which=='input': x=backing[...,::2]
    else: w=backing[...,::2]
    with pytest.raises(ValueError,match='contiguous'):
        short_conv(x,w,3)
