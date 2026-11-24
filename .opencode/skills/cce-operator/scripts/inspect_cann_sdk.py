"""Read-only inventory of the active CANN/CCE SDK. No NPU initialization required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HEADERS = [
    '__clang_cce_runtime_wrapper.h', '__clang_cce_runtime.h', '__clang_cce_defines.h',
    '__clang_cce_types.h', 'cce_aicore_intrinsics.h',
    '__clang_cce_aicore_builtin_vars.h', '__clang_cce_aicore_intrinsics.h',
    '__clang_cce_aicore_functions.h', '__clang_cce_vector_types.h',
    '__clang_cce_vector_intrinsics.h', '__clang_cce_simt.h',
    '__clang_cce_simt_dav3510_intrinsic.h',
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=os.environ.get('ASCEND_HOME_PATH'))
    p.add_argument('--compiler', default=os.environ.get('BISHENG', 'bisheng'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler:
        p.error('bisheng unavailable: load the target CANN environment or pass --compiler')
    resource = Path(subprocess.check_output([compiler, '-print-resource-dir'], text=True).strip()).resolve()
    root = Path(args.root).resolve() if args.root else Path(compiler).absolute().parent.parent.resolve()
    def file_info(path):
        info = {'path': str(path), 'exists': path.is_file()}
        if info['exists']:
            info.update(resolved=str(path.resolve()), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return info
    result = {
        'sdk_root': str(root), 'compiler': str(Path(compiler).resolve()),
        'compiler_version': subprocess.check_output([compiler, '--version'], text=True).strip(),
        'resource_directory': str(resource),
        'version_files': {str(path.relative_to(root)): path.read_text(errors='replace')
                          for path in (root/'compiler/version.info', root/'opp/version.info') if path.is_file()},
        'headers': {name: file_info(resource/'include'/name) for name in HEADERS},
        'sdk_entries': {name: file_info(root/name) for name in [
            'include/acl/acl_rt.h', 'pkg_inc/runtime/runtime/rt.h',
            'include/aclnnop/aclnn_fused_causal_conv1d.h',
            'asc/include/c_api/asc_simd.h', 'asc/include/c_api/sync/sync.h',
            'tools/msobjdump/msobjdump', 'bin/llvm-objdump', 'bin/msprof']},
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f'SDK: {root}\nCCE resources: {resource}\nInventory: {args.output}')
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
