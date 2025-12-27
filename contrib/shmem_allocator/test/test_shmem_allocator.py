import torch
import torch_npu
from torch import nn

from shmem_allocator import switch_to_shmem_allocator, init_shmem


def train(num_iter=500, device="npu"):
    """a tiny transformer training process"""
    model = nn.Transformer(
        d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2
    ).to(device=device)
    x = torch.randn(size=(1, 1024, 512), device=device)
    tgt = torch.rand(size=(1, 1024, 512), device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for _i in range(num_iter):
        y = model(x, tgt)
        loss = criterion(y, labels)
        loss.backward()
        if _i % 20 == 0:
            print(f"[step{_i}] loss: {loss.item()}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

def infer(device="npu"):
    from torchvision.io import decode_image
    from torchvision.models import resnet50, ResNet50_Weights

    # img = decode_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval().to(device=device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    # batch = preprocess(img).unsqueeze(0)

    batch = torch.zeros((1, 3, 224, 224), device=device)
    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    print(prediction)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    # This will allocate memory in the device using the new allocator
    switch_to_shmem_allocator()
    torch.npu.set_device(0)
    init_shmem(0, 1, 10 * (1024 ** 3), 1 * (1024 ** 3), 'tcp://127.0.0.1:3721')
    npu_tensor = torch.zeros(10, device="npu")
    print(npu_tensor)

    # full demo test
    # train()
    infer()
