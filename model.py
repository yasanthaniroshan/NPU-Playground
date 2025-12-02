import torch
import torch.nn as nn
import onnx

class MatMulViaConv2D(nn.Module):
    def __init__(self, B):
        super().__init__()
        n, p = B.shape
        # Conv2d weight: (out_channels, in_channels, H, W)
        weight = B.T.view(p, 1, n, 1)
        self.conv = nn.Conv2d(1, p, kernel_size=(n, 1), bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, A):
        out = self.conv(A)
        return out.squeeze(2).transpose(1, 2)  # reshape to (batch, H, out_channels)


# Sample input
m, n = 32, 32
A = torch.randn(m, n)
B = torch.randn(n, n)
mean = B.mean()
std = B.std()

B = (B - mean) / std

model = MatMulViaConv2D(B)

dummy_input = torch.randn(1, 1, 32, 32)  # batch=1, channel=1, height=32, width=32
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18
)



model = onnx.load("model.onnx")
onnx.save(model, "model.onnx", save_as_external_data=False, all_tensors_to_one_file=True)

print("ONNX model exported with input shape [1,1,32,32]")