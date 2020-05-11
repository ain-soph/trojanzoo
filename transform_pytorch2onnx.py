import torch
from package.model.image.resnetnew import ResNetNew
a=ResNetNew(num_classes=10)
a.load_pretrained_weights('./model.pth')
model=a._model

dummy_input = torch.randn(128, 3, 32, 32, device='cuda')
input_names = ["input"]
output_names = [ "output" ]
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)