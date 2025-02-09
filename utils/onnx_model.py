import torch
import torch.onnx
from model import ImageClassifier
from config.config import cfg

def convert_onnx(model_path, save_path="model"):
    save_as = f"{save_path}.onnx"

    model = ImageClassifier(
        cfg=cfg,
    ).get_model(model_path=model_path)
    model.eval()

    example_input = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    torch.onnx.export(
        model,
        example_input,
        save_as,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

if __name__ == "__main__":
    convert_onnx("resnet50.pth", "test_onnx")