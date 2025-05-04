import argparse
import numpy as np
import torch
import onnxruntime as ort
import cv2
from config import cfg
import gradio as gr
from model import ImageClassifier

def load_model(cfg, model_path, model_arch=None):
    if model_path.endswith(".onnx"):
        model = ort.InferenceSession(model_path)
    elif model_path.endswith(".pth"):
        if model_arch is None:
            raise ValueError("For .pth models, provide a model architecture.")
        
        model = model_arch
        state_dict = torch.load(model_path, map_location=cfg.device)

        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ValueError("Unsupported model format. Use .onnx or .pth")
    return model

def gradio_application(cfg, model, model_path, n_preds=3):
    def predict(inp):
        image_np = np.array(inp)
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        augmented = cfg.test_transform(image=image_np)
        inp = augmented['image'].unsqueeze(0)

        if model_path.endswith(".onnx"):
            inp = inp.numpy()
            ort_inputs = {model.get_inputs()[0].name: inp}
            ort_outs = model.run(None, ort_inputs)
            prediction = ort_outs[0][0]
        else:
            with torch.no_grad():
                prediction = model(inp).squeeze(0).numpy()

        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        probs = {cfg.classes[i]: float(prediction[i]) for i in range(len(cfg.classes))}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {label: prob for label, prob in sorted_probs}
    
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", height=512, width=728, scale=True, interactive=True),
        outputs=gr.Label(num_top_classes=n_preds),
        title="Intel scene image classification",
        examples=[f"./examples/test/image{img}.jpg" for img in range(1, 7)]
    ).launch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="resnet50.pth", help='Model to use (.pth or .onnx)')
    parser.add_argument('--top_n', type=int, default=3, help='Top n predictions')
    args = parser.parse_args()
    if args.model_path.endswith(".pth"):
        model_arch = ImageClassifier(cfg).get_model(args.model_path)
    else:
        model_arch = None
    model = load_model(cfg, args.model_path, model_arch)

    gradio_application(
        cfg=cfg,
        model=model,
        model_path=args.model_path,
        n_preds=args.top_n
    )