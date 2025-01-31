import numpy as np
import torch
import onnxruntime as ort
import cv2
from config import cfg
import gradio as gr
from torchvision import models
from model import ImageClassifier

def load_model(model_path, model_arch=None):
    if model_path.endswith(".onnx"):
        model = ort.InferenceSession(model_path)
    elif model_path.endswith(".pth"):
        if model_arch is None:
            raise ValueError("For .pth models, provide a model architecture.")
        
        model = model_arch
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ValueError("Unsupported model format. Use .onnx or .pth")
    return model

def gradio_application(model, model_path, transform, labels, n_preds=4):
    def predict(inp):
        image_np = np.array(inp)
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        augmented = transform(image=image_np)
        inp = augmented['image'].unsqueeze(0)

        if model_path.endswith(".onnx"):
            inp = inp.numpy()  # Convert to numpy for ONNX
            ort_inputs = {model.get_inputs()[0].name: inp}
            ort_outs = model.run(None, ort_inputs)
            prediction = ort_outs[0][0]
        else:
            with torch.no_grad():
                prediction = model(inp).squeeze(0).numpy()

        # make this code better? What if num_top_classes=4, what about sorted_probs[:3] then?
        # Apply softmax for probabilities
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        probs = {labels[i]: float(prediction[i]) for i in range(len(labels))}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {label: prob for label, prob in sorted_probs}
    
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", height=512, width=728, scale=True, interactive=True),
        outputs=gr.Label(num_top_classes=n_preds),
        title="Intel scene image classification",
        examples=["data/seg_pred/seg_pred/3.jpg"], # use another image, call it example/image1.jpg (or png) or something, do 3-5 images?
    ).launch()


if __name__ == '__main__':
    model_path = "model.onnx" # turn into arg parser? can also parser top n predictions?
    # make some argparser for arguments that go inside interface and .launch()?
    if model_path.endswith(".pth"):
        model_arch = ImageClassifier(cfg).get_model(model_path)
    else:
        model_arch = None
    model = load_model(model_path, model_arch)
    labels = cfg.classes
    transform = cfg.test_transform

    gradio_application(
        model=model,
        model_path=model_path,
        transform=transform,
        labels=labels
    )