import gradio as gr
import cv2
import onnxruntime as ort
import numpy as np
from config.config import cfg

# make one also for non-onnx?

ort_session = ort.InferenceSession("model.onnx")
labels = cfg.classes
transform = cfg.test_transform

def predict(inp):
    image_np = np.array(inp)
    if image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    augmented = transform(image=image_np)
    inp = augmented['image'].unsqueeze(0).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: inp}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = ort_outs[0][0]

    prediction = np.exp(prediction) / np.sum(np.exp(prediction))
    probs = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return {label: prob for label, prob in sorted_probs[:3]}

# see more about layout here:
# https://www.gradio.app/guides/controlling-layout
# try to add more functionality?
# maybe log whats been classified into a file?

if __name__ == '__main__':
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", height=512, width=728, scale=True, min_width=250, interactive=True),
        outputs=gr.Label(num_top_classes=3),
        title="Intel scene image classification",
    ).launch()