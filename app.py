import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner("pet_classifier_v1.pkl")

# 2. Map of English labels to Traditional Chinese (Taiwan standard)
# This adds that "Premium / Localized" feel
label_map = {
    "cat": "貓 (Cat)",
    "dog": "狗 (Dog)",
    "goldfish": "金魚 (Goldfish)",
    "hamster": "倉鼠 (Hamster)",
    "turtle": "烏龜 (Turtle)",
    "parrot": "鸚鵡 (Parrot)",
    "snake": "蛇 (Snake)",
}


# 3. Prediction Function
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    # Convert predictions to the mapped Chinese/English format
    return {label_map[c]: float(probs[i]) for i, c in enumerate(learn.dls.vocab)}


# 4. Custom CSS for "Premium" Styling
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
#header h1 { color: #ffffff; font-weight: 600; letter-spacing: -0.02em; }
#student-info { color: #a1a1aa; font-family: monospace; font-size: 0.9em; }
.gradio-container { background-color: #09090b !important; }
"""

# 5. Build the Interface with a Modern Theme
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="zinc", spacing_size="sm", radius_size="lg"),
    css=custom_css,
) as demo:

    # Header Section
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🐾 AI Pet Species Classifier")
        gr.Markdown(
            "Designed by **馬盛中** | Student ID: **4B1YZ001**", elem_id="student-info"
        )
        gr.Markdown("---")

    # Main Application
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Pet Image / 上傳寵物照片", type="pil")
            btn = gr.Button("Analyze / 開始辨識", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Classification Results / 辨識結果", num_top_classes=3
            )

    # Examples for a professional touch
    gr.Examples(
        examples=[],  # You can add paths to example images here if you upload them to HF
        inputs=input_img,
    )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

# 6. Launch
demo.launch()
