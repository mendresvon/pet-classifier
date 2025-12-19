import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner("pet_classifier_v1.pkl")

label_map = {
    "cat": "貓 (Cat)",
    "dog": "狗 (Dog)",
    "goldfish": "金魚 (Goldfish)",
    "hamster": "倉鼠 (Hamster)",
    "turtle": "烏龜 (Turtle)",
    "parrot": "鸚鵡 (Parrot)",
    "snake": "蛇 (Snake)",
}


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {label_map[c]: float(probs[i]) for i, c in enumerate(learn.dls.vocab)}


# 2. Cyber-Neon High-Contrast CSS
custom_css = """
.gradio-container { background: radial-gradient(circle at top, #0f172a, #020617) !important; color: #f8fafc !important; }

/* Header Styling */
.header-box { 
    text-align: center; padding: 60px 20px; border-radius: 24px;
    background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px);
    border: 1px solid rgba(34, 211, 238, 0.2); margin-bottom: 30px;
}
.student-name { 
    color: #ffffff !important; font-size: 3.5em !important; font-weight: 900; 
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.6); display: block;
}
.student-id { color: #22d3ee !important; font-family: monospace; font-size: 1.4em; letter-spacing: 5px; }

/* Project Info Section Styling */
.info-section { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(34, 211, 238, 0.3); border-radius: 15px; padding: 25px; }
.info-title { color: #22d3ee !important; font-size: 1.5em; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid rgba(34, 211, 238, 0.2); }
.tech-list { list-style-type: square; margin-left: 20px; color: #cbd5e1; }
.highlight { color: #d946ef !important; font-weight: bold; text-shadow: 0 0 8px rgba(217, 70, 239, 0.4); }
b, strong { color: #ffffff !important; font-weight: 700; }
"""

with gr.Blocks(css=custom_css) as demo:

    # Visual Header
    gr.HTML(
        f"""
        <div class="header-box">
            <span class="student-name">馬盛中</span>
            <span class="student-id">4B1YZ001</span>
        </div>
    """
    )

    # Main Interaction Row
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image / 上傳照片", type="pil")
            btn = gr.Button("⚡ INFER SPECIES / 開始辨識", variant="primary")

        with gr.Column():
            output_label = gr.Label(
                label="Top Classifications / 辨識結果", num_top_classes=3
            )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

    # Project Description & Technical Info (Accordion for that clean feel)
    with gr.Accordion("ℹ️ About this Project & Technical Info", open=True):
        with gr.Row():
            with gr.Column(elem_classes="info-section"):
                gr.HTML(
                    """
                    <div class="info-title">Project Description</div>
                    <p>This project showcases a Computer Vision model that correctly distinguishes between 7 common pet species. 
                    The model was created by fine-tuning a <span class="highlight">ResNet34</span> architecture, achieving high accuracy through Transfer Learning. 
                    This application is built with Python and deployed as an interactive web app for <b>STUST CSIE</b>.</p>
                """
                )
            with gr.Column(elem_classes="info-section"):
                gr.HTML(
                    """
                    <div class="info-title">Technical Details</div>
                    <ul class="tech-list">
                        <li><b>Project Type:</b> Image Classification</li>
                        <li><b>Architecture:</b> ResNet34 (Fine-Tuned)</li>
                        <li><b>Core Framework:</b> PyTorch & fastai</li>
                        <li><b>Interface:</b> Gradio UI</li>
                        <li><b>Developer:</b> 馬盛中 (4B1YZ001)</li>
                    </ul>
                """
                )

# Launch settings for Gradio 6.0 and HF Spaces
demo.launch(ssr_mode=False)
