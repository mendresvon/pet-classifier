import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner("pet_classifier_v1.pkl")

# Map labels to localized Traditional Chinese / English versions
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


# 2. Refined Premium CSS
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif !important; }
.header-box { text-align: center; padding: 50px 0; margin-bottom: 30px; border-bottom: 1px solid var(--border-color-primary); }
.student-name { 
    font-size: 3.5em !important; font-weight: 900; 
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block; letter-spacing: -1px;
}
.student-id { font-family: 'JetBrains Mono', monospace; font-size: 1.2em; color: var(--body-text-color-subdued); letter-spacing: 6px; margin-top: 5px; }

/* Supported Species Tags */
.species-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px 0; }
.species-tag { 
    padding: 6px 14px; border-radius: 20px; font-size: 0.9em; font-weight: 600;
    border: 1px solid var(--border-color-accent); background: var(--block-background-fill);
}

.info-card { background: var(--block-background-fill) !important; border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important; padding: 24px !important; }
.highlight { color: #3b82f6 !important; background: rgba(59, 130, 246, 0.12); padding: 2px 8px; border-radius: 6px; font-weight: 800; }
"""

with gr.Blocks(css=custom_css) as demo:

    # Header
    gr.HTML(
        f"""
        <div class="header-box">
            <span class="student-name">馬盛中</span>
            <span class="student-id">4B1YZ001</span>
        </div>
    """
    )

    # Main Area
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Photo / 上傳照片", type="pil")
            btn = gr.Button("🚀 Start AI Analysis / 開始辨識", variant="primary")
            gr.Examples(
                examples=["example_cat.jpg", "example_dog.jpg", "example_parrot.jpg"],
                inputs=input_img,
                label="Click an example to test / 點擊範例測試",
            )

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Classification Results / 辨識結果", num_top_classes=3
            )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

    # 3. New Supported Species Visual Section
    gr.HTML(
        """
        <div style="text-align: center; margin-top: 40px;">
            <h3 style="font-size: 1.2em; font-weight: 700;">🐾 Supported Species / 支援辨識物種</h3>
            <div class="species-container">
                <span class="species-tag">🐱 貓 (Cat)</span>
                <span class="species-tag">🐶 狗 (Dog)</span>
                <span class="species-tag">🐠 金魚 (Goldfish)</span>
                <span class="species-tag">🐹 倉鼠 (Hamster)</span>
                <span class="species-tag">🐢 烏龜 (Turtle)</span>
                <span class="species-tag">🦜 鸚鵡 (Parrot)</span>
                <span class="species-tag">🐍 蛇 (Snake)</span>
            </div>
        </div>
    """
    )

    # Documentation
    with gr.Accordion("📋 Project Documentation & Technical Specs", open=True):
        with gr.Row():
            with gr.Column(elem_classes="info-card"):
                gr.HTML(
                    """
                    <b>Project Description:</b> Distinguishes between 7 pet species using <span class="highlight">ResNet34</span>. 
                    Developed for <b>STUST CSIE</b> coursework to showcase Transfer Learning.
                """
                )
            with gr.Column(elem_classes="info-card"):
                gr.HTML(
                    "<b>Architecture:</b> ResNet34 CNN<br><b>Framework:</b> PyTorch & fastai<br><b>Developer:</b> 馬盛中 (4B1YZ001)"
                )

demo.launch(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), ssr_mode=False
)
