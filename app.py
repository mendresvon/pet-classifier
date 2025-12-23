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


# 2. Premium Professional CSS
custom_css = """
/* Adaptive Background & Text */
.gradio-container { font-family: 'Inter', -apple-system, sans-serif !important; }

/* Header Section */
.header-box {
    text-align: center;
    padding: 50px 0;
    margin-bottom: 30px;
    border-bottom: 1px solid var(--border-color-primary);
}

.student-name {
    font-size: 3.5em !important;
    font-weight: 900;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
    letter-spacing: -1px;
}

.student-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2em;
    color: var(--body-text-color-subdued);
    letter-spacing: 6px;
    margin-top: 5px;
    font-weight: 600;
}

/* Supported Species Tags */
.species-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px 0; }
.species-tag { 
    padding: 6px 14px; border-radius: 20px; font-size: 0.9em; font-weight: 600;
    border: 1px solid var(--border-color-accent); background: var(--block-background-fill);
}

/* Info Section Styling */
.info-card {
    background: var(--block-background-fill) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: var(--block-shadow);
}
.info-title {
    font-size: 1.4em; font-weight: 800; color: var(--body-text-color);
    margin-bottom: 15px; display: flex; align-items: center; gap: 10px;
}

.highlight { color: #3b82f6 !important; background: rgba(59, 130, 246, 0.12); padding: 2px 8px; border-radius: 6px; font-weight: 800; }
b, strong { color: var(--body-text-color) !important; font-weight: 700; }
ul { list-style-type: none; padding-left: 0; }
li { margin-bottom: 8px; color: var(--body-text-color-subdued); }
p { line-height: 1.6; margin-bottom: 1em; color: var(--body-text-color-subdued); }
.divider { margin: 20px 0; border-top: 1px dashed var(--border-color-primary); }
"""

with gr.Blocks(css=custom_css) as demo:

    # 3. Header
    gr.HTML(
        f"""
        <div class="header-box">
            <span class="student-name">馬盛中</span>
            <span class="student-id">4B1YZ001</span>
        </div>
    """
    )

    # 4. Main Interaction Area
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Pet Photo / 上傳寵物照片", type="pil")
            btn = gr.Button("🚀 Start AI Analysis / 開始辨識", variant="primary")

            # --- RESTORED EXAMPLE GALLERY ---
            gr.Examples(
                examples=["example_cat.jpg", "example_dog.jpg", "example_parrot.jpg"],
                inputs=input_img,
                label="Click an example to test / 點擊範例測試",
            )
            # --------------------------------

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Classification Confidence / 辨識信賴度", num_top_classes=3
            )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

    # 5. Supported Species Visuals
    gr.HTML(
        """
        <div style="text-align: center; margin-top: 40px;">
            <h3 style="font-size: 1.2em; font-weight: 700; margin-bottom: 10px;">🐾 Supported Species / 支援辨識物種</h3>
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

    # 6. Documentation (Bilingual)
    with gr.Accordion("📋 Project Documentation & Technical Specs", open=True):
        with gr.Row():
            with gr.Column(elem_classes="info-card"):
                gr.HTML(
                    """
                    <div class="info-title">📖 Project Description</div>
                    
                    <p>
                        This deep learning application distinguishes between 7 pet species. Rather than building a CNN from scratch, we leveraged <b>transfer learning</b> by <b>fine-tuning</b> a pre-trained <span class="highlight">ResNet34</span> architecture.
                    </p>
                    <p>
                        Before training, the baseline accuracy was <b>~76%</b>. After applying <b>data augmentation</b> and retraining, we achieved <b>98%</b> accuracy on the validation set.
                    </p>

                    <div class="divider"></div>

                    <p>
                        本深度學習應用程式可辨識 7 種常見寵物。我們利用<b>遷移學習 (Transfer Learning)</b> 技術，對預訓練的 <span class="highlight">ResNet34</span> 架構進行<b>微調 (Fine-tuning)</b>。
                    </p>
                    <p>
                        基準模型準確率約為 <b>76%</b>。透過<b>資料增強 (Data Augmentation)</b> 及再訓練，最終在驗證集上達到 <b>98%</b> 的高準確率。
                    </p>
                """
                )

            with gr.Column(elem_classes="info-card"):
                gr.HTML(
                    """
                    <div class="info-title">⚙️ Technical Details</div>
                    <ul>
                        <li><b>Architecture:</b> ResNet34 CNN</li>
                        <li><b>Technique:</b> Transfer Learning (Fine-Tuning)</li>
                        <li><b>Base Accuracy:</b> ~76% (Pre-training)</li>
                        <li><b>Final Accuracy:</b> 98% (Post-training)</li>
                        <li><b>Framework:</b> PyTorch & fastai</li>
                        <li><b>ID:</b> 馬盛中 (4B1YZ001)</li>
                    </ul>
                """
                )

# Launch
demo.launch(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), ssr_mode=False
)
