import gradio as gr
from fastai.vision.all import *

# 1. Load the model trained on Kaggle
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


# 2. Premium Professional CSS (System-Aware & High Contrast)
custom_css = """
/* Adaptive Background & Text */
.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Header Section - Clean & Modern */
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

/* Info Section Styling (Documentation Style) */
.info-card {
    background: var(--block-background-fill) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: var(--block-shadow);
}

.info-title {
    font-size: 1.4em;
    font-weight: 800;
    color: var(--body-text-color);
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* High-Contrast Technical Highlighting */
.highlight {
    color: #3b82f6 !important;
    background: rgba(59, 130, 246, 0.12);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 800;
}

b, strong { color: var(--body-text-color) !important; font-weight: 700; }
ul { list-style-type: none; padding-left: 0; }
li { margin-bottom: 8px; color: var(--body-text-color-subdued); }
li b { color: var(--body-text-color) !important; }
p { line-height: 1.6; margin-bottom: 1em; color: var(--body-text-color-subdued); }
.divider { margin: 20px 0; border-top: 1px dashed var(--border-color-primary); }
"""

with gr.Blocks(css=custom_css) as demo:

    # 3. Header Section
    gr.HTML(
        f"""
        <div class="header-box">
            <span class="student-name">馬盛中</span>
            <span class="student-id">4B1YZ001</span>
        </div>
    """
    )

    # 4. Main Inference Area
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Pet Photo / 上傳寵物照片", type="pil")
            btn = gr.Button("🚀 Start AI Analysis / 開始辨識", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Classification Confidence / 辨識信賴度", num_top_classes=3
            )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

    # 5. Project Documentation Area
    with gr.Accordion("📋 Project Documentation & Technical Specs", open=True):
        with gr.Row():
            with gr.Column(elem_classes="info-card"):
                gr.HTML(
                    """
                    <div class="info-title">📖 Project Description</div>
                    
                    <p>
                        This deep learning application distinguishes between 7 pet species. Rather than building a Convolutional Neural Network (CNN) from scratch, we leveraged the power of <b>transfer learning</b> by <b>fine-tuning</b> a pre-trained <span class="highlight">ResNet34</span> architecture.
                    </p>
                    <p>
                        Before our specific training, the baseline model achieved an accuracy of approximately <b>77%</b>. Through a rigorous process of training, applying <b>data augmentation</b>, and retraining, we significantly boosted the model's performance, ultimately achieving an impressive accuracy of <b>98%</b> on our validation set.
                    </p>
                    <p>
                        Developed as a core project for <b>STUST CSIE</b>, this application demonstrates the end-to-end pipeline of modern deep learning development.
                    </p>

                    <div class="divider"></div>

                    <p>
                        本深度學習應用程式可辨識 7 種常見寵物。我們並非從零開始建立卷積神經網路 (CNN)，而是利用<b>遷移學習 (Transfer Learning)</b> 技術，對預訓練的 <span class="highlight">ResNet34</span> 架構進行<b>微調 (Fine-tuning)</b>。
                    </p>
                    <p>
                        在進行特定訓練前，基準模型的準確率約為 <b>77%</b>。透過嚴格的訓練、<b>資料增強 (Data Augmentation)</b> 及再訓練流程，我們成功將模型效能大幅提升，最終在驗證資料集上達到 <b>98%</b> 的高準確率。
                    </p>
                    <p>
                        本專案為 <b>南台科技大學 資訊工程系 (STUST CSIE)</b> 之核心實作，完整展示了現代深度學習的開發流程。
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
                        <li><b>Base Accuracy:</b> ~77% (Pre-training)</li>
                        <li><b>Final Accuracy:</b> 98% (Post-training)</li>
                        <li><b>Framework:</b> PyTorch & fastai</li>
                        <li><b>ID:</b> 馬盛中 (4B1YZ001)</li>
                    </ul>
                """
                )

# Launch with Modern Soft Theme
demo.launch(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), ssr_mode=False
)
