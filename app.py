import gradio as gr
from fastai.vision.all import *

# 1. Load the model trained on Kaggle
# Ensuring compatibility with Python 3.12 exports
learn = load_learner('pet_classifier_v1.pkl')

# Map labels to localized Traditional Chinese / English versions
label_map = {
    'cat': '貓 (Cat)',
    'dog': '狗 (Dog)',
    'goldfish': '金魚 (Goldfish)',
    'hamster': '倉鼠 (Hamster)',
    'turtle': '烏龜 (Turtle)',
    'parrot': '鸚鵡 (Parrot)',
    'snake': '蛇 (Snake)'
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
"""

with gr.Blocks(css=custom_css) as demo:
    
    # 3. Header Section
    gr.HTML(f"""
        <div class="header-box">
            <span class="student-name">馬盛中</span>
            <span class="student-id">4B1YZ001</span>
        </div>
    """)

    # 4. Main Inference Area
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Pet Photo / 上傳寵物照片", type="pil")
            btn = gr.Button("🚀 Start AI Analysis / 開始辨識", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Label(label="Classification Confidence / 辨識信賴度", num_top_classes=3)

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

    # 5. Project Documentation Area
    with gr.Accordion("📋 Project Documentation & Technical Specs", open=True):
        with gr.Row():
            with gr.Column(elem_classes="info-card"):
                gr.HTML("""
                    <div class="info-title">📖 Project Description</div>
                    <p>This deep learning application distinguishes between 7 pet species using a fine-tuned 
                    <span class="highlight">ResNet34</span> architecture. Developed as a core project for 
                    <b>STUST CSIE</b>, it demonstrates the end-to-end pipeline of data preprocessing, 
                    transfer learning, and web deployment.</p>
                """)
            
            with gr.Column(elem_classes="info-card"):
                gr.HTML("""
                    <div class="info-title">⚙️ Technical Details</div>
                    <ul>
                        <li><b>Architecture:</b> ResNet34 CNN</li>
                        <li><b>Dataset:</b> 90-Species Animal Image Set</li>
                        <li><b>Framework:</b> PyTorch & fastai</li>
                        <li><b>ID:</b> 馬盛中 (4B1YZ001)</li>
                        <li><b>Deployment:</b> Hugging Face Spaces</li>
                    </ul>
                """)

# Launch with Modern Soft Theme
demo.launch(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ssr_mode=False
)