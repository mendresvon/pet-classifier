import gradio as gr
from fastai.vision.all import *

# Load the model
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


# Custom CSS for the "Premium" feel
custom_css = """
.gradio-container { background-color: #0c0c0e !important; }
.header-box { 
    text-align: center; 
    padding: 30px; 
    background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0) 100%);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 30px;
}
.student-name { color: #ffffff; font-size: 2.2em; font-weight: 700; margin-bottom: 5px; }
.student-id { color: #d4d4d8; font-family: 'Courier New', monospace; letter-spacing: 2px; font-size: 1.1em; }
.project-desc { color: #a1a1aa; max-width: 700px; margin: 20px auto; line-height: 1.6; font-size: 1.05em; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="zinc", neutral_hue="zinc"), css=custom_css
) as demo:

    # Premium Header Section
    gr.HTML(
        f"""
        <div class="header-box">
            <div class="student-name">馬盛中</div>
            <div class="student-id">STUDENT ID: 4B1YZ001</div>
            <div class="project-desc">
                本計畫採用 <b>ResNet34</b> 深度學習架構，針對七種常見家庭寵物進行精準物種辨識。
                透過遷移學習 (Transfer Learning) 技術，系統能夠在複雜背景下準確區分貓、狗、金魚、倉鼠、烏龜、鸚鵡及蛇類。
                <br><br>
                <i>This AI system utilizes ResNet34 to classify 7 pet species with high precision, 
                showcasing the power of modern computer vision in domestic animal identification.</i>
            </div>
        </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Pet Image / 上傳照片", type="pil")
            btn = gr.Button("Analyze Species / 開始辨識", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Top Predictions / 辨識結果", num_top_classes=3
            )

    btn.click(fn=predict, inputs=input_img, outputs=output_label)

demo.launch()
