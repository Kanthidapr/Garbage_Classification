"""Garbage Classification — polished Gradio UI."""

import html
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "trash_class_cnn_mvp.keras"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"
EXAMPLES_DIR = BASE_DIR / "examples"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์โมเดล: {MODEL_PATH}\n"
        "กรุณาวาง trash_class_cnn_mvp.keras ไว้ในโฟลเดอร์ model/"
    )
if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์ metadata: {METADATA_PATH}\n"
        "กรุณาวาง metadata.json ไว้ในโฟลเดอร์ model/"
    )

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

classes = metadata.get("classes", [])
IMG_SIZE = int(metadata.get("image_size", 64))
test_accuracy = metadata.get("test_accuracy")
macro_f1 = metadata.get("macro_f1")

if not classes:
    raise ValueError("ไม่พบรายการ classes ใน model/metadata.json")

model = tf.keras.models.load_model(MODEL_PATH)


def pct(value):
    try:
        return f"{float(value) * 100:.1f}%" if value is not None else "—"
    except (TypeError, ValueError):
        return "—"


def status_card(kind, title, message):
    icons = {"success": "✓", "error": "!", "info": "✦"}
    return f"""
    <div class="status-card {kind}">
      <span class="status-icon">{icons[kind]}</span>
      <div><div class="status-title">{html.escape(title)}</div>
      <div class="status-message">{message}</div></div>
    </div>
    """


def render_top_predictions(top):
    if not top:
        return '<div class="top-predictions-empty">Your top 5 predictions will appear here after you upload an image.</div>'
    rows = []
    for rank, (label, probability) in enumerate(top.items(), start=1):
        confidence = max(0.0, min(100.0, float(probability) * 100))
        rows.append(f'''<div class="prediction-row">
          <div class="prediction-row-head"><span class="prediction-rank">{rank:02d}</span><span class="prediction-name">{html.escape(str(label))}</span><strong class="prediction-percent">{confidence:.2f}%</strong></div>
          <div class="prediction-track"><div class="prediction-fill" style="width:{confidence:.2f}%"></div></div>
        </div>''')
    return '<div class="top-predictions">' + ''.join(rows) + '</div>'


def predict(image, actual_class):
    if image is None:
        return render_top_predictions({}), "", status_card("info", "Ready when you are", "Upload a photo of an item to get started."), ""

    try:
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        probabilities = model.predict(x, verbose=0)[0]

        if len(probabilities) != len(classes):
            raise ValueError(
                f"Model output has {len(probabilities)} classes, but metadata lists {len(classes)}."
            )

        order = np.argsort(probabilities)[::-1][:5]
        top = {classes[int(i)]: float(probabilities[int(i)]) for i in order}
        top_html = render_top_predictions(top)
        predicted = classes[int(order[0])]
        confidence = float(probabilities[int(order[0])]) * 100

        safe_predicted = html.escape(str(predicted))
        if not actual_class:
            result = status_card(
                "info", "Prediction complete",
                "Choose the actual class if you want to check whether the prediction is correct."
            )
        elif predicted == actual_class:
            result = status_card(
                "success", "That's a match!",
                f"Actual: <b>{html.escape(str(actual_class))}</b>"
                f"<span class='status-separator'>/</span> Predicted: <b>{safe_predicted}</b>"
            )
        else:
            result = status_card(
                "error", "Not quite this time",
                f"Actual: <b>{html.escape(str(actual_class))}</b>"
                f"<span class='status-separator'>/</span> Predicted: <b>{safe_predicted}</b>"
            )

        hero = f"""
        <div class="result-hero">
          <div class="result-eyebrow"><span class="pulse-dot"></span> MODEL PREDICTION</div>
          <div class="result-class">{safe_predicted}</div>
          <div class="result-confidence">Confidence <strong>{confidence:.2f}%</strong></div>
          <div class="confidence-track"><div class="confidence-fill" style="width:{max(0, min(100, confidence)):.2f}%"></div></div>
        </div>
        """
        return top_html, hero, result, f"{confidence:.2f}%"

    except Exception as exc:
        return render_top_predictions({}), "", status_card("error", "Prediction failed", html.escape(str(exc))), ""


css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { --ink:#172b27; --muted:#73827d; --green:#176b52; --mint:#dff6e9; --line:#e4eee8; }
body, .gradio-container { font-family:'DM Sans','Plus Jakarta Sans',sans-serif !important; }
body, .gradio-container, .gradio-container * { color-scheme: light; }
body { background:#f3f7f3 !important; }
.gradio-container { --body-background-fill:#f3f7f3 !important; --background-fill-primary:#ffffff !important; --background-fill-secondary:#f7faf7 !important; --block-background-fill:#ffffff !important; --block-border-color:#e4eee8 !important; --input-background-fill:#ffffff !important; --panel-background-fill:#ffffff !important; }
.gradio-container { max-width:1180px !important; margin:0 auto !important; background:transparent !important; }
.main-shell { padding:22px 0 8px; }
.hero-banner { position:relative; overflow:hidden; background:#153f34; border-radius:28px; padding:38px 42px; color:white; }
.hero-banner:after { content:'♻'; position:absolute; right:28px; top:-48px; font-size:230px; line-height:1; color:rgba(255,255,255,.055); transform:rotate(-15deg); }
.eyebrow { display:inline-flex; align-items:center; gap:8px; color:#b8f0d1; text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:2.2px; }
.eyebrow-dot { width:7px; height:7px; background:#7ce3a7; border-radius:50%; }
.hero-title { position:relative; z-index:1; margin:14px 0 8px; font-size:clamp(30px,4vw,46px); line-height:1.1; font-weight:800; letter-spacing:-1.5px; color:white; }
.hero-copy { position:relative; z-index:1; max-width:600px; color:#d0e4db; font-size:15px; line-height:1.75; }
.hero-tags { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
.hero-tag { border:1px solid rgba(255,255,255,.2); background:rgba(255,255,255,.08); color:#edf9f2; border-radius:999px; padding:7px 12px; font-size:12px; }
.stats-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:13px; margin:17px 0 26px; }
.stat-card { background:white; border:1px solid var(--line); border-radius:18px; padding:19px 20px; box-shadow:0 5px 18px rgba(26,65,49,.035); }
.stat-top { color:var(--muted); font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:1.15px; }
.stat-value { color:var(--ink); font-size:29px; line-height:1.25; font-weight:800; letter-spacing:-.7px; margin-top:9px; }
.stat-foot { color:#94a39d; font-size:12px; margin-top:4px; }
.section-heading { color:var(--ink); font-weight:800; font-size:20px; letter-spacing:-.4px; margin-bottom:4px; }
.section-caption { color:var(--muted); font-size:13px; line-height:1.6; margin-bottom:15px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:22px; padding:22px !important; box-shadow:0 7px 24px rgba(26,65,49,.035); }
.panel h3 { color:var(--ink); }
.upload-note { background:#f3f8f4; border:1px dashed #c9dbcf; border-radius:14px; padding:12px 14px; color:#687a70; font-size:12px; line-height:1.6; }
.result-hero { background:linear-gradient(145deg,#e7f8ee,#f5fbf6); border:1px solid #d6efdf; border-radius:18px; padding:23px; margin:7px 0 14px; }
.result-eyebrow { color:#518168; font-size:10px; letter-spacing:1.7px; font-weight:800; display:flex; align-items:center; gap:8px; }
.pulse-dot { display:inline-block; width:7px; height:7px; background:#26a269; border-radius:50%; }
.result-class { color:#174b37; font-size:clamp(25px,3vw,35px); line-height:1.2; font-weight:800; letter-spacing:-.7px; margin:13px 0 9px; overflow-wrap:anywhere; }
.result-confidence { color:#668173; font-size:13px; }
.result-confidence strong { color:#176b52; font-size:18px; margin-left:5px; }
.confidence-track { height:7px; background:#d3e9da; border-radius:99px; overflow:hidden; margin-top:14px; }
.confidence-fill { height:100%; background:linear-gradient(90deg,#299765,#8bd6a6); border-radius:99px; }
.status-card { display:flex; gap:12px; align-items:flex-start; padding:15px; border-radius:15px; margin:12px 0; }
.status-card.success { background:#ecf9f1; border:1px solid #ccebd8; }
.status-card.error { background:#fff3f0; border:1px solid #f5d8d0; }
.status-card.info { background:#f0f6f3; border:1px solid #e0ebe4; }
.status-icon { display:flex; align-items:center; justify-content:center; flex-shrink:0; width:29px; height:29px; border-radius:10px; background:white; color:#287957; font-weight:800; }
.status-card.error .status-icon { color:#bf624e; }
.status-title { color:#254538; font-size:14px; font-weight:800; }
.status-message { color:#6d8076; font-size:12px; line-height:1.65; margin-top:3px; overflow-wrap:anywhere; }
.status-separator { padding:0 6px; color:#9aaca1; }
.top-predictions { display:flex; flex-direction:column; gap:17px; padding:8px 2px 4px; }
.prediction-row { width:100%; }
.prediction-row-head { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.prediction-rank { display:flex; align-items:center; justify-content:center; width:27px; height:27px; flex:0 0 27px; border-radius:8px; background:#e7f5ec; color:#176b52 !important; font-size:11px; font-weight:800; }
.prediction-name { color:#173c2f !important; font-size:14px; font-weight:700; overflow-wrap:anywhere; flex:1; }
.prediction-percent { color:#176b52 !important; font-size:14px; font-weight:800; font-variant-numeric:tabular-nums; }
.prediction-track { width:100%; height:9px; background:#e5eee8; border-radius:999px; overflow:hidden; }
.prediction-fill { height:100%; min-width:2px; background:linear-gradient(90deg,#16805b,#51c58a); border-radius:999px; }
.top-predictions-empty { padding:22px 18px; background:#f5f9f6; border:1px dashed #c8dbcf; border-radius:14px; color:#64786b !important; font-size:13px; line-height:1.6; }
button { border-radius:12px !important; font-weight:700 !important; }
.gradio-container .wrap, .gradio-container .block, .gradio-container .form, .gradio-container [data-testid="image"] { background:#ffffff !important; border-color:#e4eee8 !important; }
.gradio-container input, .gradio-container textarea { background:#ffffff !important; color:#172b27 !important; }
.gradio-container .label, .gradio-container label, .gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container h3 { color:#172b27; }
.primary { background:#176b52 !important; }
input, textarea, .wrap, .form { border-radius:12px !important; }
.gradio-container label { color:#344b40 !important; font-weight:700 !important; }
.model-note { background:#edf4ef; border-radius:16px; padding:18px 20px; color:#5e7467; font-size:13px; line-height:1.8; }
.footer { text-align:center; color:#91a098; font-size:12px; padding:28px 10px 18px; }
@media (max-width:760px) {
 .hero-banner { padding:28px 23px; border-radius:22px; }
 .hero-banner:after { right:-8px; top:-20px; font-size:140px; }
 .stats-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
 .stat-card { padding:15px; }
 .stat-value { font-size:25px; }
 .panel { padding:15px !important; }
}

/* High-contrast fixes: keep every label and dropdown option readable. */
.gradio-container,
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container label,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container span,
.gradio-container small,
.gradio-container [data-testid="block-info"] {
  color: #18382d !important;
}
.gradio-container .hero-banner,
.gradio-container .hero-banner *,
.gradio-container .hero-banner .eyebrow,
.gradio-container .hero-banner .hero-title,
.gradio-container .hero-banner .hero-copy,
.gradio-container .hero-banner .hero-tag {
  color: #ffffff !important;
}
.gradio-container .hero-banner .eyebrow { color: #b8f0d1 !important; }
.gradio-container .hero-banner .hero-copy { color: #e0eee7 !important; }
.gradio-container .hero-banner .hero-tag { color: #f4fff8 !important; }
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="combobox"],
.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container [data-testid="dropdown"] *,
.gradio-container .wrap input {
  color: #18382d !important;
  -webkit-text-fill-color: #18382d !important;
  background-color: #ffffff !important;
  opacity: 1 !important;
}
.gradio-container [role="option"] { padding: 10px 14px !important; }
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
  background: #e3f5e9 !important;
  color: #124c38 !important;
}
.gradio-container button,
.gradio-container button span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
.gradio-container button.secondary,
.gradio-container button.secondary span {
  color: #176b52 !important;
  -webkit-text-fill-color: #176b52 !important;
}
.gradio-container .result-hero,
.gradio-container .result-hero * { color: #174b37 !important; }
.gradio-container .status-card,
.gradio-container .status-card * { color: #254538 !important; }
.gradio-container .stat-card,
.gradio-container .stat-card * { color: #172b27 !important; }
.gradio-container .upload-note { color: #425f50 !important; }
.gradio-container .model-note { color: #355646 !important; }
.gradio-container .footer { color: #61766a !important; }
"""

example_rows = []
if EXAMPLES_DIR.exists():
    for path in sorted(EXAMPLES_DIR.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            actual = path.stem.rsplit("_", 1)[0]
            if actual in classes:
                example_rows.append([str(path), actual])

kwargs = {
    "title": "Garbage Classification | CNN",
    "theme": gr.themes.Soft(primary_hue="emerald", secondary_hue="green", neutral_hue="slate"),
}
launch_kwargs = {}
if int(gr.__version__.split(".")[0]) >= 6:
    launch_kwargs["css"] = css
else:
    kwargs["css"] = css

with gr.Blocks(**kwargs) as demo:
    gr.HTML(f"""
    <div class="main-shell">
      <div class="hero-banner">
        <div class="eyebrow"><span class="eyebrow-dot"></span> MACHINE LEARNING PROJECT</div>
        <div class="hero-title">Waste, meet your<br>smart classifier.</div>
        <div class="hero-copy">An image-powered CNN that helps identify waste categories in seconds.
        Upload a photo, run a prediction, and explore the model's top results.</div>
        <div class="hero-tags">
          <span class="hero-tag">✦ CNN powered</span>
          <span class="hero-tag">♻ {len(classes)} waste categories</span>
          <span class="hero-tag">▣ {IMG_SIZE} × {IMG_SIZE} input</span>
        </div>
      </div>
    </div>
    """)

    gr.HTML(f"""
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-top">Test accuracy</div><div class="stat-value">{pct(test_accuracy)}</div><div class="stat-foot">Held-out evaluation set</div></div>
      <div class="stat-card"><div class="stat-top">Macro F1 score</div><div class="stat-value">{pct(macro_f1)}</div><div class="stat-foot">Across all categories</div></div>
      <div class="stat-card"><div class="stat-top">Waste categories</div><div class="stat-value">{len(classes)}</div><div class="stat-foot">Classes in this model</div></div>
      <div class="stat-card"><div class="stat-top">Image input</div><div class="stat-value">{IMG_SIZE}<span style="font-size:15px;color:#82968a"> × {IMG_SIZE}</span></div><div class="stat-foot">Pixels processed by CNN</div></div>
    </div>
    """)

    gr.HTML("""
    <div class="section-heading">Try the classifier</div>
    <div class="section-caption">Upload an image to see the most likely waste category and the model's confidence.</div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("### 01 / Add an image")
            image_input = gr.Image(type="pil", label="Drop a waste image here", height=300)
            gr.HTML('<div class="upload-note">Tip: Use a clear, well-lit photo with one main object in focus for a more useful prediction.</div>')
            actual_class = gr.Dropdown(
                choices=classes, label="Actual category (optional)",
                info="Select the known category to compare it with the model prediction.",
                value=None, allow_custom_value=False,
            )
            predict_button = gr.Button("Run prediction  →", variant="primary", size="lg")

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("### 02 / Prediction")
            hero_output = gr.HTML()
            status_output = gr.HTML()
            gr.Markdown("#### Top 5 possible categories")
            prediction_output = gr.HTML(render_top_predictions({}))
            confidence_output = gr.Textbox(label="Top prediction confidence", interactive=False)

    predict_button.click(
        fn=predict,
        inputs=[image_input, actual_class],
        outputs=[prediction_output, hero_output, status_output, confidence_output],
    )

    if example_rows:
        gr.HTML('<div class="section-heading" style="margin-top:28px">Need a sample?</div><div class="section-caption">Choose a sample image to prefill the image and its known category.</div>')
        gr.Examples(examples=example_rows, inputs=[image_input, actual_class], label="Sample images")

    gr.HTML("""
    <div class="section-heading" style="margin-top:26px">About this model</div>
    <div class="model-note">
      <b>Architecture:</b> Convolutional Neural Network (CNN)<br>
      <b>Preprocessing:</b> RGB image resized to the model input dimensions and normalized to the 0–1 range.<br>
      <b>Evaluation:</b> Accuracy and Macro F1 shown above are read from the project's metadata file.<br>
      <b>Keep in mind:</b> Confidence is the model's score, not a guarantee. Similar-looking materials, poor lighting, or images unlike the training data can affect results.
    </div>
    """)
    gr.HTML('<div class="footer">♻ Garbage Classification CNN &nbsp;·&nbsp; Built for a cleaner tomorrow</div>')

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        **launch_kwargs,
    )
