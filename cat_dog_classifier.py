# -------- CAT vs DOG WEB APP (Single File Python App with Image Preview & Reset) --------
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Load trained Cat/Dog model
MODEL_PATH = "cat_dog_model.keras"
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Load pre-trained detector to detect non-animals (cars, objects, etc.)
detector = MobileNetV2(weights="imagenet")


def is_cat_or_dog(pil_img):
    test_img = pil_img.resize((224, 224))
    arr = preprocess_input(np.expand_dims(np.array(test_img), axis=0))
    preds = decode_predictions(detector.predict(arr, verbose=0), top=5)[0]

    keywords = ["cat", "dog", "terrier", "retriever", "hound", "tabby", "siamese", "pug", "chihuahua", "feline",
                "canine"]
    for _, label, prob in preds:
        if any(k in label.lower() for k in keywords) and prob > 0.10:
            return True
    return False


# ---------- IMAGE PREPROCESS ----------
def prepare_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_prediction_label(raw_img):
    # Step 1: Check if the image is actually a cat or dog
    if not is_cat_or_dog(raw_img):
        return "Neither (Not a Cat or Dog) ❌"

    # Step 2: Run your 128x128 Keras model
    if model:
        processed = prepare_image(raw_img)
        score = float(model.predict(processed, verbose=0)[0][0])

        if 0.40 <= score <= 0.60:
            return f"Neither / Uncertain ❓ ({score:.4f})"
        elif score > 0.5:
            return f"Dog 🐶 ({score:.4f})"
        else:
            return f"Cat 🐱 ({score:.4f})"
    else:
        return "Model file 'cat_dog_model.keras' not found."


# Helper to convert PIL Image to base64 for display in browser
def image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ---------- EMBEDDED HTML TEMPLATE ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cat vs Dog Web App</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            background: #f7f7f8; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            margin: 0; 
            padding: 20px;
            box-sizing: border-box;
        }
        .card { 
            background: white; 
            padding: 32px; 
            border-radius: 16px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
            width: 100%; 
            max-width: 460px; 
            text-align: center; 
        }
        h2 { 
            margin-top: 0; 
            margin-bottom: 20px; 
            color: #222; 
            font-size: 22px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 8px; 
        }
        .input-group { 
            margin: 15px 0; 
            text-align: left; 
        }
        label { 
            font-size: 13px; 
            font-weight: 600; 
            color: #444; 
            display: block; 
            margin-bottom: 6px; 
        }
        input[type="file"], input[type="text"] { 
            width: 100%; 
            padding: 10px 12px; 
            border: 1px solid #d1d5db; 
            border-radius: 8px; 
            box-sizing: border-box; 
            font-size: 14px;
            background: #fafafa;
        }
        input[type="file"]:focus, input[type="text"]:focus {
            border-color: #d97706;
            outline: none;
            background: #fff;
        }
        .or { 
            margin: 12px 0; 
            color: #9ca3af; 
            font-size: 12px; 
            font-weight: bold; 
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 18px;
        }
        button.btn-predict { 
            flex: 1; 
            padding: 12px; 
            background: #d97706; 
            color: white; 
            border: none; 
            border-radius: 8px; 
            font-size: 15px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: background 0.2s;
        }
        button.btn-predict:hover { 
            background: #b45309; 
        }
        a.btn-reset { 
            padding: 12px 18px; 
            background: #f3f4f6; 
            color: #4b5563; 
            text-decoration: none; 
            border-radius: 8px; 
            font-size: 15px; 
            font-weight: 600; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            border: 1px solid #e5e7eb;
            transition: background 0.2s;
        }
        a.btn-reset:hover { 
            background: #e5e7eb; 
            color: #111827; 
        }
        .preview-container {
            margin-top: 20px;
            padding: 10px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
        }
        .preview-container img {
            max-width: 100%;
            max-height: 240px;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .result { 
            margin-top: 16px; 
            padding: 16px; 
            border-radius: 10px; 
            font-size: 18px; 
            font-weight: bold; 
        }
        .result-dog {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e40af;
        }
        .result-cat {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }
        .result-neither {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>🐾 Cat vs Dog Classifier</h2>
        <form method="POST" enctype="multipart/form-data">
            <div class="input-group">
                <label>1. Upload Image:</label>
                <input type="file" name="file" accept="image/*">
            </div>

            <div class="or">OR</div>

            <div class="input-group">
                <label>2. Image URL:</label>
                <input type="text" name="url" placeholder="https://example.com/image.jpg">
            </div>

            <div class="btn-group">
                <button type="submit" class="btn-predict">Predict</button>
                {% if result or img_data %}
                <a href="/" class="btn-reset">🔄 Reset</a>
                {% endif %}
            </div>
        </form>

        {% if img_data %}
        <div class="preview-container">
            <img src="data:image/jpeg;base64,{{ img_data }}" alt="Analyzed Image">
        </div>
        {% endif %}

        {% if result %}
        <div class="result {% if 'Dog' in result %}result-dog{% elif 'Cat' in result %}result-cat{% else %}result-neither{% endif %}">
            {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


# ---------- MAIN ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    img_data = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            try:
                raw_img = Image.open(file).convert("RGB")
                img_data = image_to_base64(raw_img)
                result = get_prediction_label(raw_img)
            except Exception as e:
                result = f"Error: {str(e)}"

        url = request.form.get("url")
        if not result and url:
            try:
                res = requests.get(url, timeout=10)
                raw_img = Image.open(BytesIO(res.content)).convert("RGB")
                img_data = image_to_base64(raw_img)
                result = get_prediction_label(raw_img)
            except Exception:
                result = "Invalid URL or unable to fetch image!"

    return render_template_string(HTML_TEMPLATE, result=result, img_data=img_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)