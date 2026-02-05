from flask import Flask, render_template, request, url_for, jsonify
import os, cv2, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
CLASS_NAMES = ["Normal", "Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D"]
# =============================
# APP CONFIG
# =============================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static/heatmaps")
REPORT_FOLDER = os.path.join(BASE_DIR, "static/reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
def build_efficientnet():

    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(5, activation="softmax")(x)

    return Model(base_model.input, output)
# =============================
# LOAD MODELS
# =============================
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
cnn = None
vgg = None
resnet = None
mobilenet = None
efficientnet = None

def load_models():
    global cnn, vgg, resnet, mobilenet, efficientnet

    if cnn is None:
        print("Loading models...")

        cnn = load_model(os.path.join(MODEL_FOLDER, "cnn_model.h5"))
        vgg = load_model(os.path.join(MODEL_FOLDER, "vgg16_model.h5"))
        resnet = load_model(os.path.join(MODEL_FOLDER, "resnet50_model.h5"))
        mobilenet = load_model(os.path.join(MODEL_FOLDER, "mobilenet_model.h5"))

        efficientnet = build_efficientnet()
        efficientnet.load_weights(os.path.join(MODEL_FOLDER, "efficientnet_weights.h5"))

        print("Models Loaded Successfully")
# =============================
# LANGUAGE FILES
# =============================
def load_language(lang):
    try:
        with open(f"languages/{lang}.json", encoding="utf-8") as f:
            return json.load(f)
    except:
        with open("languages/en.json", encoding="utf-8") as f:
            return json.load(f)

# =============================
# DOCTOR DATABASE (REAL CONTACT)
# =============================
DOCTOR_DATABASE = {
    "Vitamin A": {
        "specialist": "Ophthalmologist",
        "doctor_name": "Dr. Aristhotra Sharma",
        "description": "Eye dryness & retinal health specialist.",
        "availability": "Online",
        "contact_link": "https://wa.me/919999999999"
    },
    "Vitamin B": {
        "specialist": "Neurologist",
        "doctor_name": "Dr. Sarah Mitchell",
        "description": "B-complex nerve & metabolic disorders.",
        "availability": "Online",
        "contact_link": "mailto:mitchell.neuro@clinic.com"
    },
    "Vitamin C": {
        "specialist": "Dermatologist",
        "doctor_name": "Dr. Elena Rossi",
        "description": "Skin inflammation & collagen disorders.",
        "availability": "Online",
        "contact_link": "https://wa.me/918888888888"
    },
    "Vitamin D": {
        "specialist": "Orthopedic Surgeon",
        "doctor_name": "Dr. James Wilson",
        "description": "Bone density & hormonal regulation.",
        "availability": "Online",
        "contact_link": "mailto:wilson.ortho@clinic.com"
    },
    "Normal": {
        "specialist": "General Physician",
        "doctor_name": "Dr. Aman Varma",
        "description": "Preventive health consultation.",
        "availability": "Online",
        "contact_link": "mailto:gp@clinic.com"
    }
}

# =============================
# DIET MAP
# =============================
DIET_MAP = {
    "Vitamin A": {"foods": ["Carrot", "Spinach", "Sweet Potato", "Milk"], "advice": "Improves vision and immunity."},
    "Vitamin B": {"foods": ["Eggs", "Milk", "Fish", "Banana"], "advice": "Supports nerve function."},
    "Vitamin C": {"foods": ["Orange", "Guava", "Lemon", "Tomato"], "advice": "Boosts immunity & skin health."},
    "Vitamin D": {"foods": ["Sunlight", "Mushroom", "Fish Oil"], "advice": "Essential for bone strength."}
}

# =============================
# IMAGE HELPERS
# =============================
def preprocess_image(path):
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    return np.expand_dims(img, axis=0)

def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < 100

def is_low_light(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < 40
# =============================
# ENSEMBLE PREDICTION (STABILIZED)
# =============================
def ensemble_predict(img):
    load_models()
    preds = np.mean([
        cnn.predict(img, verbose=0),
        vgg.predict(img, verbose=0),
        resnet.predict(img, verbose=0),
        mobilenet.predict(img, verbose=0),
        efficientnet.predict(img, verbose=0)
    ], axis=0)[0]

    preds = preds / np.sum(preds)

    idx = np.argmax(preds)
    confidence = round(float(preds[idx] * 100), 2)

    model_votes = {
        "CNN": CLASS_NAMES[np.argmax(cnn.predict(img)[0])],
        "VGG16": CLASS_NAMES[np.argmax(vgg.predict(img)[0])],
        "ResNet50": CLASS_NAMES[np.argmax(resnet.predict(img)[0])],
        "MobileNet": CLASS_NAMES[np.argmax(mobilenet.predict(img)[0])],
        "EfficientNet": CLASS_NAMES[np.argmax(efficientnet.predict(img)[0])]
    }

    prob_table = {
        CLASS_NAMES[i]: round(float(preds[i]*100),2)
        for i in range(len(CLASS_NAMES))
    }

    return CLASS_NAMES[idx], confidence, model_votes, prob_table
def predict_deficiency_full_frame(img):

    load_models()

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    probs = []

    probs.append(cnn.predict(img, verbose=0)[0])
    probs.append(vgg.predict(img, verbose=0)[0])
    probs.append(resnet.predict(img, verbose=0)[0])
    probs.append(mobilenet.predict(img, verbose=0)[0])
    probs.append(efficientnet.predict(img, verbose=0)[0])

    avg_probs = np.mean(probs, axis=0)

    pred_idx = int(np.argmax(avg_probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(avg_probs[pred_idx] * 100)

    prob_dict = {
        CLASS_NAMES[i]: round(float(avg_probs[i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }

    return pred_label, round(confidence, 2), prob_dict
# =============================
# EXPLAINABLE AI
# =============================
def explain_prediction(pred):
    return {
        "Vitamin A": "Conjunctival dryness and surface irregularities detected.",
        "Vitamin B": "Texture pallor and nail/tongue changes observed.",
        "Vitamin C": "Inflammation and vascular surface irregularities detected.",
        "Vitamin D": "Skin tone inconsistency associated with low vitamin D.",
        "Normal": "No abnormal visual biomarkers detected."
    }.get(pred, "No explanation available.")

# =============================
# GRAD-CAM
# =============================
def generate_gradcam(model, img, layer="conv5_block3_out"):
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer).output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(img)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(conv[0] * pooled, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) else 1
    return heatmap

def save_heatmap(img_path, heatmap, name):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(os.path.join(HEATMAP_FOLDER, name), overlay)
    return name
def generate_pdf_report(data, filename, heatmap_path=None):
    """
    Generates a PDF medical report including:
    - Diagnosis
    - Confidence
    - Explainable AI
    - Dietary recommendations
    - Doctor information
    - Grad-CAM heatmap image
    """

    pdf_path = os.path.join(REPORT_FOLDER, filename)
    c = canvas.Canvas(pdf_path)
    y = 800

    def draw(text):
        nonlocal y
        c.drawString(40, y, text)
        y -= 18

    # =============================
    # HEADER
    # =============================
    draw("Vitamin AI – Medical Diagnostic Report")
    draw("-" * 70)
    draw("")

    # =============================
    # DIAGNOSIS
    # =============================
    draw(f"Diagnosis      : {data.get('prediction', 'N/A')}")
    draw(f"Confidence     : {data.get('confidence', 'N/A')}%")
    draw("")

    # =============================
    # EXPLAINABLE AI
    # =============================
    draw("Explainable AI:")
    explanation = data.get("ai_explanation", "No explanation available.")
    for line in explanation.split(". "):
        draw(line.strip())
    draw("")

    # =============================
    # DIET SECTION
    # =============================
    diet = data.get("diet_info")
    if diet:
        draw("Dietary Recommendations:")
        for food in diet.get("foods", []):
            draw(f"• {food}")
        draw(diet.get("advice", ""))
        draw("")

    # =============================
    # HEATMAP IMAGE SECTION
    # =============================
    if heatmap_path:
        draw("Grad-CAM Heatmap:")
        y -= 10

        heatmap_full_path = os.path.join(HEATMAP_FOLDER, str(heatmap_path))

        if os.path.exists(heatmap_full_path):
            try:
                heatmap_img = ImageReader(heatmap_full_path)
                c.drawImage(
                    heatmap_img,
                    40,           # X
                    y - 220,      # Y
                    width=320,
                    height=220,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                y -= 240
            except Exception as e:
                draw("⚠ Unable to load heatmap image.")
        else:
            draw("⚠ Heatmap image not found.")

        draw("")

    # =============================
    # DOCTOR SECTION
    # =============================
    doctor = data.get("doctor_info")
    if doctor:
        draw("Recommended Specialist:")
        draw(f"Doctor Name   : {doctor.get('doctor_name', '')}")
        draw(f"Specialist    : {doctor.get('specialist', '')}")
        draw(f"Contact       : {doctor.get('contact_link', '')}")
        draw("")

    # =============================
    # DISCLAIMER
    # =============================
    draw("-" * 70)
    draw("⚠ This report is AI-assisted and not a medical prescription.")

    # SAVE PDF
    c.save()
    return filename
# =============================
# LIVE CAMERA API
# =============================
@app.route("/live_predict", methods=["POST"])
def live_predict():

    if "frame" not in request.files:
        return jsonify({"error": "No frame received"})

    file = request.files["frame"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify({"error": "Invalid image"})

    deficiency, confidence, probabilities = predict_deficiency_full_frame(img)

    diet = DIET_MAP.get(deficiency)

    return jsonify({
        "deficiency": deficiency,
        "confidence": confidence,
        "probabilities": probabilities,
        "diet": DIET_MAP.get(deficiency)
    })
@app.route("/health")
def health():
    return "OK"
# =============================
# MAIN ROUTE
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    # ✅ Language from GET or POST
    lang = request.values.get("language", "en")
    texts = load_language(lang)

    body_part = request.values.get("body_part", "Eye")

    data = {
        "prediction": None,
        "confidence": None,
        "model_votes": None,
        "prob_table": None,
        "ai_explanation": None,
        "diet_info": None,
        "doctor_info": None,
        "image_path": None,
        "heatmap_path": None,
        "report_path": None,   # ✅ IMPORTANT
        "warnings": [],
        "texts": texts,
        "selected_lang": lang,
        "selected_body_part": body_part
    }

    # ✅ IMAGE UPLOAD ONLY WHEN POST
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]

        if file and file.filename:
            name = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, name)
            file.save(img_path)

            # --- Quality checks ---
            img_cv = cv2.imread(img_path)
            if is_blurry(img_cv):
                data["warnings"].append(texts.get("blurry_warning", "Image is blurry"))
            if is_low_light(img_cv):
                data["warnings"].append(texts.get("low_light_warning", "Low lighting detected"))

            # --- Prediction ---
            img_tensor = preprocess_image(img_path)
            pred, conf, votes, probs = ensemble_predict(img_tensor)

            diet_info = DIET_MAP.get(pred)
            doctor_info = DOCTOR_DATABASE.get(pred)

            # --- 🔥 GENERATE HEATMAP FIRST ---
            heat = generate_gradcam(resnet, img_tensor)
            hm_name = f"hm_{name}"
            save_heatmap(img_path, heat, hm_name)

            # --- 🔥 GENERATE PDF AFTER HEATMAP EXISTS ---
            report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            generate_pdf_report(
                {
                    "prediction": pred,
                    "confidence": conf,
                    "ai_explanation": explain_prediction(pred),
                    "diet_info": diet_info,
                    "doctor_info": doctor_info
                },
                report_name,
                heatmap_path=hm_name   # ✅ NOW EXISTS
            )
            # --- UI DATA ---
            data.update({
                "prediction": pred,
                "confidence": conf,
                "model_votes": votes,
                "prob_table": probs,
                "ai_explanation": explain_prediction(pred),
                "diet_info": diet_info,                 # ✅ FIXED
                "doctor_info": doctor_info,
                "image_path": name,
                "heatmap_path": hm_name,
                "report_path": report_name,              # ✅ FIXED
                "texts": texts,
                "selected_lang": lang,
                "selected_body_part": body_part
            })
    # ✅ ALWAYS RENDER (GET + POST)
    return render_template("index.html", **data)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)