import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io, base64, uuid, tempfile
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
from urllib.parse import urljoin
import requests
from twilio.rest import Client

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ✅ TensorFlow safe settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")

app = Flask(__name__)
CORS(app)

# ------------------ FIREBASE (LAZY LOAD) ------------------
db = None

def init_firebase():
    global db
    if db is None:
        try:
            cred_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase initialized")
        except Exception as e:
            print("❌ Firebase error:", e)

# ------------------ MODEL (LAZY LOAD) ------------------
model = None

def load_model():
    global model
    if model is None:
        print("🧠 Loading model...")
        model_path = os.path.join(os.path.dirname(__file__), "model.keras")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded")
    return model

# ------------------ CONSTANTS ------------------
CLASSES = [
    "eczema","warts_molluscum_viral","melanoma","atopic_dermatitis",
    "Basal Cell Carcinoma","Melanocytic Nevi","Benign Keratosis-like Lesions ",
    "psoriasis_lichen","seborrheic_keratoses","tinea_fungal"
]

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return "Backend is LIVE 🚀"

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        img = Image.open(request.files['image']).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_instance = load_model()
        preds = model_instance.predict(img_array)
        pred_class = CLASSES[int(np.argmax(preds))]

        return jsonify({"disease": pred_class})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/api/save-prescription", methods=["POST"])
def save_prescription():
    try:
        init_firebase()  # 🔥 initialize only when needed

        data = request.json
        doc_id = str(uuid.uuid4())

        db.collection("predictions").document(doc_id).set({
            "data": data,
            "createdAt": datetime.utcnow().isoformat()
        })

        return jsonify({"message": "Saved", "id": doc_id})

    except Exception as e:
        print("❌ Save error:", e)
        return jsonify({"error": str(e)}), 500

# ------------------ RUN ------------------
if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
