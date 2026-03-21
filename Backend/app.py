import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import uuid
from datetime import datetime

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ------------------ CONFIG ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")

app = Flask(__name__)
CORS(app)

# ------------------ FIREBASE ------------------
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

# ------------------ MODEL ------------------
model = None

def load_model():
    global model
    if model is None:
        try:
            print("🧠 Loading model...")
            model_path = os.path.join(os.path.dirname(__file__), "model.keras")

            if not os.path.exists(model_path):
                raise Exception("❌ model.keras not found")

            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded")
        except Exception as e:
            print("❌ Model loading error:", e)
            raise e
    return model

# ------------------ CLASSES ------------------
CLASSES = [
    "eczema", "warts_molluscum_viral", "melanoma", "atopic_dermatitis",
    "Basal Cell Carcinoma", "Melanocytic Nevi", "Benign Keratosis-like Lesions",
    "psoriasis_lichen", "seborrheic_keratoses", "tinea_fungal"
]

# ------------------ DISEASE INFO ------------------
DISEASE_INFO = {
    "eczema": {
        "description": ["Eczema is a skin condition causing redness and itching."],
        "medication": "Use corticosteroids and antihistamines.",
        "diet": "Avoid dairy and processed foods."
    },
    "melanoma": {
        "description": ["Serious skin cancer requiring immediate care."],
        "medication": "Consult doctor immediately.",
        "diet": "Eat antioxidant-rich foods."
    }
    # (You can keep your full dictionary here)
}

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return jsonify({"message": "Backend is LIVE 🚀"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        print("📥 Request received")

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_instance = load_model()

        preds = model_instance.predict(img_array)
        pred_index = int(np.argmax(preds))
        pred_class = CLASSES[pred_index]

        print(f"✅ Prediction: {pred_class}")

        info = DISEASE_INFO.get(pred_class, {
            "description": ["No info available"],
            "medication": "Consult doctor",
            "diet": "Healthy diet"
        })

        return jsonify({
            "disease": pred_class,
            "description": info["description"],
            "medication": info["medication"],
            "diet": info["diet"]
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/save-prescription", methods=["POST"])
def save_prescription():
    try:
        init_firebase()
        data = request.json

        doc_id = str(uuid.uuid4())

        db.collection("predictions").document(doc_id).set({
            "data": data,
            "createdAt": datetime.utcnow().isoformat()
        })

        return jsonify({
            "message": "Saved successfully",
            "id": doc_id
        })

    except Exception as e:
        print("❌ Save error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/find-doctors", methods=["GET"])
def find_doctors():
    try:
        return jsonify({
            "doctors": [
                {"name": "Dr. Skin Specialist", "address": "Near your location"}
            ]
        })
    except Exception as e:
        print("❌ Doctors error:", str(e))
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)
