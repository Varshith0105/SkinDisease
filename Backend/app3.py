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
    "eczema", "warts_molluscum_viral", "melanoma", "atopic_dermatitis",
    "Basal Cell Carcinoma", "Melanocytic Nevi", "Benign Keratosis-like Lesions ",
    "psoriasis_lichen", "seborrheic_keratoses", "tinea_fungal"
]

# ------------------ DISEASE INFO ------------------
DISEASE_INFO = {
    "eczema": {
        "description": [
            "Eczema is a condition that makes your skin red and itchy. It is common in children but can occur at any age.",
            "It tends to flare periodically and may be accompanied by asthma or hay fever."
        ],
        "medication": "Apply prescribed topical corticosteroids or calcineurin inhibitors. Antihistamines can relieve itching.",
        "diet": "Avoid dairy, gluten, and processed foods. Include omega-3 rich foods like salmon and flaxseed."
    },
    "warts_molluscum_viral": {
        "description": [
            "Warts and molluscum are common viral skin infections caused by HPV and poxvirus respectively.",
            "They appear as small, raised bumps and are contagious through direct contact."
        ],
        "medication": "Salicylic acid treatments, cryotherapy, or prescribed antiviral creams.",
        "diet": "Boost immunity with Vitamin C-rich foods, zinc, and probiotics."
    },
    "melanoma": {
        "description": [
            "Melanoma is the most serious type of skin cancer, developing in the cells that give skin its color.",
            "Early detection is critical for successful treatment."
        ],
        "medication": "Requires immediate medical attention. Treatment may include surgery, immunotherapy, or targeted therapy.",
        "diet": "Antioxidant-rich diet: berries, leafy greens, green tea. Avoid processed meats and alcohol."
    },
    "atopic_dermatitis": {
        "description": [
            "Atopic dermatitis is a chronic inflammatory skin condition causing itchy, inflamed skin.",
            "It often begins in childhood and can persist into adulthood."
        ],
        "medication": "Moisturizers, topical corticosteroids, dupilumab (Dupixent) for severe cases.",
        "diet": "Avoid known triggers. Include anti-inflammatory foods like turmeric, fish oil, and vegetables."
    },
    "Basal Cell Carcinoma": {
        "description": [
            "Basal cell carcinoma is the most common form of skin cancer.",
            "It rarely spreads but must be treated promptly to prevent local tissue damage."
        ],
        "medication": "Surgical removal, Mohs surgery, or topical chemotherapy cream (Efudex). Consult a dermatologist.",
        "diet": "High antioxidant diet, Vitamin D from safe sun exposure, avoid smoking."
    },
    "Melanocytic Nevi": {
        "description": [
            "Melanocytic nevi are common benign moles formed by clusters of pigmented cells.",
            "Most are harmless but should be monitored for changes in size, shape, or color."
        ],
        "medication": "Usually no treatment needed. Surgical removal if suspicious or cosmetically bothersome.",
        "diet": "No specific diet required. Maintain sun protection habits."
    },
    "Benign Keratosis-like Lesions ": {
        "description": [
            "Benign keratosis includes non-cancerous growths on the skin surface.",
            "They are common with aging and generally harmless."
        ],
        "medication": "Cryotherapy or curettage if removal desired. No medical treatment usually required.",
        "diet": "Stay hydrated. Vitamin E and C-rich foods help maintain healthy skin."
    },
    "psoriasis_lichen": {
        "description": [
            "Psoriasis is a chronic autoimmune condition causing rapid skin cell buildup, resulting in scales and red patches.",
            "Lichen planus is an inflammatory condition affecting skin and mucous membranes."
        ],
        "medication": "Topical treatments, phototherapy, systemic medications like methotrexate or biologics.",
        "diet": "Anti-inflammatory diet: fish, olive oil, leafy greens. Avoid alcohol and red meat."
    },
    "seborrheic_keratoses": {
        "description": [
            "Seborrheic keratoses are common noncancerous skin growths that appear as waxy, scaly, slightly raised growths.",
            "They are more common in older adults."
        ],
        "medication": "Cryotherapy or electrosurgery if removal is desired. No treatment needed if asymptomatic.",
        "diet": "Balanced diet with antioxidants. Staying hydrated keeps skin healthy."
    },
    "tinea_fungal": {
        "description": [
            "Tinea is a fungal infection of the skin, scalp, or nails, commonly known as ringworm or athlete's foot.",
            "It is contagious and spreads through contact with infected skin or surfaces."
        ],
        "medication": "Antifungal creams like clotrimazole or terbinafine. Oral antifungals for severe cases.",
        "diet": "Reduce sugar and refined carbs. Include garlic, coconut oil, and probiotics to fight fungal growth."
    }
}

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

        # ✅ Get disease info with fallback
        info = DISEASE_INFO.get(pred_class, {
            "description": ["No detailed information available for this condition."],
            "medication": "Please consult a dermatologist for proper treatment.",
            "diet": "Maintain a balanced and healthy diet."
        })

        return jsonify({
            "disease": pred_class,
            "description": info["description"],
            "medication": info["medication"],
            "diet": info["diet"]
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


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
        return jsonify({"message": "Saved", "id": doc_id})
    except Exception as e:
        print("❌ Save error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/find_doctors", methods=["GET"])
def find_doctors():
    try:
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        # Placeholder response — integrate Google Places API here if needed
        return jsonify({"doctors": [
            {"name": "Dr. Skin Specialist", "address": "Near your location"}
        ]})
    except Exception as e:
        print("❌ Doctors error:", e)
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)
