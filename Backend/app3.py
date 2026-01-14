import os
# from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
import io, base64, uuid
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf

import firebase_admin
from urllib.parse import urljoin
from firebase_admin import credentials, firestore
from flask import send_file
from flask import send_from_directory
import tempfile
from twilio.rest import Client
import requests

# ✅ Memory-safe TensorFlow settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load CNN model
model = None
def load_model():
    global model
    if model is None:
        print("🧠 Loading TensorFlow model into memory...")
        model_path = os.getenv("MODEL_PATH", "model.keras")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    return model
#model = tf.keras.models.load_model("model.keras")
#print("Model loaded successfully!")

CLASSES = [
    "eczema",
    "warts_molluscum_viral",
    "melanoma",
    "atopic_dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis-like Lesions ",
    "psoriasis_lichen",
    "seborrheic_keratoses",
    "tinea_fungal"
]

DISEASE_INFO = {
    "eczema": {
        "name": "Eczema",
        "description": [
            "A chronic condition causing itchy, inflamed, and dry skin.",
            "Often triggered by allergens, stress, or environmental factors."
        ],
        "medication": "Topical corticosteroids, antihistamines for itching, and moisturizers to reduce dryness.",
        "diet": "Omega-3-rich foods (like fish and flaxseeds), avoid processed foods and excessive dairy."
    },
    "warts_molluscum_viral": {
        "name": "Warts, Molluscum, and Other Viral Infections",
        "description": [
            "Caused by viral infections such as HPV or Molluscum contagiosum.",
            "Appear as small bumps or lesions, sometimes contagious through contact."
        ],
        "medication": "Cryotherapy, salicylic acid treatments, or topical antivirals under medical supervision.",
        "diet": "Boost immunity with vitamin C and zinc-rich foods (citrus, spinach, pumpkin seeds)."
    },
    "melanoma": {
        "name": "Melanoma",
        "description": [
            "A serious form of skin cancer that develops from pigment-producing cells (melanocytes).",
            "Early detection is crucial to prevent spreading."
        ],
        "medication": "Surgical removal, immunotherapy, or targeted therapy. Urgent dermatologist consultation required.",
        "diet": "High-antioxidant diet including berries, leafy greens, and vitamin D-rich foods."
    },
    "atopic_dermatitis": {
        "name": "Atopic Dermatitis",
        "description": [
            "A common type of eczema causing red, itchy, and cracked skin.",
            "Usually starts in childhood and may flare up periodically."
        ],
        "medication": "Moisturizers, steroid creams, or calcineurin inhibitors. Avoid irritants and stress.",
        "diet": "Probiotic-rich foods (yogurt, kefir) and foods high in omega-3s to reduce inflammation."
    },
    "Basal Cell Carcinoma": {
        "name": "Basal Cell Carcinoma (BCC)",
        "description": [
            "A slow-growing type of skin cancer appearing as pearly or waxy bumps.",
            "Usually caused by long-term UV exposure."
        ],
        "medication": "Surgical excision, topical treatments, or Mohs surgery. Regular dermatologist checkups are advised.",
        "diet": "Include vitamin E, green tea, and antioxidant-rich fruits."
    },
    "Melanocytic Nevi": {
        "name": "Melanocytic Nevi (NV)",
        "description": [
            "Commonly known as moles; usually harmless pigment spots on the skin.",
            "Monitor for any changes in size, color, or shape."
        ],
        "medication": "No treatment required unless suspicious. Surgical removal if necessary.",
        "diet": "Healthy balanced diet with fruits and vegetables; avoid excessive sun exposure."
    },
    "Benign Keratosis-like Lesions ": {
        "name": "Benign Keratosis-like Lesions (BKL)",
        "description": [
            "Non-cancerous skin growths that appear as rough or crusty patches.",
            "Common in older adults and may resemble warts or sun damage."
        ],
        "medication": "Laser therapy, cryotherapy, or minor surgery for cosmetic reasons.",
        "diet": "Balanced diet; include foods rich in vitamins A and E for skin health."
    },
    "psoriasis_lichen": {
        "name": "Psoriasis, Lichen Planus, and Related Diseases",
        "description": [
            "Autoimmune conditions causing thick, scaly patches of skin.",
            "May flare up due to stress, infections, or certain medications."
        ],
        "medication": "Topical corticosteroids, phototherapy, or biologic drugs depending on severity.",
        "diet": "Anti-inflammatory foods such as turmeric, salmon, and leafy greens."
    },
    "seborrheic_keratoses": {
        "name": "Seborrheic Keratoses and Other Benign Tumors",
        "description": [
            "Common, noncancerous skin growths that appear waxy or wart-like.",
            "Usually harmless but can be removed for cosmetic reasons."
        ],
        "medication": "Cryotherapy, curettage, or laser treatment if removal desired.",
        "diet": "Maintain a healthy diet; ensure proper hydration and vitamin E intake."
    },
    "tinea_fungal": {
        "name": "Tinea, Ringworm, Candidiasis, and Other Fungal Infections",
        "description": [
            "Caused by fungal organisms that thrive in moist environments.",
            "Common symptoms include itching, redness, and circular rashes."
        ],
        "medication": "Topical or oral antifungal medications (clotrimazole, fluconazole). Keep affected area dry.",
        "diet": "Low sugar diet, include garlic and probiotics to support antifungal defense."
    }
}


def preprocess_image(image_file):
    """Convert uploaded image to model input"""
    img = Image.open(image_file).convert("RGB")
    # Change resize dimensions to match model's expected input shape
    img = img.resize((224, 224))  # Changed from (125, 100) to (224, 224)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        img_input = preprocess_image(image_file)
        # Load model lazily
        model_instance = load_model()
        preds = model_instance.predict(img_input)
        pred_index = int(np.argmax(preds))
        pred_class = CLASSES[pred_index]

        disease_info = DISEASE_INFO.get(pred_class)

        return jsonify({
            "disease": disease_info["name"],
            "description": disease_info["description"],
            "medication": disease_info["medication"],
            "diet": disease_info["diet"]
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
@app.route("/download/<doc_id>", methods=["GET"])
def download_pdf(doc_id):
    """
    Retrieve the saved PDF from Firestore and send it as a downloadable file.
    """
    try:
        doc = db.collection("predictions").document(doc_id).get()

        if not doc.exists:
            return jsonify({"error": "Document not found"}), 404

        data = doc.to_dict()
        pdf_base64 = data.get("pdfBase64")

        if not pdf_base64:
            return jsonify({"error": "No PDF found for this document"}), 404

        # Decode the Base64 back into bytes
        pdf_bytes = base64.b64decode(pdf_base64)

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(pdf_bytes)
        temp_file.flush()
        temp_file.seek(0)

        # Serve the file
        return send_file(
            temp_file.name,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{doc_id}.pdf"
        )

    except Exception as e:
        print("Download error:", e)
        return jsonify({"error": "Failed to retrieve PDF"}), 500


@app.route('/reports/<path:filename>', methods=['GET'])
def serve_report(filename):
    return send_from_directory('reports', filename)

@app.route('/find_doctors', methods=['GET'])
def find_doctors():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400

    area_name = "Your current area"
    doctors = []

    try:
        # --- STEP 1: Get Area Name from NOMINATIM (This part is working and stays the same) ---
        headers = {'User-Agent': 'SmartSkinHealthApp/1.0'}
        geocode_url = (f"https://nominatim.openstreetmap.org/reverse?"
                       f"format=jsonv2&lat={lat}&lon={lon}")
        
        geocode_response = requests.get(geocode_url, headers=headers)
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
        
        if geocode_data and 'display_name' in geocode_data:
            area_name = geocode_data.get('display_name')

        # --- STEP 2: Find Doctors with OVERPASS API (UPDATED QUERY) ---
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # This is a broader query for doctors, clinics, AND hospitals
        # The '~' means "or", so it searches for any of these types.
        overpass_query = f"""
        [out:json];
        (
          node["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
          way["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
          relation["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
        );
        out center;
        """
        
        places_response = requests.post(overpass_url, data=overpass_query, headers=headers)
        places_response.raise_for_status()
        places_data = places_response.json()
        
        for place in places_data.get('elements', [])[:10]: # Increased limit to 10
            tags = place.get('tags', {})
            # We will add a fallback for the address if street is not available
            address = f"{tags.get('addr:street', '')} {tags.get('addr:housenumber', '')}".strip()
            if not address:
                address = tags.get('addr:full', 'Address not available')

            doctors.append({
                'name': tags.get('name', 'Medical Facility'),
                'address': address
            })

        # --- RETURN COMBINED DATA ---
        return jsonify({
            'area_name': area_name,
            'doctors': doctors
        })

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return jsonify({'error': 'Failed to fetch data from OpenStreetMap APIs'}), 500


@app.route("/api/save-prescription", methods=["POST"])
def save_prescription():
    try:
        data = request.json
        name = data["name"]
        age = data["age"]
        gender = data["gender"]
        whatsapp = data.get("whatsapp", "")
        disease = data["disease"]
        description = data["description"]
        medication = data["medication"]
        diet = data["diet"]

        # PDF settings
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 100  # Start position (top margin)
        line_height = 22
        x_margin = 60  # left/right margin
        max_width = width - 2 * x_margin

        # Heading Section
        heading = "Prescription and Diet Recommendation for AI-Based Skin Disease Detection"
        brand_name = "Smart Skin Health"

        # Set brand name (top-right corner)
        p.setFont("Helvetica-Bold", 14)
        p.setFillColor(colors.green)
        p.drawString(60, height - 60, brand_name)  # (x, y) for top-left placement

        # Main heading (centered below brand)
        p.setFont("Helvetica-Bold", 16)
        p.setFillColor(colors.darkblue)

        heading_lines = simpleSplit(heading, "Helvetica-Bold", 16, max_width)
        y = height - 100  # Start below the brand name
        for hl in heading_lines:
            p.drawCentredString(width / 2, y, hl)
            y -= line_height
        y -= 20  # extra space after 

        p.setFont("Helvetica", 12)
        p.setFillColor(colors.black)

        def draw_bold_label(label, value):
            """Draws a bold label and wraps long text if needed"""
            nonlocal y
            if value:
                full_text = f"{label}: {value}"
            else:
                full_text = f"{label}:"

            wrapped_lines = simpleSplit(full_text, "Helvetica-Bold", 12, max_width)
            for line in wrapped_lines:
                p.setFont("Helvetica-Bold", 12)
                p.drawString(x_margin, y, line)
                y -= line_height
            y -= 5

        def draw_bullets(items):
            """Draws bullet points and wraps text neatly"""
            nonlocal y
            for item in items:
                wrapped_lines = simpleSplit(f"• {item}", "Helvetica", 12, max_width)
                for line in wrapped_lines:
                    p.setFont("Helvetica", 12)
                    p.drawString(x_margin + 15, y, line)
                    y -= line_height
            y -= 10  # Extra space after list

        # Draw content
        draw_bold_label("Patient Name", name)
        draw_bold_label("Age", age)
        draw_bold_label("Gender", gender)
        draw_bold_label("WhatsApp", whatsapp)
        draw_bold_label("Disease", disease)
        draw_bold_label("Description", "")
        draw_bullets(description)

        # For long text like medication/diet, wrap using draw_bullets for cleaner look
        draw_bold_label("Medication", "")
        draw_bullets([medication])
        draw_bold_label("Diet Plan", "")
        draw_bullets([diet])

        # Footer
        p.setFont("Helvetica-Oblique", 10)
        p.setFillColor(colors.gray)
        p.drawCentredString(width / 2, 40, "Generated by Smart Skin Health | For clinical guidance only")

        # Save PDF
        p.showPage()
        p.save()

        pdf_bytes = buffer.getvalue()
        buffer.close()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")


        doc_id = str(uuid.uuid4())
        db.collection("predictions").document(doc_id).set({
            "patient": {"name": name, "age": age, "gender": gender, "whatsapp": whatsapp},
            "prediction": disease,
            "description": description,
            "medication": medication,
            "diet": diet,
            "pdfBase64": pdf_base64,
            "createdAt": datetime.utcnow().isoformat()
        })
        # ✅ Save PDF file locally for WhatsApp sending
        pdf_filename = f"{doc_id}.pdf"
        pdf_path = os.path.join("reports", pdf_filename)
        os.makedirs("reports", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # ✅ Construct download link (assuming backend runs locally or deployed)
        base_url = os.getenv("BASE_URL", "http://localhost:5000")
        pdf_filename = f"{doc_id}.pdf"
        media_url = urljoin(base_url, f"/reports/{pdf_filename}")

        # ✅ Send via Twilio WhatsApp
        if whatsapp:
            # Fetch credentials securely from .env
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            whatsapp_from = os.getenv("TWILIO_WHATSAPP_NUMBER")

            client = Client(account_sid, auth_token)

            message = client.messages.create(
                from_=f"whatsapp:{whatsapp_from}",
                to=f"whatsapp:{whatsapp}",
                body=f"Hello {name}, here’s your AI-generated skin diagnosis report 🩺📄",
                media_url=[media_url]
            )

            print(f"WhatsApp message sent to {whatsapp}")


        return jsonify({
            "message": "Prescription saved and WhatsApp message sent",
            "id": doc_id,
            "pdf_url": media_url
        })

    except Exception as e:
        print("Save prescription error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

# ngrok http 5000
# https://bria-unurbanized-adorably.ngrok-free.dev/

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("AI-Based Skin Disease Detection")

model = tf.keras.models.load_model("model.h5")

uploaded = st.file_uploader("Upload skin image", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded).resize((224,224))
    arr = np.array(img)/255.0
    pred = model.predict(arr.reshape(1,224,224,3))
    st.success(f"Prediction: {np.argmax(pred)}")
