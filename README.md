# SmartSkinAI
**AI-Based Skin Disease Detection with Prescription & Diet Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-yellow.svg)
![License](https://img.shields.io/badge/License-Academic-red.svg)

---

## 🩺 Overview

**SmartSkinAI** is an intelligent web-based healthcare system that leverages deep learning to detect skin diseases from uploaded images. Built with a CNN architecture based on ResNet (pre-trained on ImageNet), the system provides:

✨ **Automated disease prediction** across 10 dermatological conditions  
📄 **Personalized medical prescriptions** and dietary recommendations  
📱 **WhatsApp integration** for instant PDF report delivery  
🏥 **Nearby doctor finder** using geolocation and OpenStreetMap API  
🔒 **Secure cloud storage** with Firebase Firestore  
🌓 **Responsive UI** with dark/light theme support

This project bridges the gap between AI technology and accessible healthcare, particularly benefiting underserved and remote areas.

---

## 👥 Team Members

| Name | 
|------|
| Julakanti Varshith |
| Likita Reddy Bojja |


---

## 🎯 Features

### Core Functionality
✅ **AI-Powered Detection** - ResNet-based CNN trained on dermatological datasets  
✅ **Multi-Class Classification** - Identifies 10 skin conditions including melanoma, eczema, psoriasis, etc.  
✅ **Confusion Matrix Evaluation** - Rigorous performance metrics for model accuracy  
✅ **PDF Report Generation** - Professional medical reports with ReportLab  
✅ **WhatsApp Notifications** - Automated report delivery via Twilio API  
✅ **Doctor Locator** - Find nearby dermatologists using geolocation  
✅ **Firebase Integration** - Secure cloud storage and retrieval  
✅ **Theme Support** - Dark and light mode with localStorage persistence

### Disease Categories
- Eczema
- Warts, Molluscum & Viral Infections
- Melanoma
- Atopic Dermatitis
- Basal Cell Carcinoma
- Melanocytic Nevi
- Benign Keratosis-like Lesions
- Psoriasis & Lichen Planus
- Seborrheic Keratoses
- Tinea & Fungal Infections

---

## ⚙️ Tech Stack

### **Frontend**
- HTML5, CSS3, JavaScript (ES6+)
- Google Fonts (Inter)
- Responsive Design with CSS Grid & Flexbox
- LocalStorage for theme persistence

### **Backend**
- **Flask** - Web framework with CORS support
- **TensorFlow/Keras** - Deep learning model (ResNet-based CNN)
- **NumPy & PIL** - Image preprocessing
- **ReportLab** - PDF generation
- **Firebase Admin SDK** - Firestore database
- **Twilio** - WhatsApp API integration
- **Requests** - OpenStreetMap API calls

### **Database**
- **Firebase Firestore** - NoSQL cloud database

### **AI/ML**
- **Model Architecture:** ResNet (pre-trained on ImageNet)
- **Framework:** TensorFlow 2.x with Keras
- **Input Shape:** 224×224×3 RGB images
- **Evaluation:** Confusion Matrix

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager
- Firebase service account credentials
- Twilio account (optional, for WhatsApp)
- ngrok (optional, for public URL)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/07Saroj/SmartSkinAI.git
cd SmartSkinAI
```

### 2️⃣ Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Firebase Configuration
1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Generate a service account key (JSON)
3. Save as `serviceAccountKey.json` in the project root

### 4️⃣ Environment Variables (Optional)
Create a `.env` file for sensitive credentials:
```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
BASE_URL=https://your-ngrok-url.ngrok-free.app
```

### 5️⃣ Run the Application
```bash
python app3.py
```
Backend runs at → **http://127.0.0.1:5000**

### 6️⃣ Access Frontend
Open `index.html` in your browser or serve via:
```bash
python -m http.server 8000
```
Frontend accessible at → **http://localhost:8000**

---

## 📡 API Endpoints

### `POST /api/predict`
**Description:** Analyzes uploaded image and returns disease prediction  
**Request Body:** `multipart/form-data` with `image` field  
**Response:**
```json
{
  "disease": "Melanoma",
  "description": ["A serious form of skin cancer...", "Early detection is crucial..."],
  "medication": "Surgical removal, immunotherapy...",
  "diet": "High-antioxidant diet including berries..."
}
```

### `POST /api/save-prescription`
**Description:** Generates PDF report and sends via WhatsApp  
**Request Body:**
```json
{
  "name": "John Doe",
  "age": "35",
  "gender": "Male",
  "whatsapp": "+919876543210",
  "disease": "Melanoma",
  "description": [...],
  "medication": "...",
  "diet": "..."
}
```
**Response:**
```json
{
  "message": "Prescription saved and WhatsApp message sent",
  "id": "uuid-here",
  "pdf_url": "http://yourserver.com/reports/uuid.pdf"
}
```

### `GET /find_doctors?lat=<lat>&lon=<lon>`
**Description:** Finds nearby doctors using OpenStreetMap  
**Response:**
```json
{
  "area_name": "New York, USA",
  "doctors": [
    {"name": "Dr. Smith Clinic", "address": "123 Main St"}
  ]
}
```

### `GET /download/<doc_id>`
**Description:** Downloads PDF report from Firestore

---

## 🐳 Docker Deployment (Optional)

### Dockerfile Example
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app3.py"]
```

### Docker Compose
```yaml
version: "3.8"

services:
  backend:
    build: ./backend
    container_name: SmartSkinAI-backend
    ports:
      - "5000:5000"
    volumes:
      - ./backend/reports:/app/reports
      - ./backend/model.keras:/app/model.keras
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/serviceAccountKey.json
      - MODEL_PATH=/app/model.keras
      - BASE_URL=https://bria-unurbanized-adorably.ngrok-free.dev # Replace with your deployed backend URL

  frontend:
    build: ./frontend
    container_name: SmartSkinAI-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

```

Run with:
```bash
docker-compose up --build
```

---

## 📁 Project Structure
```
SmartSkinAI/
│
├── backend/
│   ├── app3.py
│   ├── model.keras
│   ├── Dockerfile
│   ├── requirements.txt
│   └── reports/
│
├── frontend/
│   ├── index.html
│   ├── Dockerfile
│   └── ...
│
├── docker-compose.yml
├── README.md
└── .gitignore

---

## 🧪 Model Performance

The CNN model was evaluated using **confusion matrix analysis** with the following architecture:
- **Base Model:** ResNet (ImageNet pre-trained)
- **Input Size:** 224×224×3
- **Output Classes:** 10 skin conditions
- **Training Framework:** TensorFlow/Keras

---

## 📱 WhatsApp Integration Setup

1. Sign up at [Twilio](https://www.twilio.com)
2. Activate WhatsApp sandbox
3. Add credentials to `.env` or directly in `app3.py`
4. Use ngrok to expose local server:
```bash
ngrok http 5000
```
5. Update `BASE_URL` in code with ngrok URL

---

## 🌐 Live Demo

*https://07saroj.github.io/SmartSkinAI/*
 For frontend only
---

## 🤝 Contributing

This is an academic project. For suggestions or issues:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is created as part of the **Engineering Clinics Project** at VIT-AP University.  
**© 2025 SmartSkinAI Team. All Rights Reserved.**

---

## 🙏 Acknowledgments

- VIT-AP University & SCOPE Department
- Prof. Ajith Jubilson E (Project Guide)
- Open-source community (TensorFlow, Flask, Firebase)
- Twilio for WhatsApp API
- OpenStreetMap for geolocation services

---

**⭐ If you find this project helpful, please consider starring the repository!**
```

---

## 📝 Additional Files to Include:

### `.gitignore`
```
# Credentials
serviceAccountKey.json
.env

# Python
__pycache__/
*.pyc
*.pyo
venv/
env/

# Reports
reports/*.pdf

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`
```
Flask==2.3.0
Flask-CORS==4.0.0
tensorflow==2.13.0
Pillow==10.0.0
numpy==1.24.3
firebase-admin==6.2.0
reportlab==4.0.4
twilio==8.5.0
requests==2.31.0
