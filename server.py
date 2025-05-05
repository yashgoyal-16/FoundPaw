from flask import Flask, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import uuid
import os
import numpy as np
import re
import requests
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from twilio.twiml.messaging_response import MessagingResponse
from waitress import serve
import logging

# ✅ Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Single, proper CORS setup

# Production Config
app.config['DEBUG'] = False
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Setup
client = MongoClient("mongodb+srv://chetansharma9878600494:dMqlC78qVxwSeZbV@cluster0.qjmwt22.mongodb.net/")
db = client["foundpaw"]
dogs_collection = db["dogs"]

# Load ResNet model
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Global vectorizer (fit once)
vectorizer = CountVectorizer()
vectorizer.fit(["lost", "found", "dog", "retriever", "location", "phone"])  # ✅ Dummy fit with core vocab

@app.route('/uploads/<filename>')
def get_image(filename):
    """Serve the image from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    logger.info("Received message from: %s", request.form.get("From", ""))
    body = request.form.get("Body", "").lower()
    logger.info("Body of the message: %s", body)
    print("form", request.form)

    # ✅ Get uploaded image
    file = request.files.get("image")
    if not file or not file.content_type.startswith("image"):
        logger.warning("No image found or invalid file type.")
        resp = MessagingResponse()
        resp.message("❌ Please send a photo of the dog.")
        return str(resp)

    # ✅ Save uploaded image
    img_name = f"{uuid.uuid4()}.jpg"
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    file.save(img_path)

    # ✅ Extract info
    status_match = re.search(r"(lost|found)", body)
    location_match = re.search(r"location:\s*([0-9.\-]+),\s*([0-9.\-]+)", body)
    phone_match = re.search(r"phone:\s*(\d+)", body)
    description = body.replace("\n", " ")

    resp = MessagingResponse()

    if not (status_match and location_match and phone_match):
        resp.message("⚠️ Format: 'Lost dog golden retriever... Location: 30.9,75.8 Phone: 98xxxx'")
        return str(resp)

    status = status_match.group(1)
    lat, lon = float(location_match.group(1)), float(location_match.group(2))
    phone = phone_match.group(1)

    # ✅ Process image and text
    image_emb = image_to_embedding(img_path)
    processed_desc = preprocess_text(description)
    text_emb = text_to_embedding(processed_desc)

    # ✅ Match logic
    opposite_status = "found" if status == "lost" else "lost"
    matches = match_dog(image_emb, text_emb, lat, lon, opposite_status)

    if matches:
        reply = "✅ Possible match found near you!\n"
        for m in matches:
            reply += f"\n📍 *Description*: {m['text']}\n📞 *Phone*: {m['phone']}\n🌍 *Location*: {m['lat']}, {m['lon']}\n🖼️ Image: {request.url_root}uploads/{m['image_name']}\n"
        resp.message(reply)
    else:
        # ✅ Save to DB
        dogs_collection.insert_one({
            "image_name": img_name,
            "embedding": image_emb.tolist(),
            "text_embedding": text_emb.tolist(),
            "text": processed_desc,
            "status": status,
            "lat": lat,
            "lon": lon,
            "phone": phone,
            "timestamp": datetime.now().isoformat()
        })
        logger.info("Dog info saved.")
        resp.message("📦 Dog info saved. We'll notify you if we find a match. 🙏")

    return str(resp)

# ✅ Utilities
def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise

def image_to_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_tensor).squeeze().numpy()
    return embedding

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

def text_to_embedding(text):
    vectors = vectorizer.transform([text])
    return vectors.toarray()[0]

def match_dog(image_emb, text_emb, lat, lon, opposite_status):
    matches = []
    for entry in dogs_collection.find({"status": opposite_status}):
        dist_km = haversine_distance(lat, lon, entry["lat"], entry["lon"])
        if dist_km > 80:
            continue

        # ✅ Ensure numpy arrays
        entry_image_emb = np.array(entry["embedding"])
        entry_text_emb = np.array(entry["text_embedding"])

        # ✅ Skip mismatched dims safely
        if entry_text_emb.shape != text_emb.shape:
            continue

        image_sim = cosine_similarity([image_emb], [entry_image_emb])[0][0]
        text_sim = cosine_similarity([text_emb], [entry_text_emb])[0][0]
        score = 0.5 * image_sim + 0.5 * text_sim

        entry["_id"] = str(entry["_id"])
        entry["score"] = score
        matches.append(entry)

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:3]

def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
