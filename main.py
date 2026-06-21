from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from fuzzywuzzy import fuzz
import re

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000/chat"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")
FAQ_DATA = {
   "deteksi": "Sistem deteksi jalan berlubang menggunakan model YOLO untuk mendeteksi kerusakan jalan dari gambar atau video.",

    "lapor": "Pengguna dapat mengunggah foto jalan berlubang melalui aplikasi untuk dilakukan analisis.",

    "akurasi": "Akurasi sistem bergantung pada model yang digunakan, kualitas dataset, dan proses pelatihan model.",

    "kamera": "Sistem dapat menggunakan kamera smartphone, CCTV, maupun drone sebagai sumber gambar.",

    "realtime": "Sistem dapat berjalan secara real-time jika dihubungkan dengan video stream atau kamera langsung.",

    "yolo": "YOLO (You Only Look Once) adalah algoritma object detection yang digunakan untuk mendeteksi jalan berlubang dengan cepat.",

    "dataset": "Dataset berisi kumpulan gambar jalan berlubang dan jalan normal yang digunakan untuk melatih model AI.",

    "ai": "Artificial Intelligence atau AI digunakan untuk mengenali pola kerusakan jalan secara otomatis.",

    "manfaat": "Sistem membantu mendeteksi kerusakan jalan lebih cepat sehingga dapat mengurangi risiko kecelakaan.",

    "bahaya": "Jalan berlubang dapat menyebabkan kecelakaan, kerusakan kendaraan, dan kemacetan lalu lintas.",

    "perbaikan": "Hasil deteksi dapat digunakan sebagai referensi untuk perbaikan jalan oleh pihak terkait.",

    "lokasi": "Sistem dapat menyimpan lokasi jalan berlubang menggunakan koordinat GPS.",

    "gambar": "Pengguna dapat mengunggah gambar jalan untuk dilakukan proses deteksi.",

    "video": "Sistem dapat mendeteksi jalan berlubang pada video yang diunggah atau video streaming.",

    "pengembang": "Sistem ini dikembangkan sebagai proyek deteksi jalan berlubang berbasis Artificial Intelligence.",

    "teknologi": "Teknologi yang digunakan meliputi Python, Flask, OpenCV, YOLO, dan Deep Learning.",

    "halo": "Halo! Saya adalah chatbot sistem deteksi jalan berlubang. Ada yang bisa saya bantu?",

    "terima kasih": "Sama-sama. Semoga informasi yang diberikan bermanfaat."
}

# Keyword mappings for common terms
KEYWORD_SYNONYMS = {
    "deteksi": [
        "deteksi",
        "mendeteksi",
        "identifikasi",
        "analisis"
    ],

    "lapor": [
        "lapor",
        "laporan",
        "pengaduan",
        "melapor"
    ],

    "akurasi": [
        "akurasi",
        "ketepatan",
        "precision",
        "hasil"
    ],

    "kamera": [
        "kamera",
        "cctv",
        "dashcam",
        "smartphone"
    ],

    "realtime": [
        "realtime",
        "real time",
        "langsung"
    ],

    "yolo": [
        "yolo",
        "model",
        "algoritma"
    ],

    "dataset": [
        "dataset",
        "data latih",
        "training data"
    ],

    "ai": [
        "ai",
        "artificial intelligence",
        "kecerdasan buatan"
    ],

    "manfaat": [
        "manfaat",
        "fungsi",
        "kegunaan",
        "keuntungan"
    ],

    "bahaya": [
        "bahaya",
        "risiko",
        "dampak",
        "kecelakaan"
    ],

    "perbaikan": [
        "perbaikan",
        "memperbaiki",
        "penanganan"
    ],

    "lokasi": [
        "lokasi",
        "gps",
        "koordinat",
        "titik"
    ],

    "gambar": [
        "gambar",
        "foto",
        "image"
    ],

    "video": [
        "video",
        "rekaman",
        "streaming"
    ],

    "pengembang": [
        "pengembang",
        "developer",
        "pembuat"
    ],

    "teknologi": [
        "teknologi",
        "framework",
        "tools"
    ],

    "halo": [
        "halo",
        "hai",
        "hello",
        "hi"
    ],

    "terima kasih": [
        "terima kasih",
        "makasih",
        "thanks"
    ]
}

class UserInput(BaseModel):
    message: str
    
def preprocess_input(message: str) -> str:
    """Normalize user input: lowercase, remove extra spaces, and handle basic synonyms."""
    message = re.sub(r'\s+', ' ', message.strip().lower())
    # Replace synonyms with canonical terms
    for canonical, synonyms in KEYWORD_SYNONYMS.items():
        for synonym in synonyms:
            message = message.replace(synonym, canonical)
    return message

def find_best_faq_match(user_message: str) -> tuple[str, int]:
    """Find the best FAQ match using multiple fuzzy matching strategies."""
    best_match = None
    highest_score = 0
    for question in FAQ_DATA:
        # Preprocess FAQ question
        processed_question = preprocess_input(question)
        processed_message = preprocess_input(user_message)
        # Use multiple fuzzy matching strategies
        partial_score = fuzz.partial_ratio(processed_message, processed_question)
        token_sort_score = fuzz.token_sort_ratio(processed_message,
        processed_question)
        ratio_score = fuzz.ratio(processed_message, processed_question)
        # Combine scores (weighted average for better accuracy)
        combined_score = (partial_score * 0.5 + token_sort_score * 0.3 +
        ratio_score * 0.2)
        # Lowered threshold to allow looser matches
        if combined_score > 70 and combined_score > highest_score: # Lowered from 80 to 70
            best_match = question
            highest_score = combined_score
            
        # Keyword-based matching as a fallback
        if not best_match:
            user_words = set(processed_message.split())
            question_words = set(processed_question.split())
            common_keywords = user_words.intersection(question_words)
            if len(common_keywords) >= 2: # At least 2 common keywords
          
                best_match = question
                highest_score = 75 # Arbitrary score for keyword match
    return best_match, highest_score

@app.post("/chat")
async def chat(user_input: UserInput):
    user_message = user_input.message.strip()
    
    # Find the best FAQ match
    best_match, score = find_best_faq_match(user_message)
    if best_match and score > 70:
        return {"response": FAQ_DATA[best_match]}
    
    # Fallback to NLP
    try:
        response = chatbot_pipeline(
            user_message,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=chatbot_pipeline.tokenizer.eos_token_id
        )[0]["generated_text"]
        
        # Remove input from output if present
        if response.startswith(user_message):
            response = response[len(user_message):].strip()
            
        return {"response": response or "Maaf, saya tidak memahami pertanyaan Anda. "
        "Silakan tanyakan tentang deteksi jalan berlubang, YOLO, AI, "
        "dataset, akurasi, pelaporan, atau teknologi yang digunakan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
            
@app.get("/faq")
async def faq():
    return FAQ_DATA