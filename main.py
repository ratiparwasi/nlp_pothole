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
    "terima kasih": "Sama-sama. Semoga informasi yang diberikan bermanfaat.",
    "cara kerja": "Sistem bekerja dengan menganalisis gambar atau video lalu memprediksi apakah ada bagian jalan yang rusak atau berlubang.",
    "fitur": "Fitur utama sistem ini meliputi deteksi otomatis, pelaporan, pencatatan lokasi, dan dukungan input gambar atau video.",
    "aplikasi": "Sistem ini dapat digunakan oleh pemerintah, instansi terkait, maupun masyarakat untuk memantau kondisi jalan secara lebih cepat.",
    "hasil": "Hasil deteksi dapat ditampilkan sebagai informasi kerusakan jalan, lokasi, dan rekomendasi tindak lanjut.",
    "pelatihan": "Model dilatih menggunakan dataset berisi gambar jalan berlubang dan jalan normal agar dapat mengenali pola kerusakan.",
    "faktor": "Kualitas gambar, pencahayaan, sudut pengambilan foto, dan kualitas dataset dapat memengaruhi hasil deteksi.",
    "kecepatan": "Sistem dirancang untuk memberikan hasil deteksi dengan cepat, terutama saat digunakan pada video atau stream kamera.",
    "langkah": "Langkah yang biasanya dilakukan adalah mengunggah gambar atau video, memprosesnya dengan model, lalu melihat hasil deteksi.",
    "apa itu yolo": "YOLO adalah algoritma deteksi objek yang populer karena cepat dan efisien untuk mendeteksi jalan berlubang.",
    "bagaimana cara kerja sistem": "Sistem menganalisis citra dengan model AI untuk mengenali pola jalan yang rusak lalu memberikan hasil deteksi.",
    "apakah bisa mendeteksi dari foto": "Ya, sistem dapat mendeteksi jalan berlubang dari foto yang diunggah maupun dari video.",
    "apakah sistem bisa real time": "Ya, jika terhubung dengan kamera langsung atau video stream, sistem dapat bekerja secara real-time.",
    "cara melapor": "Untuk melaporkan jalan berlubang, buka menu Lapor pada aplikasi, unggah foto lubang, isi deskripsi serta detail lokasi, lalu klik kirim.",
    "kontak": "Anda dapat menghubungi tim customer service melalui email support@pothole-app.id atau kontak pengaduan daerah yang tertera di aplikasi.",
    "siapa yang memperbaiki": "Perbaikan jalan dilakukan oleh Dinas Pekerjaan Umum (PUPR) setempat atau instansi terkait setelah menerima laporan terverifikasi.",
    "proses laporan": "Laporan Anda akan diverifikasi oleh AI/admin terlebih dahulu. Setelah valid, laporan akan diteruskan ke instansi berwenang untuk dijadwalkan perbaikan.",
    "privasi": "Kami menjaga kerahasiaan data pribadi Anda. Data hanya digunakan untuk validasi laporan jalan berlubang dan tidak akan disebarluaskan.",
    "jenis jalan": "Sistem dapat mendeteksi kerusakan pada berbagai jalan seperti jalan nasional, provinsi, kota/kabupaten, dan jalan desa.",
    "biaya": "Aplikasi ini sepenuhnya gratis untuk digunakan oleh masyarakat dalam membantu melaporkan jalan berlubang.",
    "jenis kerusakan": "Sistem ini dilatih untuk mendeteksi kerusakan aspal seperti lubang (pothole), retakan (cracks), dan gelombang pada permukaan jalan.",
    "syarat foto": "Foto harus jelas, tidak buram, diambil dengan pencahayaan cukup, dan memperlihatkan lubang jalan secara utuh agar mudah diverifikasi AI.",
    "gps aktif": "Sangat disarankan mengaktifkan GPS/Lokasi pada ponsel agar aplikasi dapat mencatat koordinat lokasi jalan berlubang secara akurat.",
    "cek status": "Anda dapat melacak status perbaikan laporan Anda di menu Riwayat Laporan (menunggu, diproses, atau selesai).",
    "ukuran file": "Maksimal ukuran foto yang dapat diunggah adalah 5MB dalam format JPG atau PNG.",
    "daftar akun": "Buat akun baru melalui menu Register di halaman masuk dengan mengisi email, nama lengkap, dan password.",
    "versi": "Aplikasi ini saat ini berjalan pada versi pengembangan terbaru untuk memastikan deteksi AI yang cepat dan stabil."
}

# Keyword mappings for common terms
KEYWORD_SYNONYMS = {
    "deteksi": ["deteksi", "mendeteksi", "identifikasi", "analisis", "temukan", "kenali"],
    "lapor": ["lapor", "laporan", "pengaduan", "melapor", "report", "laporkan", "adukan"],
    "akurasi": ["akurasi", "ketepatan", "precision", "tingkat akurasi"],
    "kamera": ["kamera", "cctv", "dashcam", "smartphone", "cam"],
    "realtime": ["realtime", "real time", "langsung", "live"],
    "yolo": ["yolo", "model", "algoritma", "detector"],
    "dataset": ["dataset", "data latih", "training data", "data pelatihan"],
    "ai": ["ai", "artificial intelligence", "kecerdasan buatan", "machine learning"],
    "manfaat": ["manfaat", "fungsi", "kegunaan", "keuntungan", "benefit"],
    "bahaya": ["bahaya", "risiko", "dampak", "kecelakaan", "ancaman"],
    "perbaikan": ["perbaikan", "memperbaiki", "penanganan", "repair", "diperbaiki"],
    "lokasi": ["lokasi", "gps", "koordinat", "tempat", "alamat"],
    "gambar": ["gambar", "foto", "image", "citra", "visual", "unggah foto"],
    "video": ["video", "rekaman", "streaming", "clip"],
    "pengembang": ["pengembang", "developer", "pembuat", "creator"],
    "teknologi": ["teknologi", "framework", "tools", "stack"],
    "halo": ["halo", "hai", "hello", "hi", "hey"],
    "terima kasih": ["terima kasih", "makasih", "thanks", "terima kasih banyak"],
    "cara kerja": ["cara kerja", "bagaimana kerja", "bagaimana cara kerja", "proses", "alur"],
    "fitur": ["fitur", "fitur utama", "feature", "fungsi utama"],
    "faktor": ["faktor", "pengaruh", "memengaruhi", "penyebab"],
    "cara melapor": ["cara melapor", "bagaimana melapor", "melaporkan jalan", "cara mengadukan", "cara kirim laporan"],
    "kontak": ["kontak", "hubungi", "call center", "telepon", "email", "cs", "customer service", "helpdesk"],
    "siapa yang memperbaiki": ["yang memperbaiki", "siapa perbaiki", "siapa membetulkan", "dinas pupr", "pemerintah"],
    "proses laporan": ["proses laporan", "setelah melapor", "alur laporan", "tindak lanjut"],
    "privasi": ["privasi", "aman", "keamanan", "data pribadi", "data saya", "secure"],
    "jenis jalan": ["jenis jalan", "kategori jalan", "jalan nasional", "jalan provinsi", "jalan kabupaten", "jalan desa"],
    "biaya": ["biaya", "gratis", "bayar", "berbayar", "tarif", "harga", "free"],
    "jenis kerusakan": ["jenis kerusakan", "rusak apa saja", "deteksi apa saja", "retakan", "retak", "gelombang"],
    "syarat foto": ["syarat foto", "foto yang baik", "kriteria foto", "upload foto", "gambar valid"],
    "gps aktif": ["gps", "lokasi aktif", "nyalakan gps", "koordinat", "maps"],
    "cek status": ["status", "cek status", "lacak", "riwayat", "history", "perkembangan", "pantau"],
    "ukuran file": ["ukuran", "mb", "maksimal", "file", "kapasitas", "size"],
    "daftar akun": ["daftar", "register", "akun", "buat akun", "sign up"]
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