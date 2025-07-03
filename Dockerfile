# Gunakan Python resmi versi 3.10 sebagai base image (atau 3.9, 3.11, 3.12)
# Gunakan versi 'slim-buster' atau 'slim-bullseye' untuk ukuran image yang lebih kecil
FROM python:3.10-slim-buster

# Atur direktori kerja di dalam container
WORKDIR /app

# Instal dependensi Python dari requirements.txt
# Copy requirements.txt dulu agar bisa di-cache jika tidak ada perubahan pada requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa kode aplikasi Anda ke dalam container
# Ini akan menyalin app.py dan fabric_recommendation_model.pkl
COPY . .

# Cloud Run akan mengekspos port 8080 secara default.
# Aplikasi Anda harus mendengarkan di port ini.
ENV PORT 8080

# Perintah untuk menjalankan aplikasi Anda
# CMD adalah perintah yang dijalankan saat container dimulai
# Ganti 'app:app' jika nama file Python utama Anda bukan 'app.py'
# atau variabel aplikasi FastAPI Anda bukan 'app'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
