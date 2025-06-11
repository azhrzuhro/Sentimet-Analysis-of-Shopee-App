import streamlit as st
import joblib
import numpy as np

# Load model dan TF-IDF Vectorizer
model = joblib.load("best_logistic_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Label Sentimen
labels = ["NEGATIF", "NETRAL", "POSITIF"]
colors = {
    "NEGATIF": "🔴",
    "NETRAL": "🟡",
    "POSITIF": "🟢"
}

# Judul Aplikasi
st.set_page_config(page_title="Analisis Sentimen Shopee", page_icon="🛒")
st.title("🛍️ Analisis Sentimen Review Aplikasi Shopee")
st.write("Masukkan ulasan pelanggan dan dapatkan prediksi sentimen secara otomatis menggunakan model Machine Learning.")

# Input Teks Review
review = st.text_area("Masukkan Review Shopee", placeholder="Contoh: Saya sangat puas dengan layanan Shopee...", height=150)

# Tombol Prediksi
if st.button("Prediksi Sentimen"):
    if review.strip() == "":
        st.warning("Mohon masukkan teks review terlebih dahulu.")
    else:
        # Transformasi dan prediksi
        X_input = tfidf.transform([review])
        prediction = model.predict(X_input.toarray())[0]
        proba = model.predict_proba(X_input.toarray())[0]

        # Menentukan sentimen dengan logika threshold
        if proba[1] > 0.5:
            label_prediksi = "NETRAL"
        elif proba[2] > proba[0]:
            label_prediksi = "POSITIF"
        else:
            label_prediksi = "NEGATIF"

        # Tampilkan hasil prediksi
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"{colors[label_prediksi]} Sentimen: **{label_prediksi}**")

        # Tampilkan probabilitas semua kelas
        st.markdown("#### 🔎 Probabilitas Sentimen:")
        for i, label in enumerate(labels):
            st.progress(float(proba[i]), text=f"{label} ({proba[i]*100:.2f}%)")

# Footer
st.markdown("---")
st.caption("🚀 Dibuat oleh tim Data Science • Model: Logistic Regression + TF-IDF")
