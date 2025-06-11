import streamlit as st
import joblib

# Load model dan TF-IDF Vectorizer
try:
    model = joblib.load("training/best_logistic_model.pkl")
    tfidf = joblib.load("training/tfidf.pkl")
except FileNotFoundError as e:
    st.error(f"❌ File model atau vectorizer tidak ditemukan: {e}")
    st.stop()

# Pastikan urutan kelas model
model_classes = list(model.classes_)  # contoh: [0, 1, 2] => NEGATIF, NETRAL, POSITIF
# Mapping indeks: kita tukar POSITIF <-> NEGATIF
# Misalnya: POSITIF = index 2, NEGATIF = index 0, NETRAL = index 1
label_mapping = {
    0: "POSITIF",  # sebelumnya NEGATIF
    1: "NETRAL",
    2: "NEGATIF"   # sebelumnya POSITIF
}
colors = {
    "NEGATIF": "🔴",
    "NETRAL": "🟡",
    "POSITIF": "🟢"
}

# Judul Aplikasi
st.set_page_config(page_title="Analisis Sentimen Shopee", page_icon="🛒")
st.title("🛍️ Analisis Sentimen Review Aplikasi Shopee")
st.write("Masukkan ulasan pelanggan dan dapatkan prediksi sentimen secara otomatis menggunakan model Machine Learning.")

# Input Review
review = st.text_area("Masukkan Review Shopee", placeholder="Contoh: Produk sangat memuaskan dan cepat sampai!", height=150)

if st.button("Prediksi Sentimen"):
    if review.strip() == "":
        st.warning("⚠️ Harap masukkan review terlebih dahulu.")
    else:
        X_input = tfidf.transform([review])
        proba = model.predict_proba(X_input.toarray())[0]

        # Mapping urutan label asli ke urutan baru
        # Contoh: model.classes_ = [0, 1, 2] => NEGATIF, NETRAL, POSITIF
        # Kita tukar posisi 0 dan 2 di tampilan/logika
        # Prediksi akhir (berdasarkan probabilitas)
        predicted_idx = proba.argmax()
        predicted_label = label_mapping[predicted_idx]

        # Tampilkan hasil
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"{colors[predicted_label]} Sentimen: **{predicted_label}**")

        # Tampilkan probabilitas dalam urutan baru
        st.markdown("#### 🔎 Probabilitas Sentimen:")
        for i in range(3):
            st.progress(float(proba[i]), text=f"{label_mapping[i]} ({proba[i]*100:.2f}%)")

# Footer
st.markdown("---")
st.caption("🚀 Dibuat oleh tim Data Science • Model: Logistic Regression + TF-IDF")
