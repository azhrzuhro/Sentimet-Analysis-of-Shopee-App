# Sentimen Analisis Review APK Shopee di Play Store

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis sentimen ulasan pengguna terhadap aplikasi Shopee di Google Play Store. Dalam era digital saat ini, ulasan pengguna sangat penting dalam membentuk persepsi terhadap suatu aplikasi. Dengan menggunakan teknik pemrosesan bahasa alami (NLP), proyek ini mengidentifikasi apakah ulasan pengguna bersifat positif, negatif, atau netral.

## Tujuan
1. Mengumpulkan ulasan aplikasi Shopee dari Google Play Store.
2. Melakukan preprocessing teks untuk analisis sentimen.
3. Menganalisis dan mengklasifikasikan ulasan menjadi sentimen positif, negatif, atau netral.
4. Memvisualisasikan hasil analisis untuk mendapatkan wawasan yang lebih dalam.

## Teknologi dan Library yang Digunakan
- Python
- google-play-scraper
- NLTK
- Sastrawi
- WordCloud
- Pandas
- Matplotlib & Seaborn

## Cara Menjalankan Proyek
1. **Instalasi Library**
   Pastikan Anda telah menginstal pustaka yang diperlukan dengan menjalankan perintah berikut:
   ```bash
   pip install google-play-scraper nltk sastrawi wordcloud pandas matplotlib seaborn
   ```
2. **Menjalankan Notebook**
   Buka dan jalankan file `Proyek_ml_Analisis_Sentiment.ipynb` menggunakan Jupyter Notebook atau Google Colab.
3. **Melakukan Scraping Data**
   Notebook ini akan mengambil data ulasan aplikasi Shopee dari Play Store menggunakan google-play-scraper.
4. **Preprocessing Data**
   Data ulasan akan dibersihkan dan diproses dengan teknik NLP sebelum dianalisis.
5. **Analisis dan Visualisasi**
   Data akan diklasifikasikan ke dalam kategori sentimen dan divisualisasikan dalam bentuk grafik dan word cloud.

## Hasil dan Manfaat
Hasil analisis ini dapat memberikan wawasan bagi pengembang aplikasi Shopee untuk memahami persepsi pengguna dan meningkatkan pengalaman pengguna berdasarkan ulasan mereka.

