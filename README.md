# Laporan Proyek Machine Learning - Ghiyas Akhtar Razi Ramadhan

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Menurut World Health Organization (WHO), sekitar 17,9 juta orang meninggal setiap tahunnya karena penyakit kardiovaskular, yang merupakan 31% dari seluruh kematian global. Diagnosis dini terhadap penyakit jantung sangat penting untuk mengurangi angka kematian dan meningkatkan kualitas hidup pasien. Namun, proses diagnosis manual sering kali memerlukan waktu, tenaga ahli, dan sumber daya yang signifikan.

Dengan kemajuan teknologi machine learning, kini kita dapat membangun model prediksi yang mampu mengidentifikasi kemungkinan penyakit jantung berdasarkan data medis pasien. Model ini dapat menjadi alat bantu bagi tenaga medis untuk mengambil keputusan lebih cepat dan akurat.

**Referensi**:

* World Health Organization. "Cardiovascular diseases (CVDs)." [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29)

## Business Understanding

### Problem Statements

* Bagaimana cara memprediksi risiko penyakit jantung berdasarkan data medis pasien?
* Algoritma machine learning mana yang memberikan performa terbaik untuk klasifikasi penyakit jantung?

### Goals

* Mengembangkan model klasifikasi untuk memprediksi kemungkinan seseorang menderita penyakit jantung.
* Membandingkan performa beberapa algoritma machine learning untuk menentukan model terbaik.

### Solution statements

* Membangun model klasifikasi menggunakan Logistic Regression, Random Forest, Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN).
* Menggunakan metrik evaluasi akurasi, precision, recall, dan f1-score untuk mengukur kinerja model.
* Memilih model terbaik berdasarkan hasil evaluasi untuk digunakan pada tahap deployment.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) yang tersedia di Kaggle. Dataset ini sering digunakan dalam penelitian medis dan machine learning untuk mendeteksi keberadaan penyakit jantung berdasarkan berbagai parameter klinis pasien.

Dataset ini Dataset ini terdiri dari **1025 baris (data pasien)** dan **14 kolom (fitur)**, yang berisi informasi medis pasien seperti tekanan darah, kolesterol, usia, detak jantung, dll.

### Variabel-variabel:

* `age`: usia pasien
* `sex`: jenis kelamin (1 = laki-laki, 0 = perempuan)
* `cp`: tipe nyeri dada (0-3)
* `trestbps`: tekanan darah saat istirahat
* `chol`: kadar kolesterol
* `fbs`: gula darah > 120 mg/dl (1 = benar, 0 = salah)
* `restecg`: hasil elektrokardiogram
* `thalach`: detak jantung maksimal
* `exang`: angina karena olahraga (1 = ya, 0 = tidak)
* `oldpeak`: depres ST saat olahraga
* `slope`: kemiringan segmen ST
* `ca`: jumlah pembuluh darah utama yang diwarnai fluoroskopi (0-3)
* `thal`: kondisi thalassemia
* `target`: 1 = penyakit jantung, 0 = tidak

### Exploratory Data Analysis (EDA)
EDA dilakukan untuk memahami distribusi data dan hubungan antar fitur, khususnya terhadap `target`. Berikut adalah ringkasan temuan berdasarkan visualisasi:

* **Distribusi Target**: Dataset cukup seimbang antara pasien dengan penyakit jantung (`target = 1`) dan tanpa penyakit jantung (`target = 0`), sehingga tidak terjadi imbalance data yang ekstrem.
* **Distribusi Usia**: Usia pasien berkisar antara 29 hingga 77 tahun, dengan distribusi yang relatif normal dan puncaknya di usia sekitar 50-an.
* **Korelasi Fitur (Heatmap)**: Visualisasi korelasi menunjukkan beberapa fitur memiliki hubungan yang cukup signifikan terhadap `target`, antara lain:
    * `cp` (nyeri dada)
    * `thalach` (detak jantung maksimal)
    * `exang` (angina karena olahraga)
    * `oldpeak` (penurunan ST)
* **Nyeri Dada vs Target**: Fitur `cp` menunjukkan hubungan yang kuat dengan `target`, di mana jenis nyeri dada tertentu lebih umum ditemukan pada pasien yang terdiagnosis penyakit jantung.

Analisis ini memberikan wawasan awal yang penting sebelum masuk ke tahap preprocessing dan modeling, termasuk pemilihan fitur relevan serta teknik handling data.

## Data Preparation
Tahapan data preparation dilakukan untuk memastikan data siap digunakan oleh algoritma machine learning dan meminimalkan potensi bias atau kesalahan pada saat pelatihan model.
* **Split Data: Fitur dan Target**
Dataset dipisahkan menjadi:
    * **Fitur** (`x`): Seluruh kolom kecuali `target`
    * **Target** (`y`): Kolom `target`, yaitu label risiko penyakit jantung
* **Train-Test Split**: Data dibagi menjadi data latih dan data uji dengan perbandingan 80:20 menggunakan `train_test_split()` dari Scikit-Learn. Parameter `stratify=y` digunakan untuk menjaga proporsi kelas target antara data latih dan data uji tetap seimbang.
    * Jumlah data latih: **820 baris**
    * Jumlah data uji: **205 baris**
* **Feature Scaling**: Proses **standardisasi fitur** dilakukan agar fitur numerik memiliki skala yang seragam, yaitu dengan mean 0 dan standar deviasi 1. Ini penting terutama untuk algoritma seperti **SVM**, **K-Nearest Neighbors (KNN)**, dan **Logistic Regression**, yang sensitif terhadap skala fitur.

Alasan tahapan ini diperlukan adalah untuk memastikan model dapat dilatih secara optimal dan mencegah bias.

## Model Development

### Model yang digunakan:
1. **Logistic Regression**  
Logistic Regression adalah model linier yang memodelkan probabilitas suatu kejadian menggunakan fungsi sigmoid. Model ini cocok sebagai baseline karena cepat dan mudah diinterpretasikan.
    * **Parameter**:
        * `max_iter=1000`: untuk memastikan konvergensi selama training
        * `random_state=42`: agar hasil replikasi tetap konsisten
    * **Kelebihan**:
        * Cepat, efisien untuk dataset kecil
        * Interpretatif (koefisien menunjukkan pengaruh fitur)
    * **Kekurangan**:
        * Tidak menangkap hubungan non-linear
        * Rentan terhadap multikolinearitas
    * **Hasil**: Akurasi pada data uji sebesar **0.81**
2. **Random Forest**  
Random Forest adalah algoritma ensemble berbasis decision tree yang membentuk banyak pohon dan menggabungkan hasil prediksi mereka (majority voting). Cocok untuk menangani data dengan interaksi non-linier antar fitur.
    * **Parameter**:
        * `random_state=42`: untuk konsistensi hasil
    * **Kelebihan**:
        * Menangani data non-linear
        * Robust terhadap noise dan outlier
    * **Kekurangan**:
        * Bisa overfitting jika tidak diatur
        * Kurang interpretatif dibanding model linier
    * **Hasil**: Akurasi sebesar **1.00**, namun model ini menunjukkan indikasi overfitting karena terlalu sempurna pada data uji.
3. **Support Vector Machine (SVM)**  
SVM bekerja dengan mencari hyperplane optimal yang memisahkan kelas dalam dimensi fitur, sangat efektif untuk data dengan margin pemisah yang jelas.
    * **Parameter**:
        * `random_state=42`: untuk konsistensi
        * Parameter lainnya menggunakan default, termasuk `kernel='rbf'`
    * **Kelebihan**:
        * Performa tinggi di dataset kecil-menengah
        * Efektif untuk data non-linear (dengan kernel)
    * **Kekurangan**:
        * Sensitif terhadap scaling
        * Relatif lambat pada dataset besar
    * **Hasil**: Akurasi sebesar **0.93** — performa tinggi dan stabil tanpa indikasi overfitting.

4. **K-Nearest Neighbors (KNN)**  
KNN mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekat (berdasarkan jarak Euclidean). Sederhana namun bisa sensitif terhadap skala data — oleh karena itu sebelumnya dilakukan standardisasi fitur.
    * **Parameter**:
        * Default (`n_neighbors=5`)
    * **Kelebihan**:
        * Sederhana, tidak perlu training secara eksplisit
        * Adaptif terhadap bentuk data
    * **Kekurangan**:
        * Boros memori dan lambat saat prediksi
        * Sensitif terhadap skala dan outlier
    * **Hasil**: Akurasi sebesar **0.86**

### Pemilihan Model Terbaik
Model terbaik yang dipilih adalah **Support Vector Machine (SVM)** karena:
* Memberikan **akurasi tertinggi tanpa overfitting**
* Lebih stabil dibanding Random Forest yang terlalu overconfident (akurasi 1.00)
* Cocok dengan karakteristik data numerik dan terstandarisasi

## Evaluation

Untuk mengevaluasi performa klasifikasi, digunakan beberapa metrik penting berikut:

### Metrik Evaluasi dan Penjelasannya
* **Accuracy**: (TP + TN) / (TP + FP + TN + FN)  
  Mengukur proporsi prediksi yang benar dari keseluruhan prediksi. Cocok digunakan jika distribusi data antar kelas seimbang.

* **Precision**: TP / (TP + FP)  
  Menunjukkan seberapa banyak prediksi positif yang benar-benar benar. Metrik ini penting ketika kesalahan prediksi positif (false positive) harus dihindari, seperti dalam diagnosis penyakit agar tidak salah mengidentifikasi pasien sehat sebagai sakit.

* **Recall (Sensitivity)**: TP / (TP + FN)   
  Mengukur seberapa baik model dalam menemukan kasus positif. Metrik ini sangat penting dalam kasus di mana kelalaian mendeteksi kasus positif (false negative) dapat berakibat fatal, seperti mendeteksi pasien yang berisiko penyakit jantung.

* **F1 Score**: 2 \* (Precision \* Recall) / (Precision + Recall)
  Merupakan harmonic mean dari Precision dan Recall. Cocok digunakan saat perlu menyeimbangkan keduanya, terutama pada dataset yang tidak seimbang atau ketika false positives dan false negatives sama-sama penting.

> Keterangan:  
> - TP = True Positive  
> - TN = True Negative  
> - FP = False Positive  
> - FN = False Negative  
  

### Hasil Evaluasi Tiap Model

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.81     | 0.76      | 0.91   | 0.83     |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     |
| SVM                 | 0.93     | 0.92      | 0.94   | 0.93     |
| KNN                 | 0.86     | 0.87      | 0.86   | 0.87     |

### Kesimpulan Evaluasi
Berdasarkan hasil evaluasi, **SVM dipilih sebagai model terbaik** karena:
* Memberikan keseimbangan yang baik antara precision dan recall.
* Memiliki performa tinggi (F1-score 0.93) tanpa indikasi overfitting.
* Lebih stabil dan generalizable dibanding Random Forest yang cenderung overfit.
Model SVM kemudian digunakan pada tahap deployment dan pembuatan solusi akhir.


---