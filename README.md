## INFORMASI PROYEK

**Judul Proyek:**  
[Klasifikasi Berita Hoaks Berbahasa Indonesia Menggunakan Machine Learning dan Deep Learning]

**Anggota Kelompok:** 
[1. Rayyan Afif / 244311025
2. Suqya Rohmatin / 244311028]  

**Program Studi:** [Teknologi Rekayasa Perangkat Lunak]  
**Mata Kuliah:** [Data Science]  
**Dosen Pengampu:** [Gus Nanang Syaifuddiin, S.Kom., M.Kom.]  
**Tahun Akademik:** [2026/2027]  
**Link GitHub Repository:** [https://github.com/suqyarohmatin/indonesian-hoax-news-classification]  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** [SDG 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh]  
**Target SDG yang Relevan:** [Target: 16.10, yaitu menjamin akses publik terhadap informasi serta melindungi kebebasan fundamental.]

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

**Catatan untuk mahasiswa:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Pastikan Anda dapat menjelaskan keterkaitan antar tahap, misalnya bagaimana hasil **P** mendorong perbaikan pada **L/M/O** (Bab 7.5).

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang
Penyebaran berita hoaks melalui media sosial telah menjadi ancaman serius bagi stabilitas sosial, politik dan kesehatan masyarakat. Tingginya volume dan kecepatan alur informasi digital menyebabkan masyarakat rentan meneruma serta menyebarkan informasi yang salah sebelum sempat memverifikasi kebenarannya. Pembuktian kebenaran berita secara manual oleh manusia membutuhkan waktu, tenaga, dan sumber daya yang banyak, sehingga sering kali terlambat dibandingkan dengan viralnya berita hoaks tersebut.

Pengembangan Sistem otomatis berbasis Mechine Learning dan Deep Learning untuk mendeteksi berita hoaks menjadi kunci untuk mempercepat proses verifikasi teks. dengan memanfaatkan teknik Natural Language Processing(NLP), model dapat mengenali pola kebahasaan, struktur teks, dan karakteristik semantik yang membedakan artikel berita hoaks dengan berita valid. Sistem ini dirancang untuk alat bantu navigasi informasi awal bagi masyarakat, bukan sebagai pengambil keputusan mutlak atas kebenaran suatu berita.

Implementasi proyek ini mendukung Sustainabel Development Goals(SDG 16), yaitu menciptakan perdamaian,keadilan dan kelembagaan yang tangguh. dengan menyediakan mekanisme otomatisasi filter berita hoaks, masyarakat dapat memperoleh akses informasi yang lebih terpercaya.

**Referensi (berformat APA/IEEE):**
> Jocelynne, C., Wijayakusuma, I. L., & Harini, L. P. I. (2025). Detection of Political Hoax News Using Fine-Tuning IndoBERT. Journal of Applied Informatics and Computing, 9(2), 354–360. https://doi.org/10.30871/jaic.v9i2.8989.
> Rachmawati, O. C. R., & Darmawan, Z. M. E. (2024). The Comparison of Deep Learning Models for Indonesian Political Hoax News Detection. CommIT Journal, 18(2), 123–135. https://doi.org/10.21512/commit.v18i2.10929


### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? |  SDG 16: Peace, Justice, and Strong Institutions (Perdamaian, Keadilan, dan Kelembagaan yang Tangguh)  |
| **Target SDG** | Target (misal 3.4, 13.1) mana yang paling relevan? |  Target 16.10: Menjamin akses publik terhadap informasi dan melindungi kebebasan fundamental, sesuai dengan perundang-undangan nasional dan perjanjian internasional  |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? |  Tingginya tingkat penyebaran hoaks di platform digital Indonesia yang dapat memicu disinformasi, konflik sosial, dan miskonsepsi publik. Pihak yang terdampak adalah masyarakat umum pengguna internet dan media sosial.  |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? |  ML/DL sangat efisien dalam mengekstraksi fitur teks berbahasa Indonesia dalam jumlah besar secara cepat. Model membantu mempercepat klasifikasi awal berita (pre-filtering) sebelum dilakukan verifikasi mendalam.  |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG (langsung/tidak langsung)? |  Kontribusi tidak langsung: Meningkatkan kualitas dan keandalan informasi yang diterima publik, serta meminimalkan penyebaran konten disinformasi yang merusak kebebasan fundamental dan perdamaian sosial.  |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? |  Risiko false positive (berita valid dianggap hoaks) yang dapat membatasi kebebasan berpendapat, serta bias dataset terhadap topik politik tertentu. Hasil prediksi wajib diposisikan sebagai rekomendasi awal, bukan keputusan final.  |

**Sumber ide:** [Ide ini muncul dikarenakan seringnya saya mendengar bahwa nenek saya termakan hoaks yang ia baca dihp]

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini dijalankan melalui alur lifecycle bertahap yang mencakup tahap L (Problem Formulation & Data Preparation), M (Modeling & Baseline Comparison), O (Optimization & Evaluation), hingga P (Deployment & Monitoring). Pada tahap L, masalah dirumuskan ke dalam kerangka binary text classification dan data berita berbahasa Indonesia diproses (cleaning, lowercasing, stopword removal, lemmatization/stemming). Pada tahap M dan O, dikembangkan tiga arsitektur model (Multinomial Naive Bayes sebagai baseline, Support Vector Machine sebagai model advanced, serta Long Short-Term Memory sebagai model Deep Learning) yang dievaluasi menggunakan F1-Score, Precision, Recall, dan Confusion Matrix. Iterasi direncanakan pada tahap pemrosesan fitur (eksperimen TF-IDF vs word embedding) serta penyesuaian hyperparameter untuk mengurangi overfitting dan bias klasifikasi.

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements
1. Efisiensi Verifikasi Teks: Bagaimana membangun model klasifikasi otomatis yang dapat membedakan teks berita hoaks dan valid berbahasa Indonesia secara presisi dan cepat guna mengurangi ketergantungan pada pemeriksaan manual?
2. Kinerja Arsitektur Model: Sejauh mana performa arsitektur Deep Learning (LSTM) dapat mengungguli metode klasik Machine Learning (Multinomial Naive Bayes dan Support Vector Machine) dalam menangkap pola kontekstual dan urutan kata pada teks berita hoaks?
3. Penanganan Unstructured Text & Imbalance Data: Bagaimana menerapkan teknik text preprocessing dan ekstraksi fitur (TF-IDF vs Embedding) yang efektif untuk menangani noise, variasi kosakata, serta potensi ketidakseimbangan kelas (class imbalance) pada dataset berita berbahasa Indonesia?

### 3.2 Formulasi Learning Problem (WAJIB)

Rumuskan masalah dalam kerangka *Learning Problem*:

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, forecasting, clustering, dll.)? |  Target Prediksi: Label kategori berita biner, yaitu Hoaks (1) atau Valid/Non-Hoaks (0). Jenis Tugas: Supervised Learning — Binary Text Classification.  |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis (misal: cross-entropy, MSE)? Metrik keberhasilan utama? |  Fungsi Logika/Rugi: Pengurangan Binary Cross-Entropy Loss (pada model Deep Learning/LSTM) dan optimasi margin/probabilitas (pada SVM dan Naive Bayes). Metrik Utama: F1-Score (karena menyeimbangkan Precision dan Recall pada kasus deteksi hoaks). |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data (i.i.d.? distribusi train vs test?) |  Sumber: Dataset Kaggle (Indonesian Fact and Hoax Political News). Ukuran & Pembagian: Dibagi secara terstruktur untuk training, validation, dan testing. Jenis Data: Teks berita dan narasi berbahasa Indonesia berlabel. Asumsi: Data diasumsikan representatif setelah melalui tahap custom cleaning untuk menghilangkan artefak sumber data.  |

### 3.3 Goals

Tujuan harus spesifik, terukur, dan selaras dengan problem statement.
**Contoh tujuan:**
1. Membangun model ML untuk memprediksi berita hoaks berdasrakan pola kalimat dengan akurasi > 85%
2. Mengukur performa tiga pendekatan model (baseline, advanced, deep learning)
3. Menentukan model terbaik berdasarkan evaluasi test set yang paling andal untuk diimplementasikan.
4. Menghasilkan sistem yang dapat bekerja secara reproducible
5. Mendukung target SDG 16.10 dengan menyediakan sistem penyaringan awal informasi hoaks untuk meningkatkan akses publik terhadap informasi yang valid

### 3.4 Solution Approach  **[M]**

Mahasiswa **WAJIB** menggunakan minimal **tiga model** dengan komposisi sebagai berikut:
#### **Model 1 – Baseline Model**
Model sederhana sebagai pembanding dasar.
**Pilihan model:**
- Multinomial Naive Bayes

**[Alasan Pemilihan]**
Pembanding yang cepat dan efisien untuk klasifikasi teks berbasis probabilitas.

#### **Model 2 – Advanced / ML Model**
Model machine learning yang lebih kompleks.
**Pilihan model:**
- Support Vector Machine (SVM)

**[Alasan Pemilihan]**
Kemampuan yang baik dalam mencari _hyperline_ pemisah optimal pada ruang fitur teks.

#### **Model 3 – Deep Learning Model (WAJIB)**
Model deep learning yang sesuai dengan jenis data.
**Pilihan Implementasi (pilih salah satu sesuai dataset):**

**Text Data:**
- Long Short-Term Memory (LSTM)
- Embedding + Dense Layers

**[Alasan Pemilihan]**
Karena LSTM mampu memproses data teks secara berurutan. Arsitektur ini memiliki _memory cell_ yang dapat mengingat konteks dan hubungan antar kata dalam satu kalimat maupun antar kalimat, sehingga efektif untuk mengenali pola kalimat dan gaya bahasa khas berita hoaks


**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set

## 4. DATA UNDERSTANDING  **[L]**
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[Indonesian Fact and Hoax Political News - Kaggle](https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news)

Dataset yang digunakan merupakan gabungan empat kumpulan data berita berbahasa Indonesia, yaitu CNN Indonesia, Kompas, Tempo, dan TurnBackHoax. File yang digunakan pada tahap ini berasal dari folder `Cleaned` di dalam arsip dataset.

Penggabungan dilakukan menggunakan bahasa Python dengan library Pandas. Setiap dataset diberikan kolom tambahan bernama `source` untuk menunjukkan asal sumber berita. Setelah itu, seluruh dataset digabungkan menjadi satu tabel.

| Sumber dataset | Jumlah baris | Label     |
| -------------- | ------------ | --------- |
| CNN Indonesia  | 9.630        | Valid (0) |
| Kompas         | 4.750        | Valid (0) |
| Tempo          | 6.592        | Valid (0) |
| TurnBackHoax   | 10.381       | Hoaks (1) |
| **Total**      | **31.353**   |           |

**Karakteristik dataset:**

| Komponen                                 | Keterangan                 |
| ---------------------------------------- | -------------------------- |
| Jumlah baris awal                        | 31.353                     |
| Jumlah kolom pada hasil penggabungan     | 13 kolom                   |
| Jumlah kolom file CNN, Kompas, dan Tempo | 9 kolom                    |
| Jumlah kolom file TurnBackHoax           | 11 kolom                   |
| Tipe data                                | Teks (text) dan metadata   |
| Format file                              | XLSX                       |
| Bahasa                                   | Indonesia                  |
| Jenis pembelajaran                       | Supervised Learning        |
| Jenis tugas                              | Binary Text Classification |
| Target prediksi                          | `hoax`                     |
| Label 0                                  | Berita valid/non-hoaks     |
| Label 1                                  | Berita hoaks               |

Dataset termasuk data teks tidak terstruktur (*unstructured text*) karena informasi utama yang digunakan dalam proses klasifikasi berupa judul dan isi artikel. Selain itu, dataset juga memiliki beberapa kolom metadata seperti waktu publikasi, penulis, tag, URL, dan sumber berita.

> **Catatan sumber:** Label 0 terdapat pada dataset CNN Indonesia, Kompas, dan Tempo, sedangkan label 1 terdapat pada dataset TurnBackHoax. Oleh karena itu, label dalam dataset ini berkaitan dengan sumber asal data. Kondisi ini menjadi salah satu keterbatasan dataset yang perlu diperhatikan ketika melakukan pemodelan dan menginterpretasikan hasil penelitian.

## 4.2 Deskripsi Fitur

Kolom yang tersedia tidak sepenuhnya sama pada keempat file. CNN Indonesia, Kompas, dan Tempo memiliki sembilan kolom, sedangkan TurnBackHoax memiliki sebelas kolom. Setelah seluruh dataset digabungkan, terdapat 13 kolom karena beberapa kolom hanya tersedia pada sumber tertentu dan terdapat tambahan kolom `source`.

Berikut deskripsi kolom berdasarkan hasil penggabungan dataset:

| Nama fitur     | Tipe data awal | Deskripsi                                                               |
| -------------- | -------------- | ----------------------------------------------------------------------- |
| `Unnamed: 0`   | Integer        | Kolom indeks dari file asal; tidak digunakan sebagai fitur model        |
| `Title`        | String/Object  | Judul artikel                                                           |
| `Timestamp`    | String/Object  | Waktu publikasi artikel                                                 |
| `FullText`     | String/Object  | Isi artikel                                                             |
| `Tags`         | String/Object  | Tag atau kategori artikel                                               |
| `Author`       | String/Object  | Nama penulis atau pihak yang tercantum sebagai penulis                  |
| `Url`          | String/Object  | Alamat sumber artikel                                                   |
| `text_new`     | String/Object  | Teks hasil pemrosesan yang tersedia pada dataset CNN, Kompas, dan Tempo |
| `hoax`         | Integer        | Label target klasifikasi: 0 atau 1                                      |
| `source`       | String/Object  | Sumber asal dataset yang ditambahkan saat proses penggabungan           |
| `politik`      | Float          | Penanda kategori politik pada dataset TurnBackHoax                      |
| `Narasi`       | String/Object  | Narasi klaim pada dataset TurnBackHoax                                  |
| `Clean Narasi` | String/Object  | Narasi yang telah dibersihkan pada dataset TurnBackHoax                 |

**Fitur yang digunakan untuk pemodelan:**

Fitur utama yang direncanakan untuk proses klasifikasi adalah:

- `Title`: judul artikel.
- `FullText`: isi artikel.
- `hoax`: label target klasifikasi.

Judul dan isi artikel digabungkan menjadi satu fitur teks baru bernama `text_raw`. Selanjutnya, teks tersebut dibersihkan dan disimpan dalam kolom `text_clean` sebagai input utama untuk proses pemodelan.

Kolom `hoax` digunakan sebagai target atau label yang akan diprediksi oleh model.

Sementara itu, kolom metadata seperti URL, penulis, waktu publikasi, tag, dan sumber dataset tidak digunakan sebagai fitur utama pemodelan. Kolom `source` hanya digunakan untuk mengetahui asal dataset dan membantu pemeriksaan karakteristik data.

> **Catatan:** Kolom `text_new`, `Narasi`, dan `Clean Narasi` tidak tersedia secara seragam pada seluruh sumber. Oleh karena itu, pemrosesan awal menggunakan `Title` dan `FullText` yang tersedia pada keempat dataset. Namun, isi `FullText` TurnBackHoax perlu ditinjau lebih lanjut karena dapat berupa artikel pemeriksaan fakta, bukan teks asli berita hoaks yang beredar.

## 4.3 Kondisi Data

### 4.3.1 Missing Values

Berdasarkan hasil pemeriksaan awal setelah empat dataset digabungkan, ditemukan beberapa nilai kosong pada sejumlah kolom.

| Nama kolom     | Jumlah missing values |
| -------------- | --------------------- |
| `Unnamed: 0`   | 0                     |
| `Title`        | 21                    |
| `Timestamp`    | 21                    |
| `FullText`     | 27                    |
| `Tags`         | 162                   |
| `Author`       | 337                   |
| `Url`          | 0                     |
| `text_new`     | 10.408                |
| `hoax`         | 0                     |
| `source`       | 0                     |
| `politik`      | 20.972                |
| `Narasi`       | 20.972                |
| `Clean Narasi` | 24.851                |

Nilai kosong pada kolom `Title` dan `FullText` perlu ditangani karena kedua kolom tersebut merupakan sumber teks utama yang digunakan dalam proses klasifikasi.

Sementara itu, beberapa nilai kosong pada kolom `text_new`, `politik`, `Narasi`, dan `Clean Narasi` terjadi karena kolom tersebut memang tidak tersedia pada seluruh sumber dataset.

Kolom metadata yang tidak digunakan sebagai fitur utama tidak perlu dipaksakan untuk diisi. Penanganan missing values difokuskan pada kolom yang benar-benar diperlukan untuk pemodelan, yaitu `Title`, `FullText`, dan `hoax`.

### 4.3.2 Duplicate Data

Pemeriksaan awal pada masing-masing file menunjukkan bahwa tidak ditemukan baris duplikat dan URL duplikat dalam setiap dataset.

Namun, setelah seluruh dataset digabungkan, dilakukan pemeriksaan ulang berdasarkan kesamaan judul dan isi berita.

Pemeriksaan dilakukan dengan mengubah teks menjadi huruf kecil dan merapikan spasi agar perbedaan penulisan kecil tidak dianggap sebagai data yang berbeda.

Hasil pemeriksaan menemukan **5 data duplikat** berdasarkan kombinasi judul dan isi berita. Data duplikat tersebut kemudian dihapus pada tahap data cleaning.

Dengan demikian, jumlah data berkurang dari 31.326 setelah penanganan missing values menjadi 31.321 setelah penghapusan duplikat.

### 4.3.3 Outliers

Outlier numerik tidak menjadi fokus utama karena fitur utama penelitian berupa teks.

Namun, dilakukan pemeriksaan terhadap panjang isi berita berdasarkan jumlah kata dalam kolom `FullText`. Pemeriksaan menggunakan metode *Interquartile Range* (IQR).

Hasil pemeriksaan menunjukkan:

| Keterangan                      | Hasil      |
| ------------------------------- | ---------- |
| Batas bawah                     | 13,5 kata  |
| Batas atas                      | 633,5 kata |
| Jumlah data terindikasi outlier | 1.756      |

Sebanyak 1.756 data terindikasi memiliki jumlah kata di luar batas IQR. Data tersebut tidak dihapus secara otomatis karena artikel yang terlalu pendek atau terlalu panjang belum tentu merupakan data yang salah atau tidak valid.

Oleh karena itu, data tetap dipertahankan agar informasi yang terkandung dalam artikel tidak hilang.

### 4.3.4 Imbalanced Data

Berdasarkan distribusi label awal, jumlah data berita valid lebih banyak dibandingkan data berita hoaks.

Distribusi label sebelum proses cleaning adalah sebagai berikut:

| Kelas     | Jumlah     | Persentase |
| --------- | ---------- | ---------- |
| Valid (0) | 20.972     | 66,89%     |
| Hoaks (1) | 10.381     | 33,11%     |
| **Total** | **31.353** | **100%**   |

Distribusi tersebut menunjukkan adanya ketidakseimbangan kelas. Jumlah data valid lebih banyak dibandingkan jumlah data hoaks.

Ketidakseimbangan kelas perlu diperhatikan karena model berpotensi lebih banyak mempelajari kelas mayoritas. Oleh karena itu, pada tahap data preparation digunakan pendekatan *Class Weight* untuk memberikan bobot lebih besar pada kelas yang memiliki jumlah data lebih sedikit.

Selain accuracy, metrik evaluasi seperti Precision, Recall, dan F1-Score juga perlu diperhatikan untuk melihat performa model pada masing-masing kelas.

### 4.3.5 Noise dan Kualitas Data

Teks berita yang berasal dari beberapa sumber memiliki kemungkinan perbedaan format penulisan, tanda baca, karakter khusus, URL, tag HTML, serta penggunaan spasi yang tidak konsisten.

Selain itu, terdapat perbedaan gaya penulisan antara artikel berita dari portal berita dan artikel pemeriksaan fakta dari TurnBackHoax.

Pada pemeriksaan judul TurnBackHoax juga ditemukan beberapa penanda kategori, seperti `[SALAH]`, `[BENAR]`, `[HOAX]`, `[KLARIFIKASI]`, dan `[DISINFORMASI]`.

Hasil pemeriksaan menemukan 425 judul dengan penanda `[BENAR]`, meskipun seluruh data TurnBackHoax diberikan label 1 atau hoaks. Kondisi tersebut menunjukkan adanya kemungkinan ketidaksesuaian antara penanda kategori dengan label yang digunakan.

Oleh karena itu, kualitas label perlu diperiksa lebih lanjut agar model tidak mempelajari informasi yang keliru.

### 4.3.6 Asumsi Data dan Potensi Data Leakage

Dataset diasumsikan sebagai data berlabel untuk kebutuhan *supervised learning*. Model akan mempelajari hubungan antara teks berita sebagai input dan label `hoax` sebagai target klasifikasi.

Namun, terdapat beberapa risiko yang perlu diperhatikan.

Pertama, seluruh data valid berasal dari CNN Indonesia, Kompas, dan Tempo, sedangkan seluruh data hoaks berasal dari TurnBackHoax. Kondisi ini menyebabkan label memiliki hubungan yang sangat kuat dengan sumber asal berita.

Akibatnya, model berpotensi mempelajari karakteristik gaya penulisan atau format artikel dari masing-masing sumber, bukan benar-benar mengenali karakteristik berita hoaks dan berita valid.

Kedua, terdapat penanda kategori pada judul TurnBackHoax, seperti `[SALAH]`, `[BENAR]`, dan `[HOAX]`. Penanda tersebut berpotensi memberikan petunjuk langsung terhadap label klasifikasi.

Untuk mengurangi risiko kebocoran informasi dari penanda judul, pada tahap feature engineering dilakukan penghapusan penanda dalam tanda kurung siku yang terdapat di awal judul.

Selain itu, kolom metadata seperti URL, nama penulis, tag, waktu publikasi, dan sumber dataset tidak digunakan sebagai fitur utama model.

Pembagian data menggunakan *stratified split* untuk menjaga proporsi kelas valid dan hoaks pada data training, validation, dan test. Meskipun demikian, metode tersebut belum sepenuhnya mengatasi bias berdasarkan sumber dataset.

Evaluasi lanjutan dengan data dari sumber yang lebih beragam dan pemisahan berdasarkan sumber berita akan lebih kuat untuk mengukur kemampuan generalisasi model.

## 4.4 Exploratory Data Analysis (EDA) - OPSIONAL

Exploratory Data Analysis (EDA) merupakan proses pemeriksaan dan pemahaman karakteristik data sebelum dilakukan pemodelan.

Pada notebook yang dijalankan, pemeriksaan awal telah dilakukan melalui informasi tipe data, jumlah missing values, distribusi label berdasarkan sumber dataset, serta pemeriksaan penanda kategori pada judul TurnBackHoax.

Namun, notebook saat ini belum menghasilkan tiga visualisasi grafis. Oleh karena itu, belum terdapat gambar plot yang dapat dimasukkan sebagai hasil EDA.

Beberapa visualisasi yang dapat digunakan pada penelitian ini adalah:

### Visualisasi 1: Distribusi Kelas Berita Valid dan Hoaks

**Status:** Data distribusi sudah diperiksa melalui tabel jumlah kelas, tetapi grafik belum dibuat.

**Insight berdasarkan hasil aktual:**

Jumlah berita valid sebanyak 20.972 atau 66,89%, sedangkan berita hoaks sebanyak 10.381 atau 33,11%. Hal ini menunjukkan bahwa dataset memiliki ketidakseimbangan kelas dengan jumlah berita valid sebagai kelas mayoritas.

### Visualisasi 2: Distribusi Jumlah Kata dalam Artikel

**Status:** Pemeriksaan panjang teks sudah dilakukan melalui metode IQR, tetapi histogram atau boxplot belum dibuat.

**Insight berdasarkan hasil aktual:**

Ditemukan 1.756 data yang terindikasi sebagai outlier berdasarkan jumlah kata pada kolom `FullText`. Data tersebut tetap dipertahankan karena panjang artikel tidak selalu menunjukkan kualitas atau kebenaran informasi.

### Visualisasi 3: Distribusi Kelas Berdasarkan Sumber Dataset

**Status:** Pemeriksaan sudah dilakukan menggunakan tabel silang (*cross tabulation*), tetapi grafik belum dibuat.

**Insight berdasarkan hasil aktual:**

CNN Indonesia, Kompas, dan Tempo seluruhnya memiliki label valid, sedangkan TurnBackHoax seluruhnya memiliki label hoaks. Hal ini menunjukkan adanya keterkaitan antara sumber dataset dengan label klasifikasi yang dapat menimbulkan bias pada proses pemodelan.

> **Catatan:** Karena EDA bersifat opsional, hasil pemeriksaan tabel di atas dapat dijelaskan dalam laporan. Namun, apabila dosen meminta minimal tiga grafik, visualisasi tersebut harus dibuat terlebih dahulu sebelum gambar dan hasilnya dicantumkan dalam laporan.

---

# 5. DATA PREPARATION

Bagian ini menjelaskan tahapan transformasi dan preprocessing yang dilakukan terhadap dataset sebelum digunakan dalam proses pemodelan.

## 5.1 Data Cleaning

Data cleaning dilakukan untuk memastikan data yang digunakan memiliki struktur dan kualitas yang sesuai untuk proses klasifikasi.

Tahapan yang dilakukan meliputi:

### 1. Handling Missing Values

Pada pemeriksaan awal ditemukan 21 nilai kosong pada kolom `Title` dan 27 nilai kosong pada kolom `FullText`.

Strategi yang digunakan adalah menghapus baris yang memiliki nilai kosong pada kolom `Title`, `FullText`, atau `hoax` menggunakan fungsi `dropna()`.

Penghapusan dipilih karena judul dan isi berita merupakan informasi utama yang dibutuhkan untuk proses klasifikasi. Apabila kedua kolom tersebut kosong, data tidak dapat memberikan informasi teks yang memadai.

Hasilnya, sebanyak 27 baris data dihapus sehingga jumlah data berkurang dari 31.353 menjadi 31.326 baris.

Nilai kosong pada kolom metadata yang tidak digunakan sebagai fitur utama tidak diisi karena tidak diperlukan dalam proses pemodelan.

### 2. Removing Duplicates

Penghapusan data duplikat dilakukan berdasarkan kombinasi judul dan isi berita.

Sebelum dibandingkan, teks pada kolom `Title` dan `FullText` diubah menjadi huruf kecil dan spasi berulang dirapikan. Langkah ini dilakukan agar perbedaan penulisan kecil tidak menyebabkan data yang sama dianggap berbeda.

Penghapusan dilakukan menggunakan fungsi `drop_duplicates()`.

Hasilnya, ditemukan dan dihapus 5 data duplikat. Jumlah data setelah proses ini menjadi 31.321 baris.

### 3. Handling Outliers

Pemeriksaan outlier dilakukan berdasarkan jumlah kata pada kolom `FullText`.

Metode yang digunakan adalah *Interquartile Range* (IQR), yaitu metode untuk menentukan data yang berada di luar rentang umum distribusi.

Hasil pemeriksaan menemukan 1.756 data yang terindikasi sebagai outlier dengan batas bawah 13,5 kata dan batas atas 633,5 kata.

Data tersebut tidak dihapus karena panjang artikel yang berbeda tidak selalu menunjukkan kesalahan data. Artikel pendek maupun panjang masih dapat mengandung informasi yang relevan.

### 4. Data Type Conversion

Konversi tipe data dilakukan untuk memastikan kolom utama memiliki tipe data yang sesuai.

Kolom `Title` dan `FullText` dikonversi menjadi tipe data teks atau string, sedangkan kolom `hoax` dipastikan berupa angka bertipe integer.

Hal ini diperlukan agar proses pembersihan teks dan pengolahan label dapat dilakukan dengan benar.

**Hasil akhir Data Cleaning:**

| Keterangan                                          | Jumlah     |
| --------------------------------------------------- | ---------- |
| Data awal                                           | 31.353     |
| Data dihapus karena missing value pada kolom wajib | 27         |
| Data setelah penanganan missing value               | 31.326     |
| Data duplikat yang dihapus                          | 5          |
| **Data setelah cleaning**                           | **31.321** |

Distribusi label setelah proses cleaning:

| Kelas     | Jumlah     | Persentase |
| --------- | ---------- | ---------- |
| Valid (0) | 20.945     | 66,87%     |
| Hoaks (1) | 10.376     | 33,13%     |
| **Total** | **31.321** | **100%**   |

## 5.2 Feature Engineering

Feature engineering dilakukan untuk membentuk fitur baru yang dapat digunakan sebagai input dalam proses klasifikasi berita.

Tahapan yang dilakukan adalah sebagai berikut:

### 1. Menghapus Penanda pada Awal Judul

Pada dataset TurnBackHoax terdapat penanda kategori di awal judul, seperti `[SALAH]`, `[BENAR]`, dan `[HOAX]`.

Penanda tersebut dihapus dari awal judul karena dapat memberikan petunjuk langsung mengenai kategori berita kepada model.

Proses ini menghasilkan kolom baru bernama `judul_tanpa_penanda`.

Penghapusan dilakukan menggunakan regular expression atau Regex.

### 2. Menggabungkan Judul dan Isi Berita

Kolom `Title` dan `FullText` digabungkan menjadi satu kolom baru bernama `text_raw`.

Penggabungan dilakukan agar model dapat memperoleh informasi dari judul sekaligus isi artikel, bukan hanya salah satunya.

### 3. Membuat Fitur Tambahan

Selain menggabungkan teks, dibuat dua fitur tambahan, yaitu:

- `jumlah_karakter`: menghitung jumlah karakter dalam teks berita.
- `jumlah_kata`: menghitung jumlah kata dalam teks berita.

Fitur tersebut digunakan untuk mengetahui karakteristik panjang teks. Namun, pada notebook ini kedua fitur tersebut belum digunakan sebagai input utama model.

### 4. Memisahkan Input dan Target

Setelah fitur teks terbentuk, dilakukan pemisahan antara variabel input dan target.

- `X` merupakan variabel input yang berisi teks berita.
- `y` merupakan variabel target yang berisi label `hoax`.

Input akhir diperbarui menjadi `text_clean` setelah melewati tahap data transformation.

## 5.3 Data Transformation

Data transformation dilakukan untuk mengubah teks berita menjadi format yang dapat diproses oleh algoritma Machine Learning dan Deep Learning.

Karena penelitian menggunakan data teks, transformasi yang dilakukan disesuaikan dengan kebutuhan masing-masing algoritma.

### 1. Text Cleaning

Sebelum teks diubah menjadi angka, dilakukan pembersihan teks menggunakan fungsi `clean_text()`.

Langkah-langkah yang dilakukan meliputi:

- **Lowercasing:** mengubah seluruh huruf menjadi huruf kecil.
- **Removing URL:** menghapus alamat URL dari teks.
- **Removing HTML tags:** menghapus tag HTML.
- **Removing punctuation and symbols:** menghapus tanda baca dan simbol yang tidak diperlukan.
- **Removing extra whitespace:** merapikan spasi berlebih.

Hasil dari proses ini disimpan dalam kolom `text_clean`.

Proses ini dilakukan agar format teks menjadi lebih konsisten dan mengurangi karakter yang tidak diperlukan dalam pemodelan.

Pada tahap ini belum dilakukan penghapusan stopwords, stemming, lemmatization, maupun penggunaan Word2Vec, GloVe, atau fastText.

### 2. TF-IDF untuk Naive Bayes dan SVM

Untuk algoritma Multinomial Naive Bayes dan Support Vector Machine (SVM), teks diubah menjadi bentuk numerik menggunakan metode Term Frequency-Inverse Document Frequency (TF-IDF).

TF-IDF digunakan untuk memberikan bobot pada kata berdasarkan frekuensi kemunculan kata dalam dokumen dan tingkat keumumannya dalam keseluruhan kumpulan dokumen.

Parameter yang digunakan adalah:

| Parameter      | Nilai  | Keterangan                                    |
| -------------- | ------ | --------------------------------------------- |
| `max_features` | 20.000 | Membatasi jumlah fitur atau kosakata maksimum |
| `ngram_range`  | (1, 2) | Menggunakan unigram dan bigram                |

Unigram merupakan fitur satu kata, sedangkan bigram merupakan kombinasi dua kata yang berurutan.

Proses `fit_transform()` hanya dilakukan pada data training untuk membentuk kosakata dan bobot TF-IDF. Sementara itu, data validation dan test hanya diubah menggunakan `transform()` dari hasil training.

Strategi ini digunakan untuk menghindari kebocoran informasi dari data validation dan test ke dalam proses pembentukan fitur.

**Hasil transformasi TF-IDF:**

| Jenis data | Jumlah sampel | Jumlah fitur |
| ---------- | ------------- | ------------ |
| Training   | 21.924        | 20.000       |
| Validation | 4.698         | 20.000       |
| Test       | 4.699         | 20.000       |

### 3. Tokenization dan Padding untuk LSTM

Untuk algoritma Long Short-Term Memory (LSTM), teks diubah menjadi urutan angka menggunakan proses tokenisasi.

Tokenisasi dilakukan menggunakan `Tokenizer` dari TensorFlow Keras.

Parameter yang digunakan adalah:

| Parameter    | Nilai   | Keterangan                            |
| ------------ | ------- | ------------------------------------- |
| `MAX_WORDS`  | 20.000  | Batas jumlah kata dalam kamus         |
| `MAX_LEN`    | 300     | Panjang maksimum urutan angka         |
| `oov_token`  | `<OOV>` | Penanda untuk kata yang tidak dikenal |
| `padding`    | `post`  | Menambahkan angka nol di akhir urutan |
| `truncating` | `post`  | Memotong urutan berlebih dari bagian akhir |

Tokenizer dipelajari hanya dari data training. Setelah itu, data training, validation, dan test diubah menjadi urutan angka menggunakan kamus yang sama.

Selanjutnya dilakukan padding untuk menyamakan panjang setiap urutan menjadi 300 angka. Proses ini diperlukan agar seluruh data memiliki panjang input yang seragam ketika digunakan oleh LSTM.

**Hasil transformasi untuk LSTM:**

| Jenis data | Bentuk data   |
| ---------- | ------------- |
| Training   | (21.924, 300) |
| Validation | (4.698, 300)  |
| Test       | (4.699, 300)  |

Dengan demikian, penelitian menggunakan dua pendekatan transformasi teks, yaitu TF-IDF untuk Naive Bayes dan SVM, serta tokenisasi dan padding untuk LSTM.

## 5.4 Data Splitting

Data splitting dilakukan untuk membagi dataset menjadi data training, validation, dan test.

Strategi yang digunakan adalah pembagian data dengan perbandingan 70:15:15.

| Jenis data | Persentase | Jumlah sampel |
| ---------- | ---------- | ------------- |
| Training   | 70%        | 21.924        |
| Validation | 15%        | 4.698         |
| Test       | 15%        | 4.699         |
| **Total**  | **100%**   | **31.321**    |

Pembagian dilakukan menggunakan fungsi `train_test_split()` dari library Scikit-learn.

Metode `stratify` digunakan untuk mempertahankan proporsi kelas valid dan hoaks pada masing-masing bagian data.

Selain itu, digunakan `random_state=42` agar proses pembagian data dapat menghasilkan hasil yang konsisten ketika kode dijalankan kembali.

Data training digunakan untuk proses pembelajaran model. Data validation disiapkan untuk membantu pemilihan dan penyetelan model pada tahap berikutnya, sedangkan data test disiapkan untuk evaluasi akhir.

Pembentukan kosakata TF-IDF dan tokenizer hanya dipelajari dari data training. Data validation dan test hanya ditransformasikan menggunakan hasil yang diperoleh dari training untuk menghindari data leakage.

## 5.5 Data Balancing

Data balancing dilakukan untuk menangani ketidakseimbangan jumlah data antara kelas berita valid dan berita hoaks.

Berdasarkan hasil pembagian data, distribusi kelas pada data training adalah sebagai berikut:

| Kelas     | Jumlah data training |
| --------- | -------------------- |
| Valid (0) | 14.661               |
| Hoaks (1) | 7.263                |
| **Total** | **21.924**           |

Jumlah data valid lebih banyak dibandingkan jumlah data hoaks. Oleh karena itu, digunakan pendekatan *Class Weight*.

Class Weight merupakan teknik yang memberikan bobot berbeda terhadap masing-masing kelas ketika model melakukan proses pembelajaran. Kelas yang jumlah datanya lebih sedikit diberikan bobot lebih besar agar model memberikan perhatian yang lebih terhadap kelas tersebut.

Perhitungan bobot dilakukan menggunakan fungsi `compute_class_weight()` dari Scikit-learn dengan parameter `class_weight='balanced'`.

Hasil perhitungan bobot kelas adalah:

| Kelas     | Bobot  |
| --------- | ------ |
| Valid (0) | 0,7477 |
| Hoaks (1) | 1,5093 |

Berdasarkan hasil tersebut, kelas hoaks memperoleh bobot lebih besar dibandingkan kelas valid karena jumlah datanya lebih sedikit.

Pada notebook ini, bobot kelas telah dihitung dan disimpan dalam variabel `class_weight_dict` untuk digunakan pada tahap pelatihan model berikutnya.

> **Catatan:** Class Weight tidak mengubah jumlah data atau menambahkan sampel baru. Teknik ini hanya menyiapkan bobot yang nantinya digunakan ketika model dilatih.

## 5.6 Ringkasan Data Preparation

Tahapan data preparation dilakukan untuk memastikan data memiliki kualitas dan format yang sesuai sebelum digunakan dalam proses pemodelan.

Berikut ringkasan proses yang dilakukan:

| Tahapan             | Apa yang dilakukan?                                                                                                     | Mengapa dilakukan?                                                                    | Bagaimana implementasinya?                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Data Cleaning       | Menangani missing values, menghapus duplikat, memeriksa outlier, dan mengonversi tipe data                              | Memastikan data yang digunakan memiliki informasi yang memadai dan format yang sesuai | Menggunakan `dropna()`, `drop_duplicates()`, metode IQR, dan konversi tipe data Pandas |
| Feature Engineering | Menghapus penanda kategori pada judul, menggabungkan judul dan isi berita, serta membuat fitur jumlah kata dan karakter | Menghasilkan fitur teks yang lebih sesuai untuk proses klasifikasi                    | Menggunakan `str.replace()`, penggabungan kolom, `str.len()`, dan `str.split()`        |
| Data Transformation | Membersihkan teks dan mengubah teks menjadi bentuk numerik                                                              | Algoritma Machine Learning dan Deep Learning membutuhkan input dalam bentuk numerik   | Menggunakan Regex, TF-IDF, Tokenizer, dan Padding                                      |
| Data Splitting      | Membagi data menjadi training, validation, dan test                                                                     | Memisahkan data untuk pembelajaran, validasi, dan evaluasi akhir                      | Menggunakan `train_test_split()` dengan stratified split dan random state 42           |
| Data Balancing      | Menghitung bobot masing-masing kelas                                                                                    | Membantu mengurangi dampak ketidakseimbangan kelas ketika model dilatih               | Menggunakan `compute_class_weight()` dengan metode balanced                            |

### Kesimpulan Data Preparation

Berdasarkan tahapan data preparation, jumlah dataset berkurang dari 31.353 menjadi 31.321 data setelah proses penanganan missing values dan penghapusan duplikat.

Selanjutnya, fitur teks dibentuk dari gabungan judul dan isi berita, kemudian dibersihkan dan ditransformasikan menggunakan dua pendekatan sesuai kebutuhan algoritma yang digunakan.

Data dibagi menjadi 70% training, 15% validation, dan 15% test. Selain itu, dilakukan perhitungan bobot kelas untuk mengatasi ketidakseimbangan jumlah berita valid dan hoaks.

Dengan demikian, dataset telah melalui tahapan persiapan hingga siap digunakan pada tahap pemodelan menggunakan Multinomial Naive Bayes, Support Vector Machine (SVM), dan Long Short-Term Memory (LSTM).

Namun, kualitas label dan keterkaitan antara sumber berita dengan label masih menjadi keterbatasan dataset yang perlu diperhatikan dalam interpretasi hasil pemodelan.

---

## 6. MODELING  **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Logistic Regression]  
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]  
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: model linear]
- **Hypothesis space:** [himpunan fungsi yang dapat direpresentasikan, misal: batas keputusan linear]
- **Inductive bias / asumsi:** [misal: hubungan fitur-target linear, fitur independen]
- **Implikasi terhadap dataset Anda:** [apakah asumsi cocok dengan data?]

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**

**Objective/loss yang dioptimalkan:** [misal: log-loss + L2]  
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.5 Hasil Awal  **[O → P]**

**[Tuliskan hasil awal pada validation set; evaluasi akhir pada test set dijelaskan di Bab 7]**

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]  
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: ensemble pohon keputusan]
- **Hypothesis space:** [fungsi non-linear piecewise, interaksi fitur otomatis, dll.]
- **Inductive bias / asumsi:** [misal: data tabular dengan struktur ambang/threshold]
- **Perbandingan dengan baseline:** [apa yang bisa dipelajari model ini yang tidak bisa baseline?]

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Search space: [rentang parameter yang dicoba]
- Skema validasi: [k-fold CV / validation set]
- Best parameters: [...]
- Skor validasi sebelum vs sesudah tuning: [...]

#### 6.2.5 Kontrol Overfitting  **[O]**
[Jelaskan regularisasi/pembatasan kompleksitas yang dipakai, misal: max_depth, min_samples_leaf, early stopping pada boosting]

#### 6.2.6 Hasil Model  **[O → P]**

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**

**Inductive bias arsitektur:** [misal: CNN → lokalitas & translation invariance; LSTM → ketergantungan sekuensial; MLP → tanpa asumsi struktur]  
**Hypothesis space:** [kapasitas model: jumlah layer, unit, parameter]

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
- Regularisasi: [Dropout / L2 / BatchNorm / augmentasi]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Exp-2 | [ ] | [ ] | [ ] | [ ] | [ ] |

#### 6.3.5 Implementasi (Ringkas)  **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| [misal: recall kelas minoritas rendah] | [L] data | [misal: class weight / SMOTE] | [ ... ] |
| [misal: gap train-val besar] | [O] optimasi | [misal: tambah dropout, early stopping] | [ ... ] |
| [misal: model terlalu sederhana] | [M] desain | [misal: tambah layer / ganti arsitektur] | [ ... ] |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | [ ... ] | [ ... ] |
| **M** – Model Design | [ ... ] | [ ... ] |
| **O** – Optimization | [ ... ] | [ ... ] |
| **P** – Predict & Evaluate | [ ... ] | [ ... ] |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**
```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2** | **L** – Data | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5) | Notebook data preparation + repo GitHub dibuat |
| **3** | **M** – Model Design | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5) | Model DL berjalan end-to-end (training awal) |
| **5** | **O** – Optimization | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7) | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10) | Laporan final + GitHub + video + presentasi |

**Catatan untuk dosen:**
- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**
- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia) | Contoh arah ide ML (inspirasi) |
|-----|------------------|--------------------------------|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| 3 | Kehidupan Sehat dan Sejahtera | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman yang Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi dan Produksi yang Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ ... ] |
| Judul ide | [ ... ] |
| SDG & target yang dipilih | [ ... ] |
| Masalah nyata yang diangkat | [ ... ] |
| Tugas ML (Target & jenis tugas) | [ ... ] |
| Objective & metrik utama | [ ... ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ ... ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ] |
| Risiko & pertimbangan etika (bias, privasi) | [ ... ] |
| Referensi awal (minimal 2) | [ ... ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
