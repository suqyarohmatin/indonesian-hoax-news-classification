# KLASIFIKASI JUDUL BERITA HOAKS BERBAHASA INDONESIA MENGGUNAKAN MACHINE LEARNING DAN DEEP LEARNING

## INFORMASI PROYEK

**Judul Proyek:**  
Klasifikasi Berita Hoaks Berbahasa Indonesia Menggunakan Machine Learning dan Deep Learning

**Anggota Kelompok:**
1. Rayyan Afif / 244311025
2. Suqya Rohmatin / 244311028

**Program Studi:** Teknologi Rekayasa Perangkat Lunak  
**Mata Kuliah:** Data Science  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.  
**Tahun Akademik:** 2026/2027  
**Link GitHub Repository:** https://github.com/suqyarohmatin/indonesian-hoax-news-classification  
**Link Video Pembahasan:** Belum tersedia.  
**SDG yang Dipilih:** SDG 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh.  
**Target SDG yang Relevan:** Target 16.10, yaitu menjamin akses publik terhadap informasi serta melindungi kebebasan fundamental.

---

# 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Proyek ini disusun mengikuti kerangka **Machine Learning Lifecycle** yang terdiri dari empat tahapan utama yang saling berkaitan dan dapat dilakukan secara berulang (iteratif).

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|---|---|---|---|
| **Learning Problem** | **L** | Apa target, objective, dan data yang digunakan? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model atau arsitektur apa yang digunakan? Bagaimana hypothesis space dan inductive bias-nya? | Bab 6 |
| **Optimization** | **O** | Bagaimana model dilatih dan dioptimasi melalui loss function, optimizer, hyperparameter tuning, dan regularisasi? | Bab 6 |
| **Predict & Evaluate** | **P** | Seberapa baik hasil prediksi model berdasarkan metrik evaluasi? | Bab 7 |

### Alur Machine Learning Lifecycle

```text
L (Learning Problem)
        |
        v
M (Model Design)
        |
        v
O (Optimization)
        |
        v
P (Predict & Evaluate)
        |
        v
Iterasi / Perbaikan Berdasarkan Hasil Evaluasi
        |
        +---------------------> Kembali ke Tahap L, M, atau O
```

Pada proyek ini, tahap Learning Problem telah dilakukan melalui perumusan masalah, pengumpulan dataset, pemeriksaan data, EDA, dan data preparation.

Tahap Model Design dan Optimization direncanakan menggunakan tiga pendekatan, yaitu Multinomial Naive Bayes, Support Vector Machine (SVM), dan Long Short-Term Memory (LSTM). Implementasi yang sudah tersedia pada notebook saat ini adalah model SVM.

Hasil evaluasi nantinya digunakan sebagai dasar untuk melakukan perbaikan terhadap dataset, pemilihan model, dan konfigurasi optimasi.

---

# 1. LEARNING OUTCOMES

Pada proyek ini, mahasiswa diharapkan dapat:

1. **[L]** Mengidentifikasi permasalahan nyata dari ide sendiri yang selaras dengan salah satu tujuan Sustainable Development Goals (SDGs), kemudian merumuskan problem statement yang terdiri dari target, objective, dan data secara jelas.
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) untuk memahami karakteristik dataset yang digunakan.
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset teks berbahasa Indonesia.
4. **[M]** Merancang tiga pendekatan model Machine Learning dengan justifikasi hypothesis space dan inductive bias, yaitu:
   - Model baseline: Multinomial Naive Bayes.
   - Model advanced Machine Learning: Support Vector Machine (SVM).
   - Model Deep Learning: Long Short-Term Memory (LSTM).
5. **[O]** Melakukan optimasi model secara sistematis melalui pengujian hyperparameter dan pengendalian overfitting.
6. **[P]** Menggunakan metrik evaluasi yang sesuai dengan jenis tugas klasifikasi, seperti Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
7. Menyusun laporan hasil eksperimen secara ilmiah dan sistematis.
8. Mengunggah seluruh kode dan dokumentasi proyek ke GitHub.
9. Menerapkan prinsip software engineering dalam pengembangan proyek agar kode lebih terstruktur dan dapat direproduksi.

---

# 2. PROJECT OVERVIEW [L]

## 2.1 Latar Belakang

Penyebaran berita hoaks melalui media sosial dan platform digital telah menjadi salah satu permasalahan serius dalam penyebaran informasi di Indonesia. Tingginya volume dan kecepatan aliran informasi digital menyebabkan masyarakat rentan menerima serta menyebarkan informasi yang salah sebelum sempat memverifikasi kebenarannya.

Pembuktian kebenaran berita secara manual membutuhkan waktu, tenaga, dan sumber daya yang cukup banyak. Proses pemeriksaan fakta yang dilakukan secara manual juga berpotensi terlambat dibandingkan dengan kecepatan penyebaran berita hoaks di media digital.

Pengembangan sistem otomatis berbasis Machine Learning dan Deep Learning dapat menjadi salah satu pendekatan untuk membantu proses penyaringan informasi. Dengan memanfaatkan teknik Natural Language Processing (NLP), teks berita dapat diolah sehingga model mampu mempelajari pola kata dan karakteristik teks yang berkaitan dengan kategori berita.

Pada proyek ini dikembangkan pendekatan klasifikasi biner untuk membedakan berita hoaks dan berita valid berbahasa Indonesia. Model menggunakan judul berita sebagai fitur utama dalam proses klasifikasi.

Sistem yang dikembangkan dirancang sebagai alat bantu penyaringan awal informasi bagi masyarakat, bukan sebagai pengambil keputusan mutlak mengenai kebenaran suatu berita. Hasil prediksi tetap memerlukan verifikasi lebih lanjut dari sumber informasi yang dapat dipercaya.

Implementasi proyek ini mendukung Sustainable Development Goals (SDG) 16, yaitu Perdamaian, Keadilan, dan Kelembagaan yang Tangguh. Secara khusus, proyek berkaitan dengan Target 16.10 melalui upaya membantu masyarakat memperoleh akses informasi yang lebih terpercaya.

**Referensi (berformat APA):**

> Jocelynne, C., Wijayakusuma, I. L., & Harini, L. P. I. (2025). Detection of Political Hoax News Using Fine-Tuning IndoBERT. Journal of Applied Informatics and Computing, 9(2), 354–360. https://doi.org/10.30871/jaic.v9i2.8989

> Rachmawati, O. C. R., & Darmawan, Z. M. E. (2024). The Comparison of Deep Learning Models for Indonesian Political Hoax News Detection. CommIT Journal, 18(2), 123–135. https://doi.org/10.21512/commit.v18i2.10929

## 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban |
|---|---|---|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | SDG 16: Peace, Justice, and Strong Institutions (Perdamaian, Keadilan, dan Kelembagaan yang Tangguh). |
| **Target SDG** | Target mana yang paling relevan? | Target 16.10: Menjamin akses publik terhadap informasi dan melindungi kebebasan fundamental sesuai dengan perundang-undangan nasional dan perjanjian internasional. |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Tingginya tingkat penyebaran berita hoaks di platform digital Indonesia yang dapat memicu disinformasi, konflik sosial, dan kesalahpahaman masyarakat. Pihak yang terdampak adalah masyarakat umum pengguna internet dan media sosial. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? | Machine Learning dan Deep Learning dapat membantu melakukan klasifikasi awal terhadap teks berita secara otomatis berdasarkan pola kata dan karakteristik teks. Sistem membantu mempercepat penyaringan informasi sebelum dilakukan verifikasi lebih mendalam. |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi terhadap target SDG? | Kontribusi tidak langsung melalui pengembangan alat bantu klasifikasi awal yang dapat membantu meningkatkan kesadaran masyarakat terhadap informasi yang diterima. |
| **Pertimbangan etika** | Apa risiko bias, privasi, atau dampak kesalahan prediksi? | Terdapat risiko false positive, yaitu berita valid dianggap hoaks, serta false negative, yaitu berita hoaks dianggap valid. Selain itu, dataset memiliki potensi bias berdasarkan sumber berita. Hasil prediksi harus diposisikan sebagai rekomendasi awal dan bukan keputusan final. |

**Sumber ide:**

Ide proyek ini muncul karena sering mendengar pengalaman anggota keluarga, khususnya nenek, yang pernah mempercayai informasi hoaks yang dibaca melalui telepon genggam. Pengalaman tersebut menjadi salah satu alasan untuk mengembangkan sistem yang dapat membantu masyarakat melakukan penyaringan awal terhadap informasi digital.

## 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini dijalankan melalui tahapan Machine Learning Lifecycle yang meliputi Learning Problem (L), Model Design (M), Optimization (O), dan Predict & Evaluate (P).

Pada tahap Learning Problem, permasalahan dirumuskan sebagai binary text classification dengan target prediksi berupa berita valid (0) dan berita hoaks (1). Dataset diperoleh dari Kaggle yang merupakan gabungan empat sumber berita berbahasa Indonesia, yaitu CNN Indonesia, Kompas, Tempo, dan TurnBackHoax.

Pada tahap data preparation, dilakukan pemeriksaan karakteristik dataset, Exploratory Data Analysis, penanganan missing values, penghapusan data duplikat, penghapusan penanda kategori pada judul, lowercasing, pemilihan fitur, serta pembagian data menggunakan stratified split.

Pada tahap Model Design, direncanakan tiga pendekatan model, yaitu Multinomial Naive Bayes sebagai baseline, Support Vector Machine sebagai advanced Machine Learning, dan Long Short-Term Memory sebagai Deep Learning.

Pada tahap Optimization, dilakukan perencanaan optimasi model melalui pengujian hyperparameter dan pengendalian overfitting. Implementasi optimasi yang telah dilakukan pada notebook adalah pengujian parameter C pada model LinearSVC.

Pada tahap Predict & Evaluate, model SVM telah diuji menggunakan test set sebanyak 4.693 data dan memperoleh accuracy sebesar 96,10%. Hasil evaluasi ini menjadi dasar untuk mengidentifikasi kesalahan klasifikasi dan merencanakan pengembangan model selanjutnya.

---

# 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING [L]

## 3.1 Problem Statements

Permasalahan yang diangkat dalam proyek ini dirumuskan sebagai berikut:

1. **Efisiensi Verifikasi Teks:** Bagaimana membangun model klasifikasi otomatis yang dapat membedakan berita hoaks dan berita valid berbahasa Indonesia berdasarkan judul berita secara cepat dan tepat sehingga dapat membantu mengurangi ketergantungan pada pemeriksaan manual?

2. **Kinerja Arsitektur Model:** Bagaimana perbandingan performa arsitektur Deep Learning (LSTM) dengan metode Machine Learning klasik (Multinomial Naive Bayes dan Support Vector Machine) dalam mengklasifikasikan berita hoaks berbahasa Indonesia?

3. **Penanganan Unstructured Text dan Imbalanced Data:** Bagaimana menerapkan teknik preprocessing dan ekstraksi fitur teks yang sesuai untuk menangani variasi kosakata, penanda kategori, dan ketidakseimbangan kelas pada dataset berita berbahasa Indonesia?

4. **Kemampuan Generalisasi Model:** Bagaimana kemampuan model dalam melakukan klasifikasi terhadap data yang belum pernah digunakan selama proses training, dengan mempertimbangkan adanya keterkaitan antara sumber dataset dan label klasifikasi?

## 3.2 Formulasi Learning Problem (WAJIB)

Permasalahan dirumuskan menggunakan kerangka Learning Problem sebagai berikut:

| Komponen | Pertanyaan Panduan | Jawaban |
|---|---|---|
| **Target** | Apa yang diprediksi? Jenis tugas apa yang digunakan? | Target prediksi berupa label kategori berita biner, yaitu Hoaks (1) atau Valid/Non-Hoaks (0). Jenis tugas: Supervised Learning - Binary Text Classification. |
| **Objective** | Apa yang dioptimalkan? Apa metrik keberhasilan utama? | Mengoptimalkan kemampuan model dalam membedakan berita valid dan hoaks. Metrik evaluasi utama adalah F1-Score, khususnya pada kelas hoaks, dengan tetap memperhatikan Precision, Recall, dan Accuracy. |
| **Data** | Apa sumber, ukuran, jenis, dan asumsi data? | Sumber data berasal dari Kaggle, yaitu Indonesian Fact and Hoax Political News. Jumlah data awal adalah 31.353 baris dan setelah cleaning menjadi 31.282 baris. Fitur pemodelan aktual adalah judul berita (`Title`) dengan target `hoax`. Dataset dibagi menjadi training, validation, dan testing menggunakan perbandingan 70:15:15. |

## 3.3 Goals

Tujuan proyek dirumuskan sebagai berikut:

1. Membangun model Machine Learning untuk memprediksi berita hoaks berdasarkan pola judul berita.
2. Mengembangkan tiga pendekatan model, yaitu baseline, advanced Machine Learning, dan Deep Learning.
3. Mencapai performa klasifikasi dengan target accuracy lebih dari 85%.
4. Mengukur performa ketiga pendekatan model menggunakan Accuracy, Precision, Recall, dan F1-Score.
5. Menentukan model terbaik berdasarkan hasil evaluasi pada test set.
6. Menghasilkan sistem klasifikasi yang dapat direproduksi.
7. Mendukung Target SDG 16.10 dengan menyediakan sistem penyaringan awal informasi hoaks untuk membantu meningkatkan kualitas informasi yang diterima masyarakat.

Berdasarkan eksperimen yang telah dilakukan, model SVM memperoleh accuracy sebesar 96,10% pada test set. Dengan demikian, target accuracy awal telah tercapai pada model yang sudah diuji. Namun, pencapaian perbandingan tiga model masih memerlukan implementasi model lainnya.

## 3.4 Solution Approach [M]

Proyek ini dirancang menggunakan tiga model dengan karakteristik yang berbeda.

### Model 1 - Baseline Model

**Nama Model:** Multinomial Naive Bayes.

**Teori Singkat:**

Multinomial Naive Bayes merupakan algoritma klasifikasi berbasis probabilitas yang menggunakan Teorema Bayes untuk menentukan kategori data berdasarkan distribusi fitur.

Dalam klasifikasi teks, algoritma ini dapat digunakan untuk mempelajari hubungan antara kemunculan kata dan kategori berita.

**Alasan Pemilihan:**

Multinomial Naive Bayes dipilih sebagai model baseline karena memiliki struktur yang sederhana, efisien, dan sesuai untuk klasifikasi teks.

Model ini direncanakan sebagai pembanding awal untuk mengetahui kemampuan klasifikasi menggunakan pendekatan probabilistik.

**Status Implementasi:** Belum diimplementasikan pada notebook yang tersedia.

### Model 2 - Advanced / ML Model

**Nama Model:** Support Vector Machine (SVM) menggunakan LinearSVC.

**Teori Singkat:**

Support Vector Machine merupakan algoritma supervised learning yang digunakan untuk menyelesaikan permasalahan klasifikasi dengan mencari hyperplane optimal yang memisahkan kelas data.

Pada proyek ini, SVM menggunakan representasi fitur teks hasil TF-IDF untuk mempelajari pola kata dan kombinasi kata yang berkaitan dengan kategori berita.

**Alasan Pemilihan:**

SVM dipilih karena memiliki kemampuan yang baik dalam mengolah data teks dengan jumlah fitur yang tinggi dan representasi sparse seperti TF-IDF.

Model LinearSVC juga sesuai untuk melakukan klasifikasi teks menggunakan batas keputusan linear.

**Status Implementasi:** Sudah diimplementasikan, dilatih, dilakukan hyperparameter tuning, dan dievaluasi menggunakan test set.

### Model 3 - Deep Learning Model (WAJIB)

**Nama Model:** Long Short-Term Memory (LSTM).

**Teori Singkat:**

LSTM merupakan salah satu arsitektur Recurrent Neural Network yang dirancang untuk memproses data berurutan.

Arsitektur ini memiliki mekanisme memory cell yang memungkinkan model mempertahankan informasi penting dari urutan kata.

**Alasan Pemilihan:**

LSTM dipilih karena sesuai untuk data teks yang memiliki karakteristik sekuensial. Model ini direncanakan untuk mempelajari hubungan antarkata dan pola urutan kata dalam judul berita.

**Status Implementasi:** Belum diimplementasikan pada notebook yang tersedia.

**Minimum Requirements untuk Deep Learning sesuai panduan proyek:**

- Training minimal 10 epochs.
- Memiliki plot training dan validation loss.
- Memiliki plot accuracy atau metrik per epoch.
- Memiliki hasil prediksi pada test set.
- Training time dicatat untuk dokumentasi.

---

# 4. DATA UNDERSTANDING [L]

## 4.1 Informasi Dataset

**Sumber Dataset:**

[Indonesian Fact and Hoax Political News - Kaggle](https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news)

Dataset yang digunakan merupakan gabungan empat kumpulan data berita berbahasa Indonesia, yaitu CNN Indonesia, Kompas, Tempo, dan TurnBackHoax.

File dataset yang digunakan berasal dari folder `Cleaned` di dalam arsip ZIP.

Penggabungan dilakukan menggunakan bahasa Python dengan library Pandas. Setiap dataset diberikan kolom tambahan bernama `source` untuk menunjukkan asal sumber berita. Setelah itu, seluruh dataset digabungkan menjadi satu DataFrame menggunakan fungsi `pd.concat()`.

### Distribusi Dataset Awal

| Sumber Dataset | Jumlah Baris | Label |
|---|---:|---|
| CNN Indonesia | 9.630 | Valid (0) |
| Kompas | 4.750 | Valid (0) |
| Tempo | 6.592 | Valid (0) |
| TurnBackHoax | 10.381 | Hoaks (1) |
| **Total** | **31.353** | |

### Karakteristik Dataset

| Komponen | Keterangan |
|---|---|
| Jumlah baris awal | 31.353 |
| Jumlah kolom pada hasil penggabungan | 13 kolom |
| Jumlah kolom file CNN, Kompas, dan Tempo | 9 kolom |
| Jumlah kolom file TurnBackHoax | 11 kolom |
| Tipe data | Teks dan metadata |
| Format file | XLSX |
| Bahasa | Indonesia |
| Jenis pembelajaran | Supervised Learning |
| Jenis tugas | Binary Text Classification |
| Target prediksi | `hoax` |
| Label 0 | Berita valid/non-hoaks |
| Label 1 | Berita hoaks |
| Total empat file XLSX dalam folder Cleaned | 33,56 MB |
| Jumlah data gabungan | 31.353 baris |
| Jumlah kolom setelah penggabungan | 13 kolom |

Dataset termasuk data teks tidak terstruktur (*unstructured text*) karena informasi utamanya berupa judul dan isi artikel. Selain itu, dataset memiliki beberapa kolom metadata seperti waktu publikasi, penulis, tag, URL, dan sumber berita.

> **Catatan sumber:** Label 0 terdapat pada dataset CNN Indonesia, Kompas, dan Tempo, sedangkan label 1 terdapat pada dataset TurnBackHoax. Oleh karena itu, label dalam dataset ini memiliki keterkaitan yang kuat dengan sumber asal data. Kondisi tersebut menjadi salah satu keterbatasan yang perlu diperhatikan ketika melakukan pemodelan dan menginterpretasikan hasil penelitian.

## 4.2 Deskripsi Fitur

Kolom yang tersedia tidak sepenuhnya sama pada keempat file dataset. CNN Indonesia, Kompas, dan Tempo memiliki sembilan kolom, sedangkan TurnBackHoax memiliki sebelas kolom.

Setelah seluruh dataset digabungkan, terdapat 13 kolom karena beberapa kolom hanya tersedia pada sumber tertentu dan terdapat tambahan kolom `source`.

Berikut deskripsi kolom berdasarkan hasil penggabungan dataset:

| Nama Fitur | Tipe Data Awal | Deskripsi |
|---|---|---|
| `Unnamed: 0` | Integer | Kolom indeks dari file asal; tidak digunakan sebagai fitur model. |
| `Title` | String/Object | Judul artikel berita. |
| `Timestamp` | String/Object | Waktu publikasi artikel. |
| `FullText` | String/Object | Isi artikel berita. |
| `Tags` | String/Object | Tag atau kategori artikel. |
| `Author` | String/Object | Nama penulis atau pihak yang tercantum sebagai penulis. |
| `Url` | String/Object | Alamat sumber artikel. |
| `text_new` | String/Object | Teks hasil pemrosesan yang tersedia pada sebagian sumber. |
| `hoax` | Integer | Label target klasifikasi, yaitu 0 atau 1. |
| `source` | String/Object | Sumber asal dataset yang ditambahkan saat proses penggabungan. |
| `politik` | Float | Penanda kategori politik pada dataset TurnBackHoax. |
| `Narasi` | String/Object | Narasi klaim pada dataset TurnBackHoax. |
| `Clean Narasi` | String/Object | Narasi yang telah dibersihkan. |

### Fitur yang Digunakan untuk Pemodelan

Pada tahap penyederhanaan awal, dibuat kolom sementara `text_clean` berdasarkan ketersediaan kolom `text_new` dan `FullText`.

Namun, pada proses pemodelan akhir dalam notebook, fitur yang digunakan adalah:

- `Title`: Judul berita sebagai fitur input.
- `hoax`: Label target klasifikasi.

Dengan demikian, model SVM menggunakan judul berita saja sebagai input, bukan gabungan judul dan isi artikel.

Kolom `text_clean` dibuat pada tahap awal pengolahan data, tetapi kemudian dihapus sebelum proses data splitting. Sementara itu, metadata seperti URL, penulis, waktu publikasi, tag, dan sumber dataset tidak digunakan sebagai fitur utama pemodelan.

## 4.3 Kondisi Data

### 4.3.1 Missing Values

Berdasarkan hasil pemeriksaan awal setelah empat dataset digabungkan, ditemukan beberapa nilai kosong pada sejumlah kolom.

| Nama Kolom | Jumlah Missing Values |
|---|---:|
| `Unnamed: 0` | 0 |
| `Title` | 21 |
| `Timestamp` | 21 |
| `FullText` | 27 |
| `Tags` | 162 |
| `Author` | 337 |
| `Url` | 0 |
| `text_new` | 10.408 |
| `hoax` | 0 |
| `source` | 0 |
| `politik` | 20.972 |
| `Narasi` | 20.972 |
| `Clean Narasi` | 24.851 |

Nilai kosong pada kolom `Title` perlu ditangani karena kolom tersebut merupakan fitur utama yang digunakan dalam proses klasifikasi.

Pada tahap penyederhanaan dataset, kolom `text_clean` juga memiliki 27 nilai kosong. Setelah dilakukan penghapusan baris berdasarkan kolom `Title`, masih terdapat enam nilai kosong pada `text_clean`.

Namun, kolom `text_clean` tidak digunakan dalam pemodelan akhir sehingga enam nilai kosong tersebut tidak memengaruhi fitur input SVM.

Penanganan missing values difokuskan pada kolom yang benar-benar diperlukan untuk pemodelan, yaitu `Title` dan `hoax`.

### 4.3.2 Duplicate Data

Pemeriksaan duplikasi dilakukan setelah proses penghapusan penanda kategori dan lowercasing pada judul berita.

Pemeriksaan dilakukan berdasarkan kesamaan nilai pada kolom `Title`.

Hasil pemeriksaan menunjukkan:

- Jumlah kelompok judul duplikat: 49.
- Jumlah baris duplikat tambahan: 50.
- Jumlah judul dengan konflik label: 0.

Sebelum penghapusan, dilakukan pemeriksaan konflik label untuk memastikan judul yang sama tidak memiliki label berbeda.

Karena tidak ditemukan konflik label, data duplikat dihapus menggunakan fungsi `drop_duplicates()` dengan parameter `subset=['Title']` dan `keep='first'`.

Hasilnya, jumlah data berkurang dari 31.332 menjadi 31.282 baris.

### 4.3.3 Outliers

Outlier numerik tidak menjadi fokus utama karena fitur akhir yang digunakan dalam penelitian berupa teks.

Namun, dilakukan pemeriksaan terhadap panjang judul berita berdasarkan jumlah kata pada kolom `Title`.

Pemeriksaan ditampilkan menggunakan visualisasi Boxplot untuk mengetahui variasi panjang judul pada masing-masing kategori berita.

Hasil visualisasi menunjukkan adanya sejumlah judul yang memiliki jumlah kata berbeda cukup jauh dibandingkan sebagian besar data lainnya.

Data tersebut tetap dipertahankan karena panjang judul yang berbeda tidak secara otomatis menunjukkan bahwa data tersebut salah atau tidak valid.

Pada notebook yang digunakan, tidak dilakukan perhitungan IQR maupun penghapusan data berdasarkan outlier.

### 4.3.4 Imbalanced Data

Berdasarkan distribusi label awal, jumlah data berita valid lebih banyak dibandingkan data berita hoaks.

Distribusi label sebelum proses cleaning adalah sebagai berikut:

| Kelas | Jumlah | Persentase |
|---|---:|---:|
| Valid (0) | 20.972 | 66,89% |
| Hoaks (1) | 10.381 | 33,11% |
| **Total** | **31.353** | **100%** |

Setelah proses cleaning, distribusi label menjadi:

| Kelas | Jumlah | Persentase |
|---|---:|---:|
| Valid (0) | 20.940 | 66,94% |
| Hoaks (1) | 10.342 | 33,06% |
| **Total** | **31.282** | **100%** |

Distribusi tersebut menunjukkan bahwa dataset memiliki ketidakseimbangan kelas dengan perbandingan sekitar 2:1.

Pada implementasi SVM yang tersedia, tidak dilakukan oversampling maupun undersampling. Distribusi kelas asli tetap dipertahankan dalam proses pelatihan.

Metrik Precision, Recall, dan F1-Score digunakan untuk mengetahui kemampuan model dalam memprediksi setiap kategori, terutama kelas hoaks yang memiliki jumlah data lebih sedikit.

### 4.3.5 Noise dan Kualitas Data

Teks berita yang berasal dari beberapa sumber memiliki kemungkinan perbedaan format penulisan, tanda baca, karakter khusus, dan penggunaan spasi yang tidak konsisten.

Selain itu, terdapat perbedaan gaya penulisan antara artikel berita dari portal berita dan artikel pemeriksaan fakta dari TurnBackHoax.

Pada pemeriksaan judul TurnBackHoax ditemukan beberapa penanda kategori seperti:

- `[SALAH]`
- `[BENAR]`
- `[HOAX]`
- `[KLARIFIKASI]`
- `[DISINFORMASI]`
- `[BERITA]`
- `[FALSE]`
- `[EDUKASI]`

Hasil pemeriksaan menunjukkan bahwa terdapat 425 judul dengan penanda `[BENAR]`, meskipun seluruh data TurnBackHoax diberikan label 1 atau hoaks.

Kondisi tersebut menunjukkan adanya kemungkinan ketidaksesuaian antara penanda kategori dengan label yang digunakan.

Oleh karena itu, penanda kategori di awal judul dihapus menggunakan Regular Expression (Regex) sebelum digunakan dalam proses pemodelan.

Meskipun demikian, penghapusan penanda kategori belum sepenuhnya menyelesaikan permasalahan kualitas label sehingga pemeriksaan lebih lanjut tetap diperlukan.

### 4.3.6 Asumsi Data dan Potensi Data Leakage

Dataset digunakan untuk kebutuhan *supervised learning*. Model mempelajari hubungan antara teks judul berita sebagai input dan label `hoax` sebagai target klasifikasi.

Namun, terdapat beberapa risiko yang perlu diperhatikan.

Pertama, seluruh data valid berasal dari CNN Indonesia, Kompas, dan Tempo, sedangkan seluruh data hoaks berasal dari TurnBackHoax. Kondisi ini menyebabkan label memiliki hubungan yang sangat kuat dengan sumber asal berita.

Akibatnya, model berpotensi mempelajari karakteristik gaya penulisan atau format judul dari masing-masing sumber, bukan benar-benar mengenali karakteristik berita hoaks dan berita valid.

Kedua, terdapat penanda kategori pada judul TurnBackHoax seperti `[SALAH]`, `[BENAR]`, dan `[HOAX]`. Penanda tersebut berpotensi memberikan petunjuk langsung mengenai label klasifikasi.

Untuk mengurangi risiko tersebut, pada tahap feature engineering dilakukan penghapusan penanda kategori di awal judul menggunakan Regex.

Selain itu, kolom metadata seperti URL, penulis, waktu publikasi, tag, dan sumber dataset tidak digunakan sebagai fitur utama model.

Pembagian data menggunakan metode stratified split dengan `random_state=42` untuk menjaga proporsi kelas valid dan hoaks pada setiap bagian data.

Meskipun demikian, stratified split secara acak belum sepenuhnya mengatasi bias berdasarkan sumber dataset karena data dari sumber yang sama tetap dapat muncul pada training, validation, dan test.

Evaluasi lanjutan menggunakan data dari sumber yang lebih beragam dan pemisahan berdasarkan sumber berita akan lebih kuat untuk mengukur kemampuan generalisasi model.

## 4.4 Exploratory Data Analysis (EDA) - OPSIONAL

Exploratory Data Analysis (EDA) merupakan proses pemeriksaan dan pemahaman karakteristik dataset sebelum dilakukan pemodelan.

Pada notebook yang dijalankan, telah dibuat tiga visualisasi utama untuk memahami distribusi label, panjang judul, serta keterkaitan sumber dataset dengan label klasifikasi.

### Visualisasi 1: Distribusi Kelas Berita Valid dan Hoaks

Visualisasi pertama menggunakan grafik batang untuk memperlihatkan jumlah berita berdasarkan label.

Hasil pemeriksaan sebelum cleaning:

| Kelas | Jumlah |
|---|---:|
| Valid (0) | 20.972 |
| Hoaks (1) | 10.381 |

**Insight berdasarkan hasil aktual:**

Berdasarkan visualisasi distribusi kelas, jumlah berita valid sebanyak 20.972 data atau 66,89%, sedangkan berita hoaks sebanyak 10.381 data atau 33,11%.

Hal ini menunjukkan bahwa dataset memiliki ketidakseimbangan kelas (*imbalanced data*), dengan jumlah berita valid sebagai kelas mayoritas.

Kondisi tersebut perlu diperhatikan dalam proses pelatihan dan evaluasi model agar performa klasifikasi pada kelas hoaks tetap dapat diukur dengan baik.

### Visualisasi 2: Distribusi Jumlah Kata dalam Judul Berita

Visualisasi kedua menggunakan Boxplot untuk memperlihatkan distribusi jumlah kata pada kolom `Title` berdasarkan kategori berita valid dan hoaks.

Pemeriksaan dilakukan dengan menghitung jumlah kata dari setiap judul, kemudian memisahkan data berdasarkan label 0 dan 1.

**Insight berdasarkan hasil aktual:**

Berdasarkan visualisasi Boxplot, kedua kategori berita memiliki variasi panjang judul yang berbeda.

Terdapat sejumlah judul yang memiliki jumlah kata lebih tinggi dibandingkan sebagian besar data lainnya. Kondisi tersebut menunjukkan adanya variasi panjang teks pada masing-masing kategori.

Meskipun demikian, panjang judul tidak dapat dijadikan indikator langsung untuk menentukan kebenaran suatu berita.

### Visualisasi 3: Distribusi Kelas Berdasarkan Sumber Dataset

Visualisasi ketiga menggunakan *stacked bar chart* untuk menunjukkan distribusi kelas berita pada setiap sumber dataset.

**Insight berdasarkan hasil aktual:**

Berdasarkan visualisasi distribusi kelas berdasarkan sumber dataset, CNN Indonesia, Kompas, dan Tempo seluruhnya memiliki label valid, sedangkan TurnBackHoax seluruhnya memiliki label hoaks.

Hal ini menunjukkan bahwa terdapat keterkaitan yang sangat kuat antara sumber dataset dan label klasifikasi.

Kondisi tersebut berpotensi menimbulkan bias karena model dapat mempelajari karakteristik penulisan atau format judul dari sumber berita, bukan benar-benar mengenali karakteristik berita hoaks dan berita valid.

> **Catatan:** Ketiga visualisasi tersebut telah dibuat pada notebook Google Colab. Grafik distribusi kelas menggunakan data sebelum cleaning, sedangkan visualisasi panjang judul menggunakan kolom `Title`, bukan isi artikel `FullText`.

---

# 5. DATA PREPARATION

Bagian ini menjelaskan tahapan transformasi dan preprocessing yang dilakukan terhadap dataset sebelum digunakan dalam proses pemodelan.

## 5.1 Data Cleaning

Data cleaning dilakukan untuk memastikan data yang digunakan memiliki struktur dan kualitas yang sesuai untuk proses klasifikasi.

Tahapan yang dilakukan meliputi:

### 1. Handling Missing Values

Pada pemeriksaan awal ditemukan 21 nilai kosong pada kolom `Title`.

Strategi yang digunakan adalah menghapus baris yang memiliki nilai kosong pada kolom `Title` menggunakan fungsi `dropna()`.

Implementasi:

```python
data = data.dropna(subset=['Title'])
```

Penghapusan dipilih karena judul merupakan informasi utama yang digunakan sebagai input dalam proses klasifikasi.

Hasilnya, sebanyak 21 baris data dihapus sehingga jumlah data berkurang dari 31.353 menjadi 31.332 baris.

Nilai kosong pada kolom `text_clean` masih tersisa sebanyak enam baris setelah proses tersebut. Namun, kolom `text_clean` tidak digunakan dalam pemodelan akhir sehingga tidak dilakukan penghapusan tambahan berdasarkan kolom tersebut.

### 2. Removing Duplicates

Penghapusan data duplikat dilakukan berdasarkan kesamaan judul berita.

Sebelum dibandingkan, teks pada kolom `Title` diubah menjadi huruf kecil dan penanda kategori pada awal judul dihapus.

Langkah ini dilakukan agar judul dengan perbedaan penanda kategori dan huruf kapital dapat diperiksa secara lebih konsisten.

Pemeriksaan dilakukan dengan mengelompokkan judul dan menghitung jumlah kemunculannya.

Hasilnya ditemukan 49 kelompok judul duplikat dengan jumlah 50 baris duplikat tambahan.

Sebelum penghapusan, dilakukan pemeriksaan konflik label untuk memastikan judul yang sama tidak memiliki label berbeda. Hasilnya menunjukkan tidak ditemukan konflik label.

Penghapusan dilakukan menggunakan fungsi:

```python
data = data.drop_duplicates(
    subset=['Title'],
    keep='first'
).copy()
```

Hasilnya, jumlah data berkurang dari 31.332 menjadi 31.282 baris.

### 3. Handling Outliers

Pemeriksaan karakteristik panjang judul dilakukan melalui visualisasi Boxplot berdasarkan jumlah kata pada kolom `Title`.

Namun, tidak dilakukan penghapusan data outlier karena panjang judul yang terlalu pendek atau panjang belum tentu menunjukkan bahwa data tersebut salah atau tidak valid.

Dengan demikian, seluruh data tetap dipertahankan.

### 4. Data Type Conversion

Konversi tipe data dilakukan untuk memastikan kolom `Timestamp` dapat diproses menjadi format tanggal.

Pada proses ini, format tanggal dari sumber yang berbeda diseragamkan menggunakan bantuan pemetaan nama bulan Bahasa Indonesia ke Bahasa Inggris, kemudian dikonversi menggunakan fungsi `pd.to_datetime()`.

Hasil konversi:

| Keterangan | Jumlah |
|---|---:|
| Total data | 31.282 |
| Timestamp berhasil dikonversi | 31.282 |
| Timestamp gagal dikonversi | 0 |

Kolom `Timestamp` selanjutnya dihapus karena tidak digunakan sebagai fitur utama dalam pemodelan.

### Hasil Akhir Data Cleaning

| Keterangan | Jumlah |
|---|---:|
| Data awal | 31.353 |
| Data dihapus karena Title kosong | 21 |
| Data setelah penanganan missing values | 31.332 |
| Data duplikat tambahan yang dihapus | 50 |
| **Data setelah cleaning** | **31.282** |

Distribusi label setelah proses cleaning:

| Kelas | Jumlah | Persentase |
|---|---:|---:|
| Valid (0) | 20.940 | 66,94% |
| Hoaks (1) | 10.342 | 33,06% |
| **Total** | **31.282** | **100%** |

## 5.2 Feature Engineering

Feature engineering dilakukan untuk memilih fitur yang relevan dengan tujuan klasifikasi dan menghapus fitur yang tidak diperlukan.

Tahapan yang dilakukan adalah sebagai berikut:

### 1. Menghapus Penanda Kategori pada Judul

Menghapus penanda kategori yang terdapat pada awal judul berita, seperti `[SALAH]`, `[BENAR]`, `[HOAX]`, dan penanda lainnya.

Penghapusan dilakukan menggunakan Regular Expression (Regex) agar model tidak terlalu bergantung pada penanda kategori yang berpotensi memberikan petunjuk langsung terhadap label.

### 2. Menghapus Fitur yang Tidak Digunakan

Setelah proses pemeriksaan dan transformasi awal, fitur yang digunakan dalam pemodelan akhir adalah:

- `Title` sebagai variabel input (X).
- `hoax` sebagai variabel target (y).

Kolom `Timestamp` dihapus karena distribusi data berdasarkan waktu tidak merata dan dikhawatirkan memengaruhi model dalam menentukan kategori berita berdasarkan waktu publikasi.

Kolom `FullText` dan `Clean Narasi` juga tidak digunakan sebagai input akhir. Hal ini karena `FullText` berisi isi artikel, sedangkan `Clean Narasi` pada dataset TurnBackHoax berisi narasi atau komentar terkait klaim yang beredar.

Pada eksperimen ini, pemodelan difokuskan pada judul berita agar fitur input yang digunakan lebih seragam.

### 3. Pemisahan Fitur dan Target

Setelah fitur dipilih, dilakukan pemisahan variabel input dan target menggunakan:

```python
X = data['Title']
y = data['hoax']
```

Dengan demikian, model menggunakan judul berita untuk memprediksi label klasifikasi.

## 5.3 Data Transformation

Data transformation dilakukan untuk mengubah teks berita menjadi format yang dapat diproses oleh algoritma Machine Learning.

Karena penelitian menggunakan data teks, transformasi dilakukan berdasarkan kebutuhan pemodelan.

### 1. Text Cleaning

Text cleaning dilakukan dengan menghapus penanda kategori yang terdapat pada awal judul menggunakan Regular Expression.

Implementasi:

```python
import re

def hapus_tags(title):
    cleaned = re.sub(
        r'^\s*[\[\(][^\]\)]+[\]\)]\s*',
        '',
        str(title)
    )
    return cleaned.strip()

data['Title'] = data['Title'].apply(hapus_tags)
```

### 2. Text Lowercasing

Seluruh teks pada kolom `Title` diubah menjadi huruf kecil menggunakan fungsi `str.lower()`.

Implementasi:

```python
data['Title'] = data['Title'].str.lower()
```

Tahapan ini dilakukan untuk menyeragamkan format penulisan sehingga perbedaan huruf kapital tidak menjadi pembeda dalam pengolahan teks.

### 3. TF-IDF Vectorization

Pada pemodelan SVM, teks judul diubah menjadi representasi numerik menggunakan metode Term Frequency-Inverse Document Frequency (TF-IDF).

Konfigurasi yang digunakan:

| Parameter | Nilai |
|---|---|
| `max_features` | 20.000 |
| `ngram_range` | (1, 2) |
| `sublinear_tf` | True |

Implementasi:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# TF-IDF dipelajari hanya dari data training
x_train_tfidf = tfidf.fit_transform(X_train)

# Validation dan testing menggunakan vectorizer yang sama
x_val_tfidf = tfidf.transform(X_val)
x_test_tfidf = tfidf.transform(X_test)
```

TF-IDF hanya dipelajari dari data training. Data validation dan testing menggunakan hasil transformasi dari proses training agar tidak terjadi kebocoran informasi dari data evaluasi ke proses pembentukan fitur.

## 5.4 Data Splitting

Data splitting dilakukan untuk membagi dataset menjadi tiga bagian, yaitu data training, validation, dan testing.

Strategi yang digunakan adalah pembagian data dengan perbandingan 70:15:15.

### Hasil Pembagian Data

| Jenis Data | Persentase | Jumlah Sampel |
|---|---:|---:|
| Training | 70% | 21.897 |
| Validation | 15% | 4.692 |
| Test | 15% | 4.693 |
| **Total** | **100%** | **31.282** |

Pembagian data dilakukan menggunakan fungsi `train_test_split()` dari library Scikit-learn melalui dua tahap pembagian.

Tahap pertama memisahkan 70% data training dan 30% data sementara. Selanjutnya, data sementara dibagi kembali menjadi data validation dan testing dengan perbandingan yang sama.

Metode stratify digunakan untuk mempertahankan proporsi kelas valid dan hoaks pada masing-masing bagian data. Selain itu, digunakan `random_state=42` agar proses pembagian menghasilkan hasil yang konsisten ketika kode dijalankan kembali.

Implementasi:

```python
from sklearn.model_selection import train_test_split

# Memisahkan fitur dan target
X = data['Title']
y = data['hoax']

# Pembagian data training dan data sementara
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Membagi data sementara menjadi validation dan testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
```

Distribusi kelas pada setiap bagian data:

| Dataset | Valid (0) | Hoaks (1) | Total |
|---|---:|---:|---:|
| Training | 14.658 | 7.239 | 21.897 |
| Validation | 3.141 | 1.551 | 4.692 |
| Testing | 3.141 | 1.552 | 4.693 |
| **Total** | **20.940** | **10.342** | **31.282** |

Data training digunakan untuk mempelajari pola klasifikasi. Data validation digunakan untuk memantau performa dan memilih konfigurasi model, sedangkan data testing digunakan untuk evaluasi akhir model.

## 5.5 Data Balancing

Pada implementasi notebook yang tersedia, data balancing tidak dilakukan menggunakan teknik oversampling maupun undersampling.

Distribusi asli dataset tetap dipertahankan dengan jumlah data valid sebanyak 20.940 dan data hoaks sebanyak 10.342.

Keputusan tersebut diterapkan pada eksperimen SVM yang tersedia. Evaluasi model tetap memperhatikan Precision, Recall, dan F1-Score untuk melihat kemampuan klasifikasi pada masing-masing kelas.

Class Weight belum diterapkan pada proses pelatihan SVM dalam notebook.

## 5.6 Ringkasan Data Preparation

Tahapan data preparation dilakukan untuk memastikan dataset memiliki kualitas, struktur, dan format yang sesuai sebelum digunakan dalam proses pemodelan.

Berikut ringkasan tahapan data preparation yang dilakukan:

| Tahapan | Apa yang dilakukan? | Mengapa dilakukan? | Bagaimana implementasinya? |
|---|---|---|---|
| **Data Cleaning** | Menangani missing values pada `Title`, menghapus judul duplikat, dan mengonversi `Timestamp` menjadi format tanggal. | Memastikan data memiliki informasi utama yang dibutuhkan dan mengurangi redundansi data. | Menggunakan `dropna()`, `drop_duplicates()`, dan `pd.to_datetime()`. |
| **Feature Engineering** | Memilih `Title` sebagai fitur input dan `hoax` sebagai target klasifikasi serta menghapus fitur yang tidak digunakan. | Memusatkan proses klasifikasi pada judul berita dan menghindari penggunaan metadata yang tidak diperlukan. | Menggunakan pemilihan kolom pada DataFrame Pandas. |
| **Data Transformation** | Menghapus penanda kategori di awal judul, mengubah teks menjadi huruf kecil, dan mengubah teks menjadi representasi numerik untuk SVM. | Mengurangi petunjuk label yang terdapat pada penanda kategori dan membuat teks dapat diproses oleh algoritma. | Menggunakan Regex, `str.lower()`, dan `TfidfVectorizer()`. |
| **Data Splitting** | Membagi dataset menjadi training, validation, dan testing dengan perbandingan 70:15:15. | Memisahkan data untuk pembelajaran, pemantauan performa, dan evaluasi akhir model. | Menggunakan `train_test_split()` sebanyak dua tahap dengan `stratify` dan `random_state=42`. |
| **Data Balancing** | Tidak melakukan oversampling maupun undersampling. | Mempertahankan distribusi kelas asli pada dataset yang digunakan dalam eksperimen. | Menggunakan data asli tanpa menerapkan teknik penyeimbangan kelas. |

### Kesimpulan Data Preparation

Berdasarkan seluruh tahapan yang telah dilakukan, dataset berhasil dipersiapkan dari 31.353 data awal menjadi 31.282 data setelah proses penanganan missing values dan penghapusan duplikat.

Fitur yang digunakan untuk pemodelan akhir adalah judul berita (`Title`) sebagai input dan kolom `hoax` sebagai target klasifikasi.

Proses transformasi teks dilakukan melalui penghapusan penanda kategori pada awal judul dan lowercasing. Pada model SVM, teks kemudian ditransformasikan menggunakan TF-IDF dengan konfigurasi `max_features=20000`, `ngram_range=(1, 2)`, dan `sublinear_tf=True`.

Dataset selanjutnya dibagi menjadi training, validation, dan testing menggunakan perbandingan 70:15:15 dengan metode stratified split dan `random_state=42`.

Dengan demikian, tahapan Data Preparation telah diselesaikan dan dataset telah siap digunakan dalam proses pemodelan sesuai kebutuhan masing-masing algoritma.

## 6. MODELING [M + O]

Tahap modeling dilakukan untuk membangun model klasifikasi berita hoaks berbahasa Indonesia menggunakan pendekatan Machine Learning dan Deep Learning. Pada tahap ini, model pertama yang digunakan adalah **Multinomial Naive Bayes** sebagai baseline model. Model ini digunakan sebagai pembanding awal untuk mengetahui kemampuan klasifikasi berdasarkan fitur teks yang telah direpresentasikan menggunakan TF-IDF.

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model [M]

**Nama Model:** Multinomial Naive Bayes (MNB)

**Teori Singkat:**

Multinomial Naive Bayes merupakan algoritma klasifikasi probabilistik yang menggunakan Teorema Bayes untuk menentukan kelas suatu data berdasarkan probabilitas kemunculan fitur terhadap masing-masing kelas.

Model ini bekerja dengan menghitung probabilitas suatu dokumen termasuk ke dalam kategori tertentu berdasarkan fitur kata yang terdapat di dalam dokumen tersebut. Pada proyek ini, fitur teks berita direpresentasikan menggunakan metode TF-IDF dengan 20.000 fitur dan kombinasi unigram serta bigram.

Multinomial Naive Bayes menggunakan asumsi bahwa setiap fitur bersifat independen terhadap fitur lainnya dengan mempertimbangkan kelas yang diprediksi. Untuk menghindari probabilitas bernilai nol ketika terdapat fitur yang belum pernah muncul pada suatu kelas, digunakan teknik smoothing melalui parameter alpha.

**Alasan Pemilihan:**

Multinomial Naive Bayes dipilih sebagai baseline karena memiliki proses training yang relatif cepat, sederhana, dan sesuai untuk permasalahan klasifikasi teks. Model ini dapat digunakan sebagai pembanding awal sebelum menerapkan algoritma yang lebih kompleks, seperti Support Vector Machine dan Long Short-Term Memory.

Selain itu, model ini mendukung penggunaan fitur TF-IDF sehingga dapat dimanfaatkan untuk mengklasifikasikan judul berita ke dalam dua kategori, yaitu berita valid dan berita hoaks.

#### 6.1.2 Hypothesis Space & Inductive Bias [M]

- **Model Family / Arsitektur:**

  Probabilistic Generative Model dengan pendekatan Naive Bayes untuk klasifikasi teks multivariat berbasis distribusi multinomial.

- **Hypothesis Space:**

  Himpunan distribusi probabilitas yang digunakan untuk memprediksi kelas berita berdasarkan probabilitas prior kelas dan probabilitas fitur terhadap kelas tersebut. Model memilih kelas dengan probabilitas posterior paling tinggi.

  Secara matematis, prediksi dilakukan dengan:

  \[
  \hat{y} = \arg\max_c P(c)\prod_j P(x_j|c)^{x_j}
  \]

  Keterangan:
  - \(P(c)\): probabilitas suatu kelas.
  - \(P(x_j|c)\): probabilitas fitur ke-j terhadap kelas tertentu.
  - \(x_j\): bobot fitur dari representasi TF-IDF.

- **Inductive Bias / Asumsi:**

  1. Setiap fitur kata dianggap independen secara kondisional terhadap fitur lainnya apabila kelas telah diketahui.
  2. Dokumen direpresentasikan sebagai kumpulan fitur kata tanpa mempertimbangkan urutan kata secara langsung.
  3. Probabilitas fitur terhadap kelas dapat digunakan untuk menentukan kategori berita.
  4. Penggunaan Laplace smoothing membantu menghindari probabilitas nol pada fitur tertentu.

- **Implikasi terhadap Dataset:**

  Asumsi Multinomial Naive Bayes sesuai untuk eksperimen awal pada dataset berita berbahasa Indonesia karena data telah diubah menjadi representasi numerik menggunakan TF-IDF.

  Meskipun hubungan antar kata dalam sebuah kalimat sebenarnya dapat saling bergantung, pendekatan Naive Bayes tetap dapat digunakan sebagai baseline yang sederhana dan efisien. Penggunaan unigram dan bigram pada TF-IDF juga membantu merepresentasikan kata tunggal maupun kombinasi dua kata.

#### 6.1.3 Implementasi (Ringkas) [M]

Implementasi Multinomial Naive Bayes dilakukan menggunakan library Scikit-learn. Dataset yang digunakan merupakan hasil preprocessing dari notebook sebelumnya, sehingga proses pembagian data dan transformasi TF-IDF tetap menggunakan konfigurasi yang sama dengan eksperimen SVM.

**Konfigurasi TF-IDF:**
- `max_features = 20000`
- `ngram_range = (1, 2)`
- `sublinear_tf = True`

**Implementasi model:**

```python
from sklearn.naive_bayes import MultinomialNB

# Membuat model baseline
nb_model = MultinomialNB(
    alpha=1.0,
    fit_prior=True
)

# Training model
nb_model.fit(
    x_train_tfidf,
    y_train
)

# Prediksi data validation
y_val_pred_nb = nb_model.predict(
    x_val_tfidf
)
```

Pembagian dataset yang digunakan:

| Jenis Data | Jumlah Data |
|---|---:|
| Training | 21.897 |
| Validation | 4.692 |
| Testing | 4.693 |
| **Total** | **31.282** |

Matriks fitur TF-IDF memiliki 20.000 fitur dengan rincian:
- Training: 21.897 × 20.000
- Validation: 4.692 × 20.000
- Testing: 4.693 × 20.000

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi [O]

**Objective / Loss yang Dioptimalkan:**

Multinomial Naive Bayes melakukan estimasi probabilitas kelas dan probabilitas fitur berdasarkan data training. Prediksi ditentukan berdasarkan probabilitas posterior terbesar.

Dalam proses estimasi probabilitas fitur, digunakan teknik additive smoothing untuk menghindari probabilitas nol. Teknik ini dikendalikan melalui parameter alpha.

Berbeda dengan model yang menggunakan gradient descent, Multinomial Naive Bayes tidak melakukan optimasi iteratif menggunakan optimizer seperti Adam atau SGD. Parameter probabilitas model dihitung berdasarkan distribusi fitur dan kelas pada data training.

**Parameter yang Digunakan:**

```python
nb_model = MultinomialNB(
    alpha=1.0,
    fit_prior=True
)
```

| Parameter | Nilai | Keterangan |
|---|---|---|
| `alpha` | 1.0 | Menggunakan additive/Laplace smoothing |
| `fit_prior` | True | Probabilitas prior kelas dihitung dari data training |
| `class_prior` | None | Prior kelas tidak ditentukan secara manual |

**Proses Training:**

Model dilatih menggunakan data training yang telah ditransformasikan ke dalam bentuk TF-IDF. Berdasarkan hasil eksekusi notebook, proses training menghasilkan:
- Jumlah fitur: 20.000
- Jumlah kelas: 2, yaitu 0 dan 1
- Training time: 0,0337 detik

Pada tahap ini, parameter model masih menggunakan konfigurasi baseline dengan `alpha=1.0`. Belum dilakukan pencarian hyperparameter secara sistematis sehingga konfigurasi ini digunakan sebagai hasil awal untuk perbandingan model.

#### 6.1.5 Hasil Awal [O → P]

Evaluasi awal Multinomial Naive Bayes dilakukan menggunakan validation set yang berjumlah 4.692 data. Evaluasi ini bertujuan mengetahui kemampuan awal model dalam membedakan berita valid dan berita hoaks sebelum dilakukan evaluasi akhir menggunakan test set.

**Hasil Accuracy:**

- Accuracy: 0,9397
- Accuracy dalam persentase: **93,97%**
- Training Time: **0,0337 detik**

**Classification Report:**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Non-Hoaks | 0,9304 | 0,9834 | 0,9562 | 3.141 |
| Hoaks | 0,9621 | 0,8511 | 0,9032 | 1.551 |
| **Macro Average** | **0,9463** | **0,9173** | **0,9297** | **4.692** |
| **Weighted Average** | **0,9409** | **0,9397** | **0,9387** | **4.692** |

**Confusion Matrix:**

Hasil confusion matrix pada validation set menunjukkan:

| Keterangan | Prediksi Non-Hoaks | Prediksi Hoaks |
|---|---:|---:|
| Aktual Non-Hoaks | 3.089 | 52 |
| Aktual Hoaks | 231 | 1.320 |

Berdasarkan confusion matrix tersebut, model berhasil mengklasifikasikan 3.089 berita valid dan 1.320 berita hoaks dengan benar. Sementara itu, terdapat 52 berita valid yang salah diklasifikasikan sebagai hoaks dan 231 berita hoaks yang salah diklasifikasikan sebagai berita valid.

**Analisis Hasil Awal:**

Berdasarkan hasil evaluasi pada validation set, model Multinomial Naive Bayes memperoleh accuracy sebesar **93,97%**. Pada kelas hoaks, model menghasilkan precision sebesar **96,21%**, yang menunjukkan bahwa sebagian besar berita yang diprediksi sebagai hoaks memang termasuk dalam kategori hoaks. Sementara itu, nilai recall sebesar **85,11%** menunjukkan bahwa model berhasil mengidentifikasi sebagian besar berita hoaks, meskipun masih terdapat sejumlah data yang belum terdeteksi dengan benar.

Berdasarkan confusion matrix, terdapat **1.320 berita hoaks** yang berhasil diklasifikasikan dengan benar, sedangkan **231 berita hoaks** salah diklasifikasikan sebagai berita valid. Selain itu, terdapat 52 berita valid yang salah diprediksi sebagai hoaks. Hasil tersebut menunjukkan bahwa model memiliki ketepatan prediksi hoaks yang tinggi, tetapi masih perlu ditingkatkan dalam mengenali seluruh berita yang sebenarnya termasuk kategori hoaks.

Jika dibandingkan dengan model Linear Support Vector Machine (SVM), Multinomial Naive Bayes memiliki nilai precision kelas hoaks yang lebih tinggi, yaitu 96,21% dibandingkan 94,95%. Namun, Linear SVM memperoleh hasil yang lebih baik pada accuracy, recall, dan F1-Score. Linear SVM mencatat accuracy sebesar 95,76%, recall sebesar 92,07%, dan F1-Score sebesar 93,49%, sedangkan Multinomial Naive Bayes memperoleh masing-masing 93,97%, 85,11%, dan 90,32%.

Dari segi waktu pelatihan, Multinomial Naive Bayes membutuhkan waktu sekitar **0,0337 detik**, sehingga menunjukkan proses training yang relatif cepat. Secara keseluruhan, Multinomial Naive Bayes mampu memberikan performa awal yang baik dan memenuhi fungsinya sebagai baseline model. Meskipun demikian, berdasarkan hasil evaluasi validation set, Linear SVM menunjukkan kemampuan klasifikasi yang lebih baik secara keseluruhan, terutama dalam mengenali berita hoaks. Hasil perbandingan ini selanjutnya dapat digunakan sebagai dasar untuk mengevaluasi model lanjutan pada tahap berikutnya.



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
