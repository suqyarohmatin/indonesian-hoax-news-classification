
# 📌 Ide Penelitian Machine Learning

## 👥 Identitas Mahasiswa

| No. | Nama | NIM |
|:---:|---|:---:|
| 1 | Rayyan Afif | 244311025 |
| 2 | Suqya Rohmatin | 244311028 |

---

## 1. Judul Ide

**Klasifikasi Berita Hoaks Berbahasa Indonesia Menggunakan Machine Learning dan Deep Learning**

## 2. SDG dan Target yang Dipilih

**SDG 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh.**

Target 16.10, yaitu menjamin akses publik terhadap informasi serta melindungi kebebasan fundamental.

Proyek ini mendukung akses informasi yang lebih baik melalui pengembangan model untuk mengidentifikasi indikasi berita hoaks.

## 3. Masalah Nyata yang Diangkat

Penyebaran berita hoaks berbahasa Indonesia melalui internet dan media sosial dapat menyebabkan masyarakat menerima informasi yang salah.

Pemeriksaan berita secara manual membutuhkan waktu yang lama. Oleh karena itu, dibutuhkan pendekatan berbasis Machine Learning dan Deep Learning untuk mengklasifikasikan berita berdasarkan teks ke dalam kategori hoaks dan valid.

Hasil model hanya digunakan sebagai alat bantu awal, bukan pengganti proses verifikasi fakta.

## 4. Tugas Machine Learning

| Komponen | Keterangan |
|---|---|
| Target | Label berita: hoaks dan valid/non-hoaks |
| Jenis Tugas | Supervised Learning |
| Kategori | Binary Text Classification |
| Input | Teks berita atau klaim |
| Output | Prediksi kategori berita |

## 5. Metode Penelitian

Penelitian ini menggunakan **metode eksperimen komparatif (Comparative Experiment)** dengan pendekatan **Supervised Learning** untuk klasifikasi teks (*Text Classification*).

Metode ini digunakan untuk melakukan eksperimen dan membandingkan performa tiga algoritma dalam mengklasifikasikan berita berbahasa Indonesia ke dalam kategori hoaks dan non-hoaks.

### Tahapan Penelitian

1. **Pengumpulan Data:** Menggunakan dataset berita hoaks berbahasa Indonesia dari Mendeley Data.
2. **Preprocessing Data:** Melakukan pembersihan dan persiapan teks sebelum digunakan dalam pemodelan.
3. **Pemodelan:** Menerapkan tiga algoritma, yaitu:
   - Multinomial Naive Bayes dengan TF-IDF.
   - Support Vector Machine (SVM) dengan TF-IDF.
   - Long Short-Term Memory (LSTM).
4. **Evaluasi Model:** Mengukur performa setiap model menggunakan F1-Score, Accuracy, Precision, Recall, dan Confusion Matrix.
5. **Perbandingan Model:** Membandingkan hasil evaluasi untuk mengetahui model dengan performa terbaik.

Hasil penelitian diharapkan dapat menunjukkan perbedaan performa ketiga algoritma dalam mengklasifikasikan berita hoaks berbahasa Indonesia.

## 6. Objective dan Metrik Evaluasi

**Tujuan:** Membangun dan membandingkan model Machine Learning dan Deep Learning untuk mengklasifikasikan berita hoaks berbahasa Indonesia.

Metrik evaluasi:

- **F1-Score:** Metrik utama untuk menyeimbangkan Precision dan Recall.
- **Accuracy:** Mengukur persentase prediksi yang benar.
- **Precision:** Mengukur ketepatan prediksi berita hoaks.
- **Recall:** Mengukur kemampuan model menemukan berita hoaks.
- **Confusion Matrix:** Menampilkan jumlah prediksi benar dan salah.

## 7. Dataset

| Komponen | Keterangan |
|---|---|
| Sumber | Mendeley Data |
| URL | [Dataset Berita Hoaks Indonesia](https://data.mendeley.com/datasets/mzx5rd397v/1) |
| Total Data | 18.310 |
| Training | 15.187 |
| Validation | 1.561 |
| Testing | 1.562 |
| Jenis Data | Teks berita atau artikel berbahasa Indonesia |
| Label | Hoaks dan non-hoaks |

Dataset merupakan kumpulan teks berita berbahasa Indonesia yang telah melalui proses preprocessing dan memiliki label klasifikasi.

## 8. Rencana Tiga Model

### 1. Baseline: Multinomial Naive Bayes + TF-IDF

Model dasar yang sederhana dan cepat digunakan sebagai pembanding awal dalam klasifikasi teks.

### 2. Advanced: Support Vector Machine (SVM) + TF-IDF

Model Machine Learning untuk memisahkan kategori berita berdasarkan pola fitur teks.

### 3. Deep Learning: Long Short-Term Memory (LSTM)

Model Deep Learning yang digunakan untuk mempelajari pola dan urutan kata dalam teks berita.

Ketiga model menggunakan pembagian data dan metrik evaluasi yang sama agar perbandingan lebih adil.

## 9. Risiko dan Pertimbangan Etika

Beberapa risiko yang perlu diperhatikan:

1. **Ketidakseimbangan Data:** Jumlah berita hoaks dan valid mungkin tidak seimbang.
2. **Ketepatan Label:** Label dataset mungkin tidak selalu benar atau mutakhir.
3. **Kesalahan Klasifikasi:** Model dapat salah mengategorikan berita valid sebagai hoaks maupun sebaliknya.
4. **Bias Sumber Berita:** Model berpotensi mempelajari pola sumber berita, bukan isi berita.

Upaya mitigasi:

- Memeriksa kualitas dataset.
- Menghindari duplikasi data.
- Melakukan evaluasi menggunakan data yang tidak digunakan saat training.
- Menggunakan dataset sesuai lisensi dan ketentuan sumber.
- Tidak menjadikan hasil prediksi sebagai keputusan final kebenaran berita.

## 10. Referensi Awal

1. Jocelynne, C., Wijayakusuma, I. L., & Harini, L. P. I. (2025). Detection of Political Hoax News Using Fine-Tuning IndoBERT. *Journal of Applied Informatics and Computing, 9*(2), 354–360. [https://doi.org/10.30871/jaic.v9i2.8989](https://doi.org/10.30871/jaic.v9i2.8989)

2. Rachmawati, O. C. R., & Darmawan, Z. M. E. (2024). The Comparison of Deep Learning Models for Indonesian Political Hoax News Detection. *CommIT Journal, 18*(2), 123–135. [https://doi.org/10.21512/commit.v18i2.10929](https://doi.org/10.21512/commit.v18i2.10929)
