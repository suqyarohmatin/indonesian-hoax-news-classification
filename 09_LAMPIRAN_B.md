## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ 1. Rayyan Afif / 244311025 
                 2. Suqya Rohmatin / 244311028 ]|
| Judul ide | [ Klasifikasi Berita Hoaks Berbahasa Indonesia Menggunakan Machine Learning dan Deep Learning ] |
| SDG & target yang dipilih | [ SDG 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh, khususnya Target 16.10, yaitu menjamin akses publik terhadap informasi serta melindungi kebebasan fundamental. Proyek ini mendukung akses informasi yang lebih baik melalui pengembangan model untuk mengidentifikasi indikasi berita hoaks.] |
| Masalah nyata yang diangkat | [ Penyebaran berita hoaks berbahasa Indonesia melalui internet dan media sosial dapat menyebabkan masyarakat menerima informasi yang salah. Sedangkan pemeriksaan berita secara manual membutuhkan waktu yang lama. Maka dari itu, dibutuhkan pendekatan berbasis Machine Learning dan Deep Learning yang dapat mengklasifikasikan berita berdasarkan teks nya ke dalam kategori hoaks dan valid. Hasil model hanya digunakan untuk alat bantu awal dan bukan sebagai pengganti proses verifikasi fakta. ] |
| Tugas ML (Target & jenis tugas) | [ Target : Label berita, yaitu hoaks dan valid/non-hoaks.
Jenis tugas :  Supervised Learning, khususnya klasifikasi teks biner (binary text classification). Input berupa teks berita atau klaim, sedangkan output berupa prediksi kategori berita.] |
| Objective & metrik utama | [ Tujuan proyek adalah membangun dan membandingkan model Machine Learning dan Deep Learning untuk mengklasifikasikan berita hoaks berbahasa Indonesia. Metrik utama yang digunakan adalah F1-Score, karena mempertimbangkan Precision dan Recall. Metrik tambahan meliputi Accuracy, Precision, Recall, dan Confusion Matrix. Model diharapkan dapat mengidentifikasi berita hoaks dengan baik sekaligus mengurangi kesalahan klasifikasi. ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ Sumber: Mendeley Data. URL: https://data.mendeley.com/datasets/mzx5rd397v/1. Ukuran: 18.310 data, terdiri dari 15.187 data training, 1.561 data validation, dan 1.562 data testing. Jenis data: Data teks berupa berita atau artikel berbahasa Indonesia yang telah melalui proses preprocessing, dengan label klasifikasi hoaks dan non-hoaks. ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ 1. Baseline: Multinomial Naive Bayes dengan fitur TF-IDF, sebagai model awal yang sederhana dan cepat. 2. Advanced: Support Vector Machine (SVM) dengan TF-IDF, untuk membandingkan performa Machine Learning yang lebih kuat dalam klasifikasi teks. 3. Deep Learning: Long Short-Term Memory (LSTM), untuk mempelajari pola dan urutan kata dalam teks berita. Ketiga model akan menggunakan pembagian data dan metrik evaluasi yang sama agar perbandingan lebih adil. ] |
| Risiko & pertimbangan etika (bias, privasi) | [ Dataset dapat memiliki ketidakseimbangan jumlah berita hoaks dan valid, sehingga berpotensi menimbulkan bias. Label dataset mungkin tidak selalu benar atau mutakhir. Model dapat salah mengklasifikasikan berita valid sebagai hoaks maupun sebaliknya. Model juga berpotensi mempelajari pola sumber berita, bukan isi berita. Oleh karena itu, data akan diperiksa, duplikasi dihindari, dan evaluasi dilakukan pada data yang tidak digunakan saat training. Dataset digunakan sesuai lisensi dan ketentuan sumber. Hasil prediksi tidak dianggap sebagai keputusan final tentang kebenaran berita. ] |
| Referensi awal (minimal 2) | [ 1. Jocelynne, C., Wijayakusuma, I. L., & Harini, L. P. I. (2025). Detection of Political Hoax News Using Fine-Tuning IndoBERT. Journal of Applied Informatics and Computing, 9(2), 354–360. https://doi.org/10.30871/jaic.v9i2.8989 

 2. Rachmawati, O. C. R., & Darmawan, Z. M. E. (2024). The Comparison of Deep Learning Models for Indonesian Political Hoax News Detection. CommIT Journal, 18(2), 123–135. https://doi.org/10.21512/commit.v18i2.10929 ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu