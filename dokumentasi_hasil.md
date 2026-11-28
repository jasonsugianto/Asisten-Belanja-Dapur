# Dokumentasi Hasil Uji Empat Model

Dibuat otomatis 2026-11-28 15:20. Semua model dievaluasi pada **80 rekaman uji yang sama** (pembicara dan kalimat yang belum dilihat model yang dilatih).

## 1. Ringkasan metrik

| Model | Item F1 | Presisi | Recall | Intent | Kalimat benar | Latensi p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| M0 Whisper tiny (+prompt) | 0.855 | 0.934 | 0.789 | 73.8% | 45.0% | 356 / 388 |
| M1 Zipformer streaming tanpa fine-tune (int8) | 0.808 | 0.955 | 0.700 | 72.5% | 37.5% | 30 / 35 |
| M1 Zipformer streaming fine-tune (int8) | 0.966 | 0.994 | 0.939 | 98.8% | 86.2% | 31 / 38 |
| M2 wav2vec2 XLS-R Indonesia (fine-tune) | 0.986 | 0.994 | 0.978 | 98.8% | 93.8% | 1761 / 3113 |
| M3 BC-ResNet (kosakata tertutup) | 0.241 | 0.194 | 0.317 | 47.5% | 0.0% | 9 / 12 |

Catatan latensi: ditulis apa adanya dari berkas hasil (M2 diukur di CPU Colab; model lain di laptop). M1 dan M3 streaming: dihitung sejak potongan audio terakhir diterima; M0 dan M2: waktu memproses seluruh kalimat. Lihat bagian keterbatasan.

Catatan M1: kedua baris memakai model Zipformer fonem yang sama dan NLU fonem bawaan. Baris fine-tune dilatih dengan pembagian `splits.json` (5 putaran, sama dengan M2/M3), sehingga pembicara dan kalimat uji tidak dilihat saat pelatihan.

## 2. Akurasi per jenis perintah (kalimat benar)

| Model | tambah | centang | hapus |
|---|---|---|---|
| M0 Whisper tiny (+prompt) | 22/40 | 13/30 | 1/10 |
| M1 Zipformer streaming tanpa fine-tune (int8) | 15/40 | 15/30 | 0/10 |
| M1 Zipformer streaming fine-tune (int8) | 34/40 | 26/30 | 9/10 |
| M2 wav2vec2 XLS-R Indonesia (fine-tune) | 37/40 | 29/30 | 9/10 |
| M3 BC-ResNet (kosakata tertutup) | 0/40 | 0/30 | 0/10 |

## 3. Barang yang paling sering terlewat

- **M0 Whisper tiny (+prompt)**: telur (5), tisu (3), wortel (3), sawi (2), sirup (2), tepung terigu (2)
- **M1 Zipformer streaming tanpa fine-tune (int8)**: wortel (4), minyak goreng (3), jus jeruk (3), teh celup (3), apel (3), air mineral (3)
- **M1 Zipformer streaming fine-tune (int8)**: tahu (1), jahe (1), cabai (1), apel (1), air mineral (1), mie instan (1)
- **M2 wav2vec2 XLS-R Indonesia (fine-tune)**: tepung beras (1), mie instan (1), teh celup (1), brokoli (1)
- **M3 BC-ResNet (kosakata tertutup)**: cabai (5), susu (5), bayam (5), bawang putih (4), telur (4), jeruk (3)

## 4. Contoh keluaran per model

Setiap contoh menampilkan **keluaran mentah model**, lalu **hasil setelah pencocokan** (intent dan barang). Tanda ✓ berarti intent dan seluruh barang benar.

### spk01 kalimat 2 (tambah)

- **Naskah:** Beli telur sama ikan.
- **Jawaban benar:** intent `tambah`, barang: ikan, telur

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `belitelur sama ikan` | tambah | ikan | ✗ |
| M1z | `l i \| t ə l u r \| s a m a \| i k a n` | tambah | ikan, telur | ✓ |
| M1 | `b ə l i \| t ə l u r \| s a m a \| i k a n` | tambah | ikan, telur | ✓ |
| M2 | `beli telur sama ikan` | tambah | ikan, telur | ✓ |
| M3 | `jahe(0.67) santan(0.54) telur(0.53) telur puyuh(0.37)` | tambah | jahe, santan, telur, telur puyuh | ✗ |

### spk04 kalimat 23 (tambah)

- **Naskah:** Apa ya, tambahin tepung terigu sama mentega.
- **Jawaban benar:** intent `tambah`, barang: mentega, tepung terigu

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `apaya tambahin, tepung, teregu, sama, mentega,` | tambah | mentega, tepung terigu | ✓ |
| M1z | `a p a \| j a \| t a m b a h i n \| t ə p u ŋ \| t ə r i ɡ u \| s a m a \| m ə n t e ɡ a` | tambah | mentega, tepung terigu | ✓ |
| M1 | `a p a \| i j a \| t a m b a h i n \| t ə p u ŋ \| t ə r i ɡ u \| s a m a \| m ə n t e ɡ a` | tambah | mentega, tepung terigu | ✓ |
| M2 | `apa ya tambahin tepung terigu sama mentega` | tambah | mentega, tepung terigu | ✓ |
| M3 | `ikan(0.46) centang(0.96)` | centang | ikan | ✗ |

### spk02 kalimat 22 (tambah)

- **Naskah:** Oh iya, beli sabun cuci piring sama saus sambal.
- **Jawaban benar:** intent `tambah`, barang: sabun cuci piring, saus sambal

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `oh ya, beli sabun cuci piring, sama saus sambal` | tambah | sabun cuci piring, saus sambal | ✓ |
| M1z | `o h \| i j a \| b ə l i \| s a b u n \| tʃ u tʃ i \| p i r i ŋ \| s a m a \| s a u s \| s a m b a l` | tambah | sabun cuci piring, saus sambal | ✓ |
| M1 | `o h \| i j a \| b ə l i \| s a b u n \| tʃ u tʃ i \| p i r i ŋ \| s a m a \| s a ʔ u s \| s a m b a l` | tambah | sabun cuci piring, saus sambal | ✓ |
| M2 | `oyabeli sabun cuci piring sama saus sambal` | tambah | sabun cuci piring, saus sambal | ✓ |
| M3 | `tepung terigu(0.69)` | tambah | tepung terigu | ✗ |

### spk03 kalimat 25 (tambah)

- **Naskah:** Stok habis, perlu tisu, air mineral, dan kopi.
- **Jawaban benar:** intent `tambah`, barang: air mineral, kopi, tisu

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `stock habits, perluti su, air mineral dan kopi` | tambah | air mineral, kopi | ✗ |
| M1z | `s t o tʃ ʔ \| h a b i s \| p ə r l u t i s u \| a i r \| m ə n ə r a l \| d a n \| k o p i` | tambah | kopi, tisu | ✗ |
| M1 | `s t o k \| h a b i s \| b ə l u \| t i s u \| k a i r \| m ə r a l \| d a n \| k o p i` | tambah | kopi, tisu | ✗ |
| M2 | `stok habis perlu tisu air meneral dan kopi` | tambah | air mineral, kopi, tisu | ✓ |
| M3 | `jus jeruk(0.82) susu(0.79)` | tambah | jus jeruk, susu | ✗ |

### spk05 kalimat 66 (centang)

- **Naskah:** Kangkung dua ikat udah dapat.
- **Jawaban benar:** intent `centang`, barang: kangkung

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `kangkung, dua ikan, udandapat,` | tambah | ikan, kangkung | ✗ |
| M1z | `k a m p u ŋ \| d u a \| i k a t \| m u d a h \| d a p a t` | none | - | ✗ |
| M1 | `k a ŋ k u ŋ \| d u ʔ a \| i k a t \| u d a h \| d a p a t` | centang | kangkung | ✓ |
| M2 | `kangkung dua ikat udah dapat` | centang | kangkung | ✓ |
| M3 | `semangka(0.42)` | tambah | semangka | ✗ |

### spk04 kalimat 60 (centang)

- **Naskah:** Kecap manis sudah diambil.
- **Jawaban benar:** intent `centang`, barang: kecap manis

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `kecap manis, sudah diambil` | centang | kecap manis | ✓ |
| M1z | `tʃ a p \| m a n i \| s u d a h \| d i a m b i l` | none | - | ✗ |
| M1 | `e tʃ a p \| m a n i s \| s u d a h \| d i ʔ a m b i l` | centang | kecap manis | ✓ |
| M2 | `kecap manis sudah diambil` | centang | kecap manis | ✓ |
| M3 | `centang(0.87)` | centang | - | ✗ |

### spk01 kalimat 67 (centang)

- **Naskah:** Sudah ambil bawang bombay, jahe, dan tepung terigu.
- **Jawaban benar:** intent `centang`, barang: bawang bombay, jahe, tepung terigu

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `sedang belakang bomb bay, jahe, dan tepung kriku` | tambah | bawang bombay, jahe | ✗ |
| M1z | `s u d a h \| h i l \| b o m b o b a j \| dʒ a h e \| d a n \| t ə p u ŋ \| k ə r i ɡ u` | centang | bawang bombay, jahe, tepung terigu | ✓ |
| M1 | `s u d a h \| a m b i l \| b a w a ŋ \| p o b a j a h e \| d a n \| t ə p u ŋ \| t ə r i ɡ u` | centang | bawang bombay, tepung terigu | ✗ |
| M2 | `sudah ambil bawang bobai jahe dan tepung teriku` | centang | bawang bombay, jahe, tepung terigu | ✓ |
| M3 | `ayam(0.34) bayam(0.73) kangkung(0.54) kol(0.70) mie instan(0.71) sawi(0.72) tahu(0.61) timun(0.91)` | tambah | ayam, bayam, kangkung, kol, mie instan, sawi, tahu, timun | ✗ |

### spk02 kalimat 77 (hapus)

- **Naskah:** Batal air mineral sama jus jeruk.
- **Jawaban benar:** intent `hapus`, barang: air mineral, jus jeruk

| Model | Keluaran mentah | Intent | Barang terbaca | |
|---|---|---|---|---|
| M0 | `batal air mineral sama jujjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj` | hapus | air mineral | ✗ |
| M1z | `b a t a l \| a i r \| m i n ə r a l \| s a m a \| dʒ u s \| dʒ ə r u ʔ` | hapus | jeruk | ✗ |
| M1 | `b a t a l \| a ʔ i r \| m i n e r a l \| s a m a \| dʒ u s \| dʒ ə r u k` | hapus | air mineral, jus jeruk | ✓ |
| M2 | `batal air mineral sama jus jeruk` | hapus | air mineral, jus jeruk | ✓ |
| M3 | `air mineral(0.76) centang(0.42)` | centang | air mineral | ✗ |

Jenis keluaran mentah: M0 = teks Whisper; M1z = urutan fonem; M1 = urutan fonem; M2 = teks CTC; M3 = kata kunci terdeteksi (probabilitas).

## 5. Kasus ringkas

- Kalimat yang benar di **semua** model: 0 dari 80
- Kalimat yang salah di **semua** model: 3 dari 80
- Kalimat yang hanya benar di M2: 6

## 6. Keterbatasan yang perlu disebut

- Set uji hanya 80 rekaman; selisih kecil antar model tidak pasti.
- Naskah (pola kalimat dan 50 barang) sama untuk data latih dan uji; hasil berlaku untuk domain tertutup ini.
- Latensi tidak diukur pada satu mesin yang sama untuk semua model; latensi M2 di Colab jauh di atas laptop.
- Latensi belum termasuk waktu NLU (kecuali M3) dan deteksi akhir ucapan (VAD/endpoint) yang dibutuhkan aplikasi nyata.
- Pencocokan teks/fonem ke barang dirancang setelah melihat kesalahan pada data yang sama (M0 dan M1 sedikit optimistis).
- Belum diuji: noise, perangkat target (HP), barang di luar 50 item.
