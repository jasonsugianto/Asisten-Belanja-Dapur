# Panduan Fine-tune M1 (Zipformer streaming fonem)

Tujuan: menaikkan akurasi M1 dengan data Anda, sambil mempertahankan latensi sekitar 30 ms.
Pembagian data memakai `splits.json` yang sama dengan M2/M3. Uji = pembicara baru dan kalimat baru, sehingga hasilnya bisa dibandingkan langsung.

```
LAPTOP                         COLAB (GPU)                                   LAPTOP
siapkan_colab.py  ──zip──>  unduh_m1_pytorch.py                    ┌──>  run_m1_zipformer.py ×5
                            latih_m1.py --cek                      │     gabung_hasil.py
                            latih_m1.py --uji-cepat                │     evaluasi.py --uji80
                            latih_m1.py            (5 putaran)     │     buat_dokumentasi.py
                            ekspor_m1.py  ── models/m1_ft/ ────────┘
```

Pelatihan dijalankan di Colab. Evaluasi dan pengukuran latensi dijalankan di **laptop**, supaya angkanya setara dengan M1 lama (laptop, 2 thread).

---

## Langkah 1. Laptop: siapkan paket Colab

```powershell
python cek_fonem.py          # harus: "Fonem g2p yang TIDAK ada di model: tidak ada (cocok)"
python siapkan_colab.py      # -> belanja_colab.zip (kode + processed/ + tokens.txt)
```
Unggah `belanja_colab.zip` ke Google Drive.

## Langkah 2. Colab: instalasi

Runtime → Change runtime type → **GPU (T4 cukup)**. Jalankan setiap blok di sel terpisah.

```python
!nvidia-smi
import torch; print(torch.__version__, torch.version.cuda)
```

**k2** harus cocok dengan versi torch dan CUDA yang dicetak di atas. Buka https://k2-fsa.github.io/k2/cuda.html, cari wheel dengan `cuda<versi>.torch<versi>` yang sama, lalu pasang:
```python
!pip install k2==<VERSI_DARI_HALAMAN_ITU> -f https://k2-fsa.github.io/k2/cuda.html
```
Kalau tidak ada wheel untuk versi torch Colab, pasang versi torch yang punya wheel k2 (`pip install torch==<versi> torchaudio==<versi>`), lalu restart runtime.

```python
!git clone https://github.com/k2-fsa/icefall /content/icefall
!pip install -r /content/icefall/requirements.txt
!pip install lhotse g2p_id_py onnx onnxruntime huggingface_hub soundfile
import os
os.environ["PYTHONPATH"] = "/content/icefall:" + os.environ.get("PYTHONPATH", "")
RESEP = "/content/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
```

```python
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p /content/belanja && cd /content/belanja && unzip -q -o /content/drive/MyDrive/belanja_colab.zip
%cd /content/belanja
```

## Langkah 3. Colab: unduh bobot PyTorch M1

File ONNX di `models/zipformer_id` hanya bisa dipakai untuk inferensi. Fine-tune membutuhkan checkpoint PyTorch dari repo `bookbot/pruned-transducer-stateless7-streaming-id`.
```python
!python unduh_m1_pytorch.py
```
Skrip ini mencetak semua isi repo dan path berkas yang diunduh (misalnya `models/m1_pytorch/exp/pretrained.pt`). Pakai path itu untuk `--pretrained` di langkah berikutnya. Kalau skrip berhenti karena tidak menemukan `.pt`, kirim daftar berkasnya.

## Langkah 4. Colab: cek kecocokan (wajib, sekitar 1 menit)

```python
!python latih_m1.py --resep $RESEP --pretrained <PATH_DARI_LANGKAH_3> --cek
```
Hasil yang diharapkan:
- `Arsitektur cocok. ~70 juta parameter, vocab 33`
- `Loss val pretrained ... < 1` (kira-kira). Kalau jauh lebih besar, fitur atau token tidak cocok. Hentikan dan kirim keluarannya.

Kalau muncul **"Ukuran bobot tidak cocok"**, berarti bookbot melatih dengan argumen arsitektur yang berbeda dari bawaan resep. Kirim pesan error-nya. Ukuran tensor di pesan itu menunjukkan nilai yang harus diisi di `--model-args`.

## Langkah 5. Colab: uji cepat (sekitar 2–3 menit)

```python
!python latih_m1.py --resep $RESEP --pretrained <PATH> --uji-cepat
```
Harus selesai dengan `UJI CEPAT SELESAI tanpa error`. Hasilnya di `exp_m1_uji/`, jangan dipakai untuk laporan.

## Langkah 6. Colab: latih 5 putaran

```python
!python latih_m1.py --resep $RESEP --pretrained <PATH>
```
- Setiap putaran: 144 rekaman latih, 16 val. Rekaman uji tidak disentuh.
- Epoch terbaik dipilih dari **loss val**, lalu disimpan ke `exp_m1/fold{i}/epoch-{n}.pt` (sekitar 280 MB per putaran).
- Perhatikan baris `val X -> Y`. Y harus lebih kecil dari X. Kalau muncul `PERINGATAN: fine-tune tidak menurunkan loss val`, ulangi putaran itu dengan `--lr 0.002 --folds <i>`.
- Kalau sesi Colab terputus, lanjutkan putaran yang belum selesai dengan `--folds 3 4`. Putaran yang sudah ada di `exp_m1/` tidak perlu diulang.

Sebaiknya salin `exp_m1/` ke Drive setelah setiap putaran:
```python
!cp -r exp_m1 /content/drive/MyDrive/
```

## Langkah 7. Colab: ekspor ke ONNX

```python
!python ekspor_m1.py --resep $RESEP
!cd models && zip -r /content/drive/MyDrive/m1_ft.zip m1_ft -i "*.int8.onnx" "*tokens.txt" "*info.json"
```
Hasilnya: `models/m1_ft/fold0..fold4/`, masing-masing berisi encoder/decoder/joiner `.int8.onnx` dan `tokens.txt`. Hanya file int8 yang dimasukkan ke zip supaya ukurannya kecil.

## Langkah 8. Laptop: evaluasi dan latensi

Ekstrak `m1_ft.zip` ke `E:\speech\belanja_asr\models\`, sehingga ada `models\m1_ft\fold0\...`.

Setiap putaran hanya diuji pada pembicara ujinya (dari `splits.json`):

| Putaran | Pembicara uji |
|---|---|
| fold0 | spk01 |
| fold1 | spk02 |
| fold2 | spk03 |
| fold3 | spk04 |
| fold4 | spk05 |

```powershell
python run_m1_zipformer.py --model-dir models/m1_ft/fold0 --int8 --speakers spk01 --nama M1_finetune_int8 --out hasil/m1_ft_f0.csv
python run_m1_zipformer.py --model-dir models/m1_ft/fold1 --int8 --speakers spk02 --nama M1_finetune_int8 --out hasil/m1_ft_f1.csv
python run_m1_zipformer.py --model-dir models/m1_ft/fold2 --int8 --speakers spk03 --nama M1_finetune_int8 --out hasil/m1_ft_f2.csv
python run_m1_zipformer.py --model-dir models/m1_ft/fold3 --int8 --speakers spk04 --nama M1_finetune_int8 --out hasil/m1_ft_f3.csv
python run_m1_zipformer.py --model-dir models/m1_ft/fold4 --int8 --speakers spk05 --nama M1_finetune_int8 --out hasil/m1_ft_f4.csv
python gabung_hasil.py hasil/m1_ft_f0.csv hasil/m1_ft_f1.csv hasil/m1_ft_f2.csv hasil/m1_ft_f3.csv hasil/m1_ft_f4.csv -o hasil/m1_finetune_full.csv
```
Setiap perintah menjalankan 80 kalimat milik pembicara itu. Yang dilaporkan hanya 16 kalimat uji, karena `--uji80` dan `buat_dokumentasi.py` menyaring ke 80 rekaman uji.

## Langkah 9. Laptop: bandingkan dan buat dokumentasi

```powershell
python evaluasi.py hasil/m1_int8.csv hasil/m1_finetune_full.csv --uji80
python diagnosa_hasil.py hasil/m1_finetune_full.csv
python buat_dokumentasi.py --m1 hasil/m1_finetune_full.csv
```
Untuk pembanding, pakai `hasil/m1_int8.csv`, **bukan** `m1_int8_kw30.csv`. Hasil `kw30` disetel memakai 48 dari 80 rekaman uji, sedangkan M1 hasil fine-tune memakai pengaturan NLU bawaan.

---

## Pengaturan yang bisa diubah

| Argumen | Bawaan | Kapan diubah |
|---|---|---|
| `--lr` | 0.004 | Loss val naik atau tidak turun: 0.002. Turun sangat lambat: 0.008. |
| `--epochs` | 30 | Epoch terbaik selalu di akhir: tambah. Selalu di awal: kurangi. |
| `--batch` | 8 | Kehabisan memori GPU: 4. |
| `--model-args` | kosong | Hanya kalau langkah 4 gagal karena ukuran bobot. |

## Yang belum bisa saya verifikasi dari sandbox

Dari sandbox, saya tidak bisa mengakses HuggingFace atau GitHub, dan tidak ada GPU, jadi skrip ini belum pernah dijalankan. Langkah 4 dan 5 dibuat untuk memastikan hal-hal berikut sebelum pelatihan panjang:
1. Repo bookbot menyediakan checkpoint PyTorch. `unduh_m1_pytorch.py` mencari `pretrained.pt`, lalu `epoch-*.pt`, lalu `cpu_jit.pt`.
2. Model bookbot memakai argumen arsitektur bawaan resep `pruned_transducer_stateless7_streaming`.
3. `export-onnx.py` di icefall versi terbaru memakai `--tokens`. `ekspor_m1.py` mengecek ini dan berhenti dengan pesan jelas kalau tidak.
