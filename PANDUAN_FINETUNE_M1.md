# Panduan Fine-tune M1 (Zipformer streaming fonem)

Tujuan: menaikkan akurasi M1 dengan data Anda, sambil mempertahankan latensi sekitar 30 ms.
Pembagian data memakai `splits.json` yang sama dengan M2/M3. Uji = pembicara baru dan kalimat baru, sehingga hasilnya bisa dibandingkan langsung.

```
COLAB (GPU, tanpa Google Drive)                                  LAPTOP
git clone repo  ->  unduh_m1_pytorch.py                    ┌──>  run_m1_zipformer.py ×5
                    latih_m1.py --cek                      │     gabung_hasil.py
                    latih_m1.py --uji-cepat                │     evaluasi.py --uji80
                    latih_m1.py            (5 putaran)     │     buat_dokumentasi.py
                    ekspor_m1.py  ->  m1_ft.zip  (unduh) ──┘
```

Pelatihan dijalankan di Colab. Evaluasi dan pengukuran latensi dijalankan di **laptop**, supaya angkanya setara dengan M1 lama (laptop, 2 thread).

Semua file Colab disimpan di disk Colab (`/content`), bukan di Google Drive. Kalau sesi terputus, isi `/content` hilang dan pelatihan harus diulang. Biarkan tab Colab terbuka; lima putaran butuh sekitar 1–1,5 jam di T4.

Hasil yang sudah dicapai dengan panduan ini (80 rekaman uji, laptop): item F1 0,966, intent 98,8%, kalimat benar 86,2%, latensi p50/p95 31/38 ms.

---

## Colab

Runtime → Change runtime type → **T4 GPU**. Jalankan setiap sel berurutan.

### Sel 1. Cek versi
```python
!nvidia-smi
!python --version
import torch; print("torch", torch.__version__, "| CUDA", torch.version.cuda)
```
Sel 2–3 ditulis untuk Python 3.13 + CUDA 13.0 (kondisi Colab saat panduan ini dibuat). Kalau berbeda, sesuaikan versi torch dan wheel k2.

### Sel 2. torch 2.14.0 tanpa torchaudio
k2 dikompilasi untuk satu versi torch tertentu, dan wheel k2 terbaru tersedia untuk torch 2.14.0 (bukan 2.14.1). torchaudio dilepas karena versi 2.14 tidak ada dan tidak dipakai.
```python
!pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
!pip uninstall -y torchaudio
```
Lalu **Runtime → Restart session** (bukan "Disconnect and delete runtime").

### Sel 3. k2
Pilih wheel dari https://k2-fsa.github.io/k2/cuda.html yang namanya cocok dengan CUDA, torch, dan Python (`cuda13.0.torch2.14.0-cp313`).
```python
!pip install "k2==1.24.4.dev20261125+cuda13.0.torch2.14.0" -f https://k2-fsa.github.io/k2/cuda.html
import sys, torch, k2
print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available())
```
Harus: `3.13.x 2.14.0+cu130 13.0 True`. Kalau pip ikut mengganti torch, hentikan dan periksa versinya.

### Sel 4. icefall, paket lain, dan repo
```python
!git clone https://github.com/k2-fsa/icefall /content/icefall
!pip install -r /content/icefall/requirements.txt
!pip install lhotse g2p_id_py onnx onnxruntime huggingface_hub soundfile
!pip show torch | grep Version
!git clone --branch feat/m2-wav2vec2-bilstm-ctc --single-branch https://github.com/jasonsugianto/Asisten-Belanja-Dapur.git /content/belanja
import os
os.environ["PYTHONPATH"] = "/content/icefall:" + os.environ.get("PYTHONPATH", "")
RESEP = "/content/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
EXP = "/content/belanja/exp_m1"
%cd /content/belanja
!python cek_processed.py
!python cek_fonem.py
```
Harus: torch tetap `2.14.0+cu130`, `total 400 file ... 0 format salah`, dan `tidak ada (cocok)`.
Data rekaman (`processed/`) ikut ter-clone dari repo, jadi tidak perlu diunggah.
Setelah restart session, jalankan ulang bagian `import os` sampai `%cd` supaya `RESEP` dan `EXP` terisi lagi.

### Sel 5. Unduh bobot PyTorch M1
File ONNX di `models/zipformer_id` hanya untuk inferensi; fine-tune butuh checkpoint PyTorch dari `bookbot/pruned-transducer-stateless7-streaming-id`.
```python
!python unduh_m1_pytorch.py
```
Salin path dari baris `Pakai di latih_m1.py: --pretrained <PATH>`:
```python
PRE = "<PATH_YANG_DICETAK>"
```

### Sel 6. Cek arsitektur (wajib, sekitar 1 menit)
```python
!python latih_m1.py --resep $RESEP --pretrained $PRE --cek
```
Harus: `Arsitektur cocok ... vocab 33` dan loss val pretrained di bawah sekitar 1 (sebelumnya sekitar 0,27).
Kalau muncul "Ukuran bobot tidak cocok", checkpoint memakai argumen arsitektur lain; ukuran tensor di pesan error menunjukkan nilai untuk `--model-args`.

### Sel 7. Uji cepat (sekitar 1 menit)
```python
!python latih_m1.py --resep $RESEP --pretrained $PRE --exp-dir $EXP --uji-cepat
```
Harus selesai dengan `UJI CEPAT SELESAI tanpa error`, dan val loss bergerak. Hasilnya di `exp_m1_uji/`, jangan dipakai untuk laporan.

### Sel 8. Latih 5 putaran
```python
!python latih_m1.py --resep $RESEP --pretrained $PRE --exp-dir $EXP
```
- Setiap putaran: 144 rekaman latih, 16 val. Rekaman uji tidak disentuh.
- Epoch terbaik dipilih dari **loss val** dan disimpan ke `exp_m1/fold{i}/epoch-{n}.pt`.
- Baris `val X -> Y`: Y harus lebih kecil dari X. Hasil sebelumnya: turun 51–64%, epoch terbaik 14–24.
- `[N langkah dilewati]` berarti GradScaler melewati langkah karena overflow fp16; sesekali di awal wajar.
- Kalau `PERINGATAN: fine-tune tidak menurunkan loss val`, ulangi putaran itu dengan `--lr 0.002 --folds <i>`.

### Sel 9. Ekspor, zip, unduh
```python
!python ekspor_m1.py --resep $RESEP --exp-dir $EXP
!cd models && zip -r /content/m1_ft.zip m1_ft -i "*.int8.onnx" "*tokens.txt" "*info.json"
!unzip -tq /content/m1_ft.zip && ls -la /content/m1_ft.zip
from google.colab import files
files.download("/content/m1_ft.zip")
```
- Setiap putaran harus mencetak `-> .../models/m1_ft/foldN (int8 total ... MB)`.
- Zip sekitar 560 MB; catat ukurannya dari `ls -la`.
- Kalau unduhan tidak mulai, buka panel **Files** di kiri Colab, klik kanan `m1_ft.zip` → **Download**.
- **Jangan tutup Colab** sebelum zip di laptop terbukti lengkap (langkah laptop pertama).

---

## Laptop

### Ekstrak dan periksa
```powershell
cd E:\speech\belanja_asr
(Get-Item $HOME\Downloads\m1_ft.zip).Length
tar -tf $HOME\Downloads\m1_ft.zip
tar -xf $HOME\Downloads\m1_ft.zip -C models
Get-ChildItem -Recurse models\m1_ft -Filter *.onnx | Select-Object FullName
```
Harus: ukuran sama dengan `ls -la` di Colab, 25 file di zip, dan 15 file `.onnx` di `models\m1_ft\fold0..fold4`. Folder `models/m1_ft/` sudah di-`.gitignore`.

### Evaluasi dan latensi
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
Setiap `run_m1_zipformer.py` harus mencetak `encoder: encoder-epoch-N-avg-1.int8.onnx` (model hasil fine-tune). Setiap perintah menjalankan 80 kalimat pembicara itu; yang dilaporkan hanya 16 kalimat uji, karena `--uji80` dan `buat_dokumentasi.py` menyaring ke 80 rekaman uji.

### Bandingkan dan buat dokumentasi
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
| `--folds` | semua | Melanjutkan putaran tertentu, mis. `--folds 3 4`. |
| `--model-args` | kosong | Hanya kalau Sel 6 gagal karena ukuran bobot. |

## Masalah yang pernah muncul

| Gejala | Penyebab | Penanganan |
|---|---|---|
| `No such attribute 'context_size'` | Versi lama `latih_m1.py` | `git pull` (sudah diperbaiki) |
| Val loss sama persis sebelum dan sesudah uji cepat | GradScaler bawaan melewati semua langkah | `git pull` (sudah diperbaiki, `init_scale=1.0`) |
| `KeyError: '<unk>'` saat ekspor | `tokens.txt` bookbot memakai `<UNK>` dan `<eps>` | `git pull` (sudah diperbaiki di `ekspor_m1.py`) |
| Zip tidak muncul di Google Drive | Kuota Drive penuh; unggahan gagal tanpa pesan | Pakai alur panduan ini (disk Colab + `files.download`) |
| `torchaudio==2.14.0` tidak ditemukan | torchaudio tidak lagi dirilis untuk setiap versi torch | Lepas torchaudio (Sel 2) |
