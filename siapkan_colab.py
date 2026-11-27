"""Kumpulkan file yang perlu diunggah ke Colab menjadi belanja_colab.zip."""
import zipfile
from pathlib import Path

B = Path(__file__).resolve().parent
kode = ["data.py", "metrics.py", "hasil_io.py", "nlu.py", "augment.py", "splits.py", "latih_m2.py",
        "script.csv", "splits.json"]
ada = [k for k in kode if (B / k).exists()]
kurang = [k for k in kode if k not in ada]
if kurang:
    raise SystemExit(f"Belum ada: {kurang}. Jalankan 'python splits.py' dulu bila splits.json yang hilang.")
if not (B / "processed").is_dir():
    raise SystemExit("Folder processed/ tidak ada (jalankan konversi.py).")
with zipfile.ZipFile(B / "belanja_colab.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for k in ada:
        z.write(B / k, k)
    n = 0
    for p in sorted((B / "processed").rglob("*.wav")):
        z.write(p, p.relative_to(B).as_posix()); n += 1
print(f"belanja_colab.zip: {len(ada)} berkas kode + {n} rekaman")
