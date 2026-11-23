"""Cari file .wav yang formatnya palsu, lalu ubah jadi WAV asli lewat ffmpeg.
File asli dipindahkan ke raw_backup_format/ sebagai cadangan."""
import os, shutil, subprocess
import soundfile as sf

RAW, BAK = "raw", "raw_backup_format"
fixed = failed = 0

for spk in sorted(os.listdir(RAW)):
    d = f"{RAW}/{spk}"
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".wav"):
            continue
        p = f"{d}/{f}"
        try:
            sf.info(p)          # valid, lewati
            continue
        except Exception:
            pass
        os.makedirs(f"{BAK}/{spk}", exist_ok=True)
        shutil.move(p, f"{BAK}/{spk}/{f}")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", f"{BAK}/{spk}/{f}", p],
            capture_output=True, text=True)
        if r.returncode == 0:
            fixed += 1
            print("diperbaiki:", p)
        else:
            failed += 1
            shutil.move(f"{BAK}/{spk}/{f}", p)   # kembalikan file asli
            print("GAGAL (rekam ulang):", p, r.stderr.strip()[:100])

print(f"{fixed} diperbaiki, {failed} gagal")