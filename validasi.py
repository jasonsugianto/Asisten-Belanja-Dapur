"""Validasi rekaman di raw/spkXX/ terhadap script.csv, lalu tulis metadata.csv.
Jalankan: python validasi.py   (butuh: pip install soundfile numpy)"""
import csv, os
import numpy as np, soundfile as sf

RAW = "raw"
script = {int(r["id"]): r for r in csv.DictReader(open("script.csv", encoding="utf-8"))}

def find(spk, sid):
    for name in (f"{sid}.wav", f"{sid:02d}.wav", f"{sid:03d}.wav"):
        p = f"{RAW}/{spk}/{name}"
        if os.path.exists(p): return p

rows, problems = [], []
speakers = sorted(d for d in os.listdir(RAW) if os.path.isdir(f"{RAW}/{d}")) if os.path.isdir(RAW) else []
if not speakers: print("Folder raw/ belum ada isinya.")
for spk in speakers:
    for sid, r in script.items():
        p = find(spk, sid)
        if not p: problems.append((spk, sid, "file hilang")); continue
        try:
            y, sr = sf.read(p)
        except Exception:
            problems.append((spk, sid, "format tidak terbaca")); continue
        if y.ndim > 1: y = y.mean(axis=1)
        dur, peak, rms = len(y) / sr, np.abs(y).max(), np.sqrt((y ** 2).mean())
        if dur < 0.5: problems.append((spk, sid, f"terlalu pendek {dur:.2f}s"))
        if dur > 15:  problems.append((spk, sid, f"terlalu panjang {dur:.1f}s"))
        clipped = np.mean(np.abs(y) >= 0.999)
        if clipped > 0.001: problems.append((spk, sid, f"clipping {clipped:.2%}"))
        if rms < 0.005:  problems.append((spk, sid, "terlalu pelan"))
        rows.append([p, spk, sid, r["jenis"], r["transcript"], r["items"], sr, round(dur, 2)])

with open("metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "speaker", "script_id", "jenis", "transcript", "items", "sr", "durasi"])
    w.writerows(rows)
print(f"{len(rows)} file terbaca, {len(problems)} masalah")
for p in problems: print(" ", p)
