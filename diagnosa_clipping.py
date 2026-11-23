import os
import numpy as np, soundfile as sf

SPK = "spk03"
hasil = []
for f in os.listdir(f"raw/{SPK}"):
    if not f.lower().endswith(".wav"):
        continue
    y, sr = sf.read(f"raw/{SPK}/{f}")
    if y.ndim > 1:
        y = y.mean(axis=1)
    m = np.abs(y) >= 0.999
    if not m.any():
        continue
    d = np.diff(np.concatenate(([0], m.astype(int), [0])))
    run = (np.where(d == -1)[0] - np.where(d == 1)[0]).max()
    hasil.append((m.mean() * 100, f, int(m.sum()), int(run), round(float(np.abs(y).max()), 3)))

print("persen  file        sampel_terpotong  run_terpanjang  puncak")
for p, f, n, run, pk in sorted(hasil, reverse=True):
    print(f"{p:6.3f}  {f:10s}  {n:8d}          {run:5d}        {pk}")
print(f"\n{len(hasil)} file terdampak")