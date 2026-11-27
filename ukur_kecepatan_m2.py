"""Ukur latensi CPU varian M2 TANPA mengunduh model (bobot acak, arsitektur XLS-R large).
Tujuannya memilih varian mana yang layak di-fine-tune. Hanya kecepatan, bukan akurasi.
  python ukur_kecepatan_m2.py            (audio 4 dtk, 5 kali ulang per varian)"""
import argparse, gc, time
import numpy as np, torch
from transformers import Wav2Vec2ForCTC
from m2_ringan import potong_lapisan, int8, konfigurasi_xlsr_large

ap = argparse.ArgumentParser()
ap.add_argument("--detik", type=float, default=4.0)
ap.add_argument("--ulang", type=int, default=5)
ap.add_argument("--lapisan", nargs="*", type=int, default=[24, 12, 6])
a = ap.parse_args()

torch.manual_seed(0)
x = torch.randn(1, int(16000 * a.detik))
print(f"torch {torch.__version__}, {torch.get_num_threads()} thread, audio {a.detik} dtk")
print(f"{'lapisan':>8s} {'presisi':>8s} {'p50 ms':>9s} {'x vs dasar':>11s}")
dasar = None
for k in a.lapisan:
    m = Wav2Vec2ForCTC(konfigurasi_xlsr_large()).eval()
    potong_lapisan(m, k)
    for nama in ("fp32", "int8"):
        if nama == "int8":
            mod = int8(m, inplace=True)                      # di tempat: hemat memori
        else:
            mod = m
        with torch.no_grad():
            mod(x)                                           # pemanasan
            t = []
            for _ in range(a.ulang):
                s = time.perf_counter(); mod(x); t.append((time.perf_counter() - s) * 1000)
        p50 = float(np.median(t)); dasar = dasar or p50
        print(f"{k:8d} {nama:>8s} {p50:9.0f} {dasar / p50:10.1f}x")
    del m, mod; gc.collect()
