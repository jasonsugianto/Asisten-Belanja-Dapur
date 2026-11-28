"""Perpanjang batas kata di segmen.json (penjajaran CTC cenderung memotong bagian belakang kata).
Aturan, dihitung dari segmen ASLI (cadangan segmen_asli.json dibuat otomatis, aman dijalankan berulang):
  - batas antar dua kata = titik tengah celah, tetapi akhir kata diperpanjang paling banyak --maks-akhir
    dan awal kata paling banyak --maks-awal
  - kata terakhir diperpanjang --akhir detik; kata pertama dimajukan --awal detik
Contoh:  python perbaiki_segmen.py
         python perbaiki_segmen.py --maks-akhir 0.25     (kalau masih terpotong)"""
import argparse, json, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--maks-akhir", type=float, default=0.20)
ap.add_argument("--maks-awal", type=float, default=0.08)
ap.add_argument("--akhir", type=float, default=0.20, help="perpanjangan setelah kata terakhir")
ap.add_argument("--awal", type=float, default=0.08, help="pemajuan sebelum kata pertama")
a = ap.parse_args()

orig, cur = Path("segmen_asli.json"), Path("segmen.json")
if not orig.exists():
    shutil.copy(cur, orig); print("cadangan dibuat: segmen_asli.json")
seg = json.load(open(orig, encoding="utf-8"))
dur_before = [w["e"] - w["s"] for v in seg.values() for w in v]
out = {}
for key, ws in seg.items():
    new = [dict(w) for w in ws]
    for i, w in enumerate(ws):
        s, e = w["s"], w["e"]
        if i + 1 < len(ws):
            mid = (e + ws[i + 1]["s"]) / 2
            new[i]["e"] = round(max(e, min(mid, e + a.maks_akhir)), 3)
        else:
            new[i]["e"] = round(e + a.akhir, 3)
        if i > 0:
            mid = (ws[i - 1]["e"] + s) / 2
            new[i]["s"] = round(min(s, max(mid, s - a.maks_awal)), 3)
        else:
            new[i]["s"] = round(max(0.0, s - a.awal), 3)
    out[key] = new
json.dump(out, open(cur, "w", encoding="utf-8"), ensure_ascii=False)
dur_after = [w["e"] - w["s"] for v in out.values() for w in v]
print(f"durasi kata median {np.median(dur_before):.2f} -> {np.median(dur_after):.2f} dtk | "
      f"p5 {np.percentile(dur_before, 5):.2f} -> {np.percentile(dur_after, 5):.2f} | "
      f"p95 {np.percentile(dur_before, 95):.2f} -> {np.percentile(dur_after, 95):.2f}")
print("segmen.json diperbarui. Jalankan lagi: python periksa_segmen.py  lalu dengarkan klip baru.")
