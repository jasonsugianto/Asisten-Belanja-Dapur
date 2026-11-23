"""Hitung ulang NLU dari hasil ASR yang SUDAH tersimpan (tanpa menjalankan model lagi).
  python reparse.py hasil/m1.csv            -> hasil/m1_nlu2.csv
  python reparse.py hasil/m1_int8.csv
  python reparse.py hasil/m0_tiny_prompt.csv
Opsi: --kw-rate 0.35  (toleransi lebih longgar khusus kata perintah, hanya untuk fonem)
      --tag nama           (akhiran nama file keluaran, default nlu2)"""
import sys
from pathlib import Path
from hasil_io import read_rows, open_writer
from nlu import parse
from nlu_phoneme import parse_phonemes

args = sys.argv[1:]
kw = None
if "--kw-rate" in args:
    i = args.index("--kw-rate"); kw = float(args[i + 1]); del args[i:i + 2]
tag = "nlu2"
if "--tag" in args:
    i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
src = Path(args[0])
rows = read_rows(src)
out = src.with_name(f"{src.stem}_{tag}.csv")
f, w = open_writer(out)
for r in rows:
    p = parse_phonemes(r["hyp_ph"].split(), kw_rate=kw) if r.get("hyp_ph") else parse(r["hyp"])
    r["pred_intent"], r["pred_items"] = p["intent"], "|".join(p["items"])
    r["model"] += f"+{tag}"
    w.writerow(r)
f.close()
print("Tersimpan:", out, "-> python evaluasi.py", src, out)
