"""Gabungkan beberapa file hasil (kolom sama) menjadi satu.
  python gabung_hasil.py hasil/m2_finetune.csv hasil/m2_finetune_f4.csv -o hasil/m2_finetune_full.csv"""
import sys
from hasil_io import read_rows, open_writer

args = sys.argv[1:]
if "-o" not in args:
    raise SystemExit("Pakai: python gabung_hasil.py a.csv b.csv -o gabungan.csv")
i = args.index("-o"); out = args[i + 1]; srcs = args[:i]
seen, rows = set(), []
for p in srcs:
    for r in read_rows(p):
        key = (r["spk"], r["id"])
        if key in seen:
            print("lewati duplikat:", key); continue
        seen.add(key); rows.append(r)
f, w = open_writer(out)
w.writerows(rows); f.close()
print(f"{len(rows)} rekaman -> {out}; pembicara: {sorted({r['spk'] for r in rows})}")
