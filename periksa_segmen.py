"""Potong beberapa klip barang/perintah dari rekaman berdasarkan segmen.json agar bisa DIDENGARKAN.
Periksa manual: apakah isi klip sesuai namanya? Kualitas M3 bergantung pada ini.
  python periksa_segmen.py            -> folder cek_klip/ (30 klip acak)"""
import json, random, os
import numpy as np, soundfile as sf
from data import load_dataset
from nlu import normalize
from m3_utils import locate

seg = json.load(open("segmen.json", encoding="utf-8"))
data = [d for d in load_dataset("processed") if f"{d['spk']}/{d['id']}" in seg]
random.seed(1); random.shuffle(data)
os.makedirs("cek_klip", exist_ok=True)
n = 0
for d in data:
    sg = seg[f"{d['spk']}/{d['id']}"]
    words = normalize(d["transcript"]).split()
    if len(sg) != len(words):
        continue
    y, sr = sf.read(d["path"], dtype="float32")
    for i, k, c in random.sample(locate(words, set(d["items"])), 1):
        a, b = int(sg[i]["s"] * sr), int(sg[i + k - 1]["e"] * sr)
        sf.write(f"cek_klip/{c.replace(' ', '_')}__{d['spk']}_{d['id']}.wav", y[a:b], sr); n += 1
    if n >= 30:
        break
print(n, "klip ditulis ke cek_klip/ (nama berkas = label yang diharapkan)")
