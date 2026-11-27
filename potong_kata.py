"""Langkah 1 M3: cari batas waktu tiap kata di semua rekaman (forced alignment CTC) -> segmen.json.
Memakai model wav2vec2 Indonesia (M2) TANPA dilatih; transkrip diketahui dari naskah.
Jalankan di Colab/GPU atau lokal (CPU: sekitar 1-2 detik per rekaman).
  python potong_kata.py                 -> segmen.json
  python potong_kata.py --limit 5       -> uji cepat pada 5 rekaman pertama"""
import argparse, json, time
import numpy as np, soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from data import load_dataset
from nlu import normalize
from m3_utils import ctc_align, word_spans

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="indonesian-nlp/wav2vec2-large-xlsr-indonesian")
ap.add_argument("--audio", default="processed")
ap.add_argument("--out", default="segmen.json")
ap.add_argument("--margin", type=float, default=0.04, help="detik, ditambahkan di awal/akhir kata")
ap.add_argument("--limit", type=int)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Wav2Vec2Processor.from_pretrained(a.model)
model = Wav2Vec2ForCTC.from_pretrained(a.model).eval().to(dev)
blank = model.config.pad_token_id
sec = model.config.inputs_to_logits_ratio / 16000.0          # detik per frame (20 ms)
data = load_dataset(a.audio)[: a.limit]
print(f"{len(data)} rekaman, perangkat {dev}, {sec * 1000:.0f} ms per frame")

seg, gagal, t0 = {}, [], time.time()
for i, d in enumerate(data, 1):
    y, sr = sf.read(d["path"], dtype="float32")
    assert sr == 16000
    y = y if y.ndim == 1 else y.mean(axis=1)
    words = normalize(d["transcript"]).split()
    ids = proc.tokenizer(" ".join(words)).input_ids
    lens = [len(w) for w in words]
    key = f"{d['spk']}/{d['id']}"
    if len(ids) != sum(lens) + len(words) - 1:
        gagal.append((key, "jumlah token tidak cocok")); continue
    x = torch.from_numpy((y - y.mean()) / (y.std() + 1e-7)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logp = torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy()
    spans = ctc_align(logp, ids, blank)
    if spans is None:
        gagal.append((key, "penjajaran mustahil")); continue
    letters, pos = [], 0
    for w in words:                                            # buang token pembatas "|"
        letters.extend(spans[pos:pos + len(w)]); pos += len(w) + 1
    ws = word_spans(letters, lens)
    if any(s is None for s in ws):
        gagal.append((key, "kata tanpa rentang")); continue
    dur = len(y) / 16000
    seg[key] = [{"w": w, "s": round(max(0.0, s * sec - a.margin), 3), "e": round(min(dur, e * sec + a.margin), 3)}
                for w, (s, e) in zip(words, ws)]
    if i % 50 == 0 or i == len(data):
        print(f"  {i}/{len(data)}  {time.time() - t0:.0f} dtk")

json.dump(seg, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
dur = [x["e"] - x["s"] for v in seg.values() for x in v]
print(f"\nTersimpan {a.out}: {len(seg)} rekaman, {len(gagal)} gagal")
print(f"durasi kata: median {np.median(dur):.2f} dtk, p5 {np.percentile(dur, 5):.2f}, p95 {np.percentile(dur, 95):.2f}")
for g in gagal[:10]: print("  GAGAL:", g)
