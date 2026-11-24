"""M2: inferensi wav2vec2 + BiLSTM + CTC (fonem) + nlu_phoneme, evaluasi LOSO.

Untuk tiap rekaman dipakai checkpoint fold di mana speaker-nya menjadi TEST
(leave-one-speaker-out yang benar: model tak pernah melihat speaker itu saat latih).
Latensi = waktu inferensi satu rekaman (forward + greedy decode), CPU, konsisten
dengan cara M0/M1 mengukur (sejak audio final masuk sampai hasil keluar).

Contoh:
  python run_m2_wav2vec2.py --limit 5 --speakers spk05
  python run_m2_wav2vec2.py
  python run_m2_wav2vec2.py --out hasil/m2_wav2vec2.csv

Butuh: torch, transformers, soundfile, numpy, g2p_id_py
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data import load_dataset, BASE
from hasil_io import open_writer
from m2_model import Wav2Vec2BiLSTMCTC, greedy_decode
from nlu_phoneme import parse_phonemes, to_phonemes

SR = 16000
CKPT_DIR = BASE / "models" / "m2"


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} bukan 16 kHz; jalankan konversi.py"
    return y.mean(axis=1) if y.ndim > 1 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="processed")
    ap.add_argument("--speakers", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="hasil/m2_wav2vec2.csv")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = "cpu"  # latensi dilaporkan di CPU, sebanding dengan M0/M1

    vocab_path = CKPT_DIR / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"vocab.json tak ada di {CKPT_DIR}; jalankan train_m2.py dulu")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    id2phone = {i: p for p, i in vocab.items()}

    # muat semua checkpoint fold yang tersedia -> peta speaker-test -> model
    models_by_spk = {}
    for ckpt in sorted(CKPT_DIR.glob("fold_*.pt")):
        blob = torch.load(ckpt, map_location=device)
        model = Wav2Vec2BiLSTMCTC(len(vocab), base_model=blob["base"]).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        for spk in blob["test_spk"]:
            models_by_spk[spk] = model
        print(f"  dimuat {ckpt.name} -> test {blob['test_spk']}")
    if not models_by_spk:
        raise SystemExit(f"tak ada fold_*.pt di {CKPT_DIR}; jalankan train_m2.py dulu")

    data = load_dataset(args.audio, args.speakers, args.limit)
    missing = sorted({d["spk"] for d in data} - set(models_by_spk))
    if missing:
        raise SystemExit(f"tak ada checkpoint LOSO untuk speaker: {missing} "
                         f"(latih fold-nya dengan train_m2.py)")
    print(f"{len(data)} rekaman")

    # pemanasan agar pengukuran latensi pertama tidak bias
    with torch.no_grad():
        y0 = torch.from_numpy(load_audio(data[0]["path"])).unsqueeze(0)
        models_by_spk[data[0]["spk"]](y0)

    f, w = open_writer(args.out)
    for i, dd in enumerate(data, 1):
        y = load_audio(dd["path"])
        model = models_by_spk[dd["spk"]]
        x = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).to(device)
        t = time.perf_counter()
        with torch.no_grad():
            log_probs = model(x)                         # (T', 1, V)
            phs = greedy_decode(log_probs, id2phone, blank=0)[0]
        ms = (time.perf_counter() - t) * 1000
        p = parse_phonemes(phs)
        w.writerow({"model": "M2_wav2vec2+bilstm", "spk": dd["spk"], "id": dd["id"],
                    "jenis": dd["jenis"], "ref": dd["transcript"], "gold_items": "|".join(dd["items"]),
                    "hyp": "", "pred_intent": p["intent"], "pred_items": "|".join(p["items"]),
                    "latency_ms": round(ms, 1),
                    "ref_ph": " ".join(to_phonemes(dd["transcript"])), "hyp_ph": " ".join(phs)})
        if i % 20 == 0 or i == len(data):
            print(f"  {i}/{len(data)}  ({ms:.0f} ms)  {' '.join(phs)[:70]}")
    f.close()
    print("Tersimpan:", args.out, "-> jalankan: python evaluasi.py", args.out)


if __name__ == "__main__":
    main()
