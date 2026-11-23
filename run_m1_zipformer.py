"""M1: Zipformer RNN-T streaming (fonem, sherpa-onnx) + nlu_phoneme.
Latensi = waktu sejak chunk TERAKHIR diterima sampai hasil akhir keluar.
Contoh:
  python run_m1_zipformer.py --limit 5 --speakers spk01
  python run_m1_zipformer.py
  python run_m1_zipformer.py --int8 --out hasil/m1_int8.csv"""
import argparse, time
import numpy as np, soundfile as sf
import sherpa_onnx
from pathlib import Path
from data import load_dataset
from hasil_io import open_writer
from nlu_phoneme import parse_phonemes, to_phonemes

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="models/zipformer_id")
ap.add_argument("--int8", action="store_true", help="pakai berkas *.int8.onnx bila ada")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--chunk-ms", type=int, default=100)
ap.add_argument("--audio", default="processed")
ap.add_argument("--speakers", nargs="*")
ap.add_argument("--limit", type=int)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"hasil/m1{'_int8' if a.int8 else ''}.csv"


def pick(d, key):
    c = sorted(d.glob(f"{key}*.onnx"))
    if not c:
        raise SystemExit(f"Tidak ada {key}*.onnx di {d}; jalankan unduh_model.py")
    int8 = [p for p in c if "int8" in p.name]
    plain = [p for p in c if "int8" not in p.name]
    chosen = (int8 or plain)[0] if a.int8 else (plain or int8)[0]
    return str(chosen)


d = Path(__file__).resolve().parent / a.model_dir
enc, dec, joi = pick(d, "encoder"), pick(d, "decoder"), pick(d, "joiner")
print("encoder:", Path(enc).name, "| decoder:", Path(dec).name, "| joiner:", Path(joi).name)
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(d / "tokens.txt"), encoder=enc, decoder=dec, joiner=joi,
    num_threads=a.threads, sample_rate=16000, feature_dim=80,
    decoding_method="greedy_search")


def tokens_of(stream):
    toks = rec.get_result_all(stream).tokens               # daftar fonem satu per satu
    return [t.replace("▁", "").strip() for t in toks if t.replace("▁", "").strip()]


def run_stream(y):
    """Kembalikan (fonem, latensi_ms setelah chunk terakhir, rata-rata ms per chunk)."""
    n = int(16000 * a.chunk_ms / 1000)
    chunks = [y[i:i + n] for i in range(0, len(y), n)]
    s, per_chunk = rec.create_stream(), []
    for c in chunks[:-1]:
        t = time.perf_counter()
        s.accept_waveform(16000, c)
        while rec.is_ready(s):
            rec.decode_stream(s)
        per_chunk.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()                                 # <- mulai hitung latensi
    s.accept_waveform(16000, chunks[-1])
    s.accept_waveform(16000, np.zeros(int(0.4 * 16000), dtype=np.float32))   # padding akhir
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    ms = (time.perf_counter() - t) * 1000
    return tokens_of(s), ms, float(np.mean(per_chunk)) if per_chunk else 0.0


data = load_dataset(a.audio, a.speakers, a.limit)
print(f"{len(data)} rekaman, chunk {a.chunk_ms} ms")
y0, _ = sf.read(data[0]["path"], dtype="float32")
run_stream(y0)                                              # pemanasan

f, w = open_writer(out)
for i, dd in enumerate(data, 1):
    y, sr = sf.read(dd["path"], dtype="float32")
    assert sr == 16000, f"{dd['path']} bukan 16 kHz; jalankan konversi.py"
    phs, ms, _ = run_stream(y)
    p = parse_phonemes(phs)
    w.writerow({"model": f"M1_zipformer{'_int8' if a.int8 else ''}", "spk": dd["spk"], "id": dd["id"],
                "jenis": dd["jenis"], "ref": dd["transcript"], "gold_items": "|".join(dd["items"]),
                "hyp": "", "pred_intent": p["intent"], "pred_items": "|".join(p["items"]),
                "latency_ms": round(ms, 1),
                "ref_ph": " ".join(to_phonemes(dd["transcript"])), "hyp_ph": " ".join(phs)})
    if i % 20 == 0 or i == len(data):
        print(f"  {i}/{len(data)}  ({ms:.0f} ms)  {' '.join(phs)[:70]}")
f.close()
print("Tersimpan:", out, "-> jalankan: python evaluasi.py", out)
