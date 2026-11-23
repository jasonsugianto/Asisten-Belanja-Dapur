"""M0: Whisper (faster-whisper, int8, CPU) + nlu.parse.
Contoh:
  python run_m0_whisper.py --limit 5 --speakers spk01        (uji cepat)
  python run_m0_whisper.py                                   (semua 400 rekaman)
  python run_m0_whisper.py --prompt --out hasil/m0_tiny_prompt.csv
"""
import argparse, time
import soundfile as sf
from faster_whisper import WhisperModel
from data import load_dataset
from hasil_io import open_writer
from nlu import parse, normalize, LEXICON

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="tiny")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--prompt", action="store_true", help="beri daftar barang sebagai initial_prompt")
ap.add_argument("--audio", default="processed")
ap.add_argument("--speakers", nargs="*")
ap.add_argument("--limit", type=int)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"hasil/m0_{a.size}{'_prompt' if a.prompt else ''}.csv"
prompt = ("Daftar belanja: " + ", ".join(LEXICON)) if a.prompt else None

data = load_dataset(a.audio, a.speakers, a.limit)
print(f"{len(data)} rekaman; memuat Whisper {a.size} (pertama kali mengunduh model)...")
model = WhisperModel(a.size, device="cpu", compute_type="int8", cpu_threads=a.threads)


def transcribe(y):
    segs, _ = model.transcribe(y, language="id", beam_size=1, temperature=0.0,
                               condition_on_previous_text=False, vad_filter=False,
                               initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segs)      # generator: dihitung saat dikonsumsi


y0, _ = sf.read(data[0]["path"], dtype="float32")
transcribe(y0)                                          # pemanasan, tidak dihitung

f, w = open_writer(out)
for i, d in enumerate(data, 1):
    y, sr = sf.read(d["path"], dtype="float32")
    assert sr == 16000, f"{d['path']} bukan 16 kHz; jalankan konversi.py"
    t = time.perf_counter()
    hyp = transcribe(y)
    ms = (time.perf_counter() - t) * 1000
    p = parse(hyp)
    w.writerow({"model": f"M0_{a.size}{'+prompt' if a.prompt else ''}", "spk": d["spk"], "id": d["id"],
                "jenis": d["jenis"], "ref": d["transcript"], "gold_items": "|".join(d["items"]),
                "hyp": hyp, "pred_intent": p["intent"], "pred_items": "|".join(p["items"]),
                "latency_ms": round(ms, 1)})
    if i % 20 == 0 or i == len(data):
        print(f"  {i}/{len(data)}  (terakhir {ms:.0f} ms)  {hyp!r}")
f.close()
print("Tersimpan:", out, "-> jalankan: python evaluasi.py", out)
