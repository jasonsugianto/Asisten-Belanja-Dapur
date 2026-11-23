"""Pilih toleransi kata perintah (kw_rate) TANPA mengintip pembicara uji.
  python tune_kw.py hasil/m1.csv                  -> hanya menampilkan pembicara penyetelan (spk01-03)
  python tune_kw.py hasil/m1.csv --final 0.4      -> sekali saja: bandingkan di pembicara uji (spk04-05)
Hanya untuk hasil fonem (kolom hyp_ph)."""
import sys
from hasil_io import read_rows
from metrics import score
from nlu_phoneme import parse_phonemes

args = sys.argv[1:]
final = None
if "--final" in args:
    i = args.index("--final"); final = float(args[i + 1]); del args[i:i + 2]
rows = read_rows(args[0])
TUNE, TEST = {"spk01", "spk02", "spk03"}, {"spk04", "spk05"}
sets = lambda s: set(x for x in s.split("|") if x)


def evaluate(spks, kw):
    sub = [r for r in rows if r["spk"] in spks]
    P, per = [], {}
    for r in sub:
        p = parse_phonemes(r["hyp_ph"].split(), kw_rate=kw)
        P.append({"gold_intent": r["jenis"], "pred_intent": p["intent"],
                  "gold_items": sets(r["gold_items"]), "pred_items": set(p["items"])})
    for j in ("tambah", "centang", "hapus"):
        sel = [x for x, r in zip(P, sub) if r["jenis"] == j]
        per[j] = sum(x["gold_intent"] == x["pred_intent"] for x in sel) / max(len(sel), 1)
    s = score(P)
    return s, per


def show(label, spks, kw):
    s, per = evaluate(spks, kw)
    print(f"{label:14s} intent {s['intent_acc']:.1%} (tambah {per['tambah']:.0%} centang {per['centang']:.0%} "
          f"hapus {per['hapus']:.0%}) | item F1 {s['item_F1']:.3f} (P {s['item_P']:.3f} R {s['item_R']:.3f})")


if final is None:
    print(f"Pembicara penyetelan: {sorted(TUNE)}  (pembicara uji disembunyikan)")
    for kw in (None, 0.30, 0.35, 0.40, 0.50):
        show(f"kw_rate={kw}", TUNE, kw)
    print("\nPilih nilai terbaik (perhatikan P item tidak boleh turun jauh), lalu jalankan SEKALI:")
    print("  python tune_kw.py", args[0], "--final <nilai>")
else:
    print(f"Pembicara uji: {sorted(TEST)}  (jalankan ini sekali saja)")
    show("bawaan", TEST, None)
    show(f"kw_rate={final}", TEST, final)
