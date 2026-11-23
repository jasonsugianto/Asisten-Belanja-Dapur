"""Ringkasan hasil satu atau banyak model.
  python evaluasi.py                   (semua hasil/*.csv)
  python evaluasi.py hasil/m0_tiny.csv hasil/m1.csv"""
import sys, collections, glob
from hasil_io import read_rows
from metrics import score, latency_stats, error_rate
from nlu import normalize

args, spk_filter = sys.argv[1:], None
if "--spk" in args:                                  # contoh: --spk spk04 spk05
    i = args.index("--spk"); spk_filter = args[i + 1:]; args = args[:i]
files = args or sorted(glob.glob("hasil/*.csv"))
files = [f for f in files if "ringkasan" not in f]
sets = lambda s: set(x for x in s.split("|") if x)


def prep(r):
    return {"gold_intent": r["jenis"], "pred_intent": r["pred_intent"],
            "gold_items": sets(r["gold_items"]), "pred_items": sets(r["pred_items"])}


summary = []
for path in files:
    rows = read_rows(path)
    if spk_filter:
        rows = [r for r in rows if r["spk"] in spk_filter]
    if not rows:
        continue
    name = rows[0]["model"]
    P = [prep(r) for r in rows]
    s, lat = score(P), latency_stats([float(r["latency_ms"]) for r in rows])
    extra = ""
    if rows[0].get("hyp_ph"):
        nb = lambda s: [p for p in s.split() if p != "|"]      # "|" = pembatas kata, bukan fonem
        per = error_rate([nb(r["ref_ph"]) for r in rows], [nb(r["hyp_ph"]) for r in rows])
        extra = f"PER~{per:.1%} (terhadap fonem g2p, perkiraan)"
    elif rows[0].get("hyp"):
        wer = error_rate([normalize(r["ref"]).split() for r in rows], [normalize(r["hyp"]).split() for r in rows])
        extra = f"WER {wer:.1%}"
    print("=" * 78)
    print(f"{name}   ({path}, {len(rows)} rekaman{', pembicara ' + ' '.join(spk_filter) if spk_filter else ''})")
    print(f"  item F1 {s['item_F1']:.3f} (P {s['item_P']:.3f} R {s['item_R']:.3f}) | "
          f"intent {s['intent_acc']:.1%} | kalimat benar {s['kalimat_benar']:.1%} | {extra}")
    print(f"  latensi ms: p50 {lat['p50']:.0f}  p95 {lat['p95']:.0f}  max {lat['max']:.0f}")
    print("  per pembicara (item F1):", "  ".join(
        f"{k}={score([p for p, r in zip(P, rows) if r['spk'] == k])['item_F1']:.2f}"
        for k in sorted({r["spk"] for r in rows})))
    print("  per jenis (kalimat benar):", "  ".join(
        f"{k}={score([p for p, r in zip(P, rows) if r['jenis'] == k])['kalimat_benar']:.0%}"
        for k in sorted({r["jenis"] for r in rows})))
    miss, extra_c, pair = collections.Counter(), collections.Counter(), collections.Counter()
    for p in P:
        m, e = p["gold_items"] - p["pred_items"], p["pred_items"] - p["gold_items"]
        miss.update(m); extra_c.update(e)
        if len(m) == 1 and len(e) == 1:
            pair[(next(iter(m)), next(iter(e)))] += 1
    if miss:
        print("  paling sering terlewat:", ", ".join(f"{k} ({v})" for k, v in miss.most_common(6)))
    if pair:
        print("  tertukar (seharusnya -> terbaca):", ", ".join(f"{a}->{b} ({v})" for (a, b), v in pair.most_common(6)))
    summary.append((name, s["item_F1"], s["intent_acc"], s["kalimat_benar"], lat["p50"], lat["p95"]))

if len(summary) > 1:
    print("=" * 78 + "\nPERBANDINGAN")
    print(f"{'model':28s}{'itemF1':>8s}{'intent':>8s}{'benar':>8s}{'p50ms':>8s}{'p95ms':>8s}")
    for n, f1, ia, kb, p50, p95 in summary:
        print(f"{n:28s}{f1:8.3f}{ia:8.1%}{kb:8.1%}{p50:8.0f}{p95:8.0f}")
