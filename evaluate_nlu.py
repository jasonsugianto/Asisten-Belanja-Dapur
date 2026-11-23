"""Uji NLU pada 80 kalimat naskah: (a) teks sempurna, (b) teks dengan salah dengar buatan."""
import csv, random
from nlu import parse

random.seed(0)
rows = list(csv.DictReader(open("script.csv", encoding="utf-8")))

def corrupt(text, p=0.08):
    """Simulasi salah dengar ASR: huruf diganti/dihapus pada kata >=5 huruf."""
    out = []
    for w in text.split():
        if len(w) >= 5 and random.random() < 0.3:
            i = random.randrange(len(w))
            w = w[:i] + random.choice(["", "a", "e", "i", "o", "u"]) + w[i+1:]
        out.append(w)
    return " ".join(out)

def run(transform, label):
    ok_intent = ok_all = tp = fp = fn = 0
    errs = []
    for r in rows:
        gold_items = set(r["items"].split("|")); gold_int = r["jenis"]
        pred = parse(transform(r["transcript"]))
        pi = set(pred["items"])
        ok_intent += pred["intent"] == gold_int
        tp += len(pi & gold_items); fp += len(pi - gold_items); fn += len(gold_items - pi)
        if pred["intent"] == gold_int and pi == gold_items: ok_all += 1
        else: errs.append((r["id"], r["transcript"], gold_int, sorted(gold_items), pred))
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2*P*R / max(P+R, 1e-9)
    print(f"[{label}] intent acc {ok_intent/len(rows):.1%} | item P {P:.3f} R {R:.3f} F1 {F:.3f} | kalimat benar semua {ok_all}/{len(rows)}")
    for e in errs[:5]: print("   salah:", e)

run(lambda t: t, "teks sempurna")
run(corrupt,   "teks salah dengar")
