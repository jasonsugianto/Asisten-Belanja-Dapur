"""Uji NLU fonem: teks naskah -> fonem -> diberi kesalahan acak (sub/hapus/sisip) -> parse."""
import csv, random, time
from nlu_phoneme import to_phonemes, parse_phonemes

PH = ["a","i","u","e","o","ə","k","t","p","b","d","ɡ","s","m","n","l","r","h","j","w","ŋ","tʃ","dʒ","ɲ"]
rows = list(csv.DictReader(open("script.csv", encoding="utf-8")))

def corrupt(phs, rate, rng):
    out = []
    for p in phs:
        r = rng.random()
        if r < rate / 3: continue                       # hapus
        elif r < 2 * rate / 3: out.append(rng.choice(PH))   # ganti
        else: out.append(p)
        if rng.random() < rate / 3: out.append(rng.choice(PH))  # sisip
    return out

def run(rate, seed=0, reps=3):
    rng = random.Random(seed); tp = fp = fn = ok = n = 0; ms = []
    for _ in range(reps if rate else 1):
        for r in rows:
            gold = set(r["items"].split("|")); phs = corrupt(to_phonemes(r["transcript"]), rate, rng)
            t = time.perf_counter(); pred = parse_phonemes(phs); ms.append((time.perf_counter()-t)*1000)
            pi = set(pred["items"]); tp += len(pi & gold); fp += len(pi - gold); fn += len(gold - pi)
            ok += pred["intent"] == r["jenis"] and pi == gold; n += 1
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1); F = 2*P*R/max(P+R,1e-9)
    ms.sort()
    print(f"PER~{rate:.0%}: item P {P:.3f} R {R:.3f} F1 {F:.3f} | kalimat benar {ok/n:.1%} | latensi p50 {ms[len(ms)//2]:.1f} ms p95 {ms[int(len(ms)*.95)]:.1f} ms")

for rate in (0, 0.05, 0.10, 0.20): run(rate)
