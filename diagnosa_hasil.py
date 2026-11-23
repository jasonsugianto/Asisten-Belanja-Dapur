"""Bedah kesalahan satu file hasil. Contoh:
  python diagnosa_hasil.py hasil/m1.csv
  python diagnosa_hasil.py hasil/m0_tiny_prompt.csv"""
import sys, collections
from hasil_io import read_rows

path = sys.argv[1]
rows = read_rows(path)
sets = lambda s: set(x for x in s.split("|") if x)
has_ph = bool(rows[0].get("hyp_ph"))
heard = lambda r: r["hyp_ph"] if has_ph else r["hyp"]
print(f"{rows[0]['model']}  ({len(rows)} rekaman)\n")

# A. intent: apa yang sebenarnya terbaca
print("A. INTENT (baris = seharusnya, kolom = terbaca)")
cm = collections.Counter((r["jenis"], r["pred_intent"]) for r in rows)
labels = ["tambah", "centang", "hapus", "none"]
print(f"{'':10s}" + "".join(f"{l:>9s}" for l in labels))
for g in labels[:3]:
    print(f"{g:10s}" + "".join(f"{cm[(g, p)]:9d}" for p in labels))
no_item = sum(1 for r in rows if r["pred_intent"] == "none")
print(f"-> {no_item} kalimat berintent 'none' karena tidak ada barang terdeteksi "
      f"(intent kini ikut gagal kalau barangnya tidak ketemu)\n")

# B. contoh intent salah
print("B. CONTOH INTENT SALAH (maks 3 per jenis)")
for jenis in ("hapus", "centang"):
    n = 0
    for r in rows:
        if r["jenis"] == jenis and r["pred_intent"] != jenis and n < 3:
            n += 1
            print(f"  [{jenis}->{r['pred_intent']}] ref : {r['ref']}")
            print(f"      terbaca: {heard(r)[:110]}")
print()

# C. halusinasi (khusus teks)
if not has_ph:
    print("C. KEMUNGKINAN HALUSINASI / KELUARAN TERLALU PANJANG")
    bad = [r for r in rows if len(r["hyp"].split()) > 2 * len(r["ref"].split()) + 3]
    print(f"  {len(bad)} kalimat keluarannya jauh lebih panjang dari naskah")
    for r in bad[:4]:
        print(f"    ref: {r['ref']}\n    hyp: {r['hyp'][:120]}  ({r['latency_ms']} ms)")
    print()

# D. kesalahan fonem
def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i and j and D[i][j] == D[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            if ref[i-1] != hyp[j-1]: ops.append(("sub", ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j] + 1: ops.append(("del", ref[i-1], "")); i -= 1
        else: ops.append(("ins", "", hyp[j-1])); j -= 1
    return ops

if has_ph:
    print("D. KESALAHAN FONEM TERBANYAK (seharusnya -> terbaca)")
    sub, dele, ins, total = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    n_bar = sum(r["hyp_ph"].split().count("|") for r in rows)
    print(f"  ({n_bar} token '|' = pembatas kata dibuang dari perhitungan)")
    for r in rows:
        ref = r["ref_ph"].split()
        total.update(ref)
        for kind, a, b in align(ref, [p for p in r["hyp_ph"].split() if p != "|"]):
            (sub if kind == "sub" else dele if kind == "del" else ins)[(a, b) if kind == "sub" else (a or b)] += 1
    print("  tertukar :", ", ".join(f"{a}->{b} ({v})" for (a, b), v in sub.most_common(12)))
    print("  hilang   :", ", ".join(f"{a} ({v}/{total[a]})" for a, v in dele.most_common(8)))
    print("  tersisip :", ", ".join(f"{a} ({v})" for a, v in ins.most_common(6)))
    print("  (jika pasangan seperti e->ə atau o->ɔ mendominasi, itu beda konvensi, bukan salah dengar)\n")

# E. barang paling sering terlewat + contoh
print("E. BARANG DENGAN RECALL RENDAH")
tot, hit = collections.Counter(), collections.Counter()
for r in rows:
    g, p = sets(r["gold_items"]), sets(r["pred_items"])
    tot.update(g); hit.update(g & p)
low = sorted(((hit[i] / tot[i], i) for i in tot), key=lambda x: x[0])[:6]
for rec, item in low:
    print(f"  {item}: {hit[item]}/{tot[item]} ({rec:.0%})")
    k = 0
    for r in rows:
        if item in sets(r["gold_items"]) - sets(r["pred_items"]) and k < 2:
            k += 1
            print(f"      ref: {r['ref']}\n      terbaca: {heard(r)[:110]}")
