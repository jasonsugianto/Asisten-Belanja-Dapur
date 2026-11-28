"""Analisis kaskade cepat -> lambat dari hasil yang SUDAH ADA (tanpa menjalankan model lagi).
Ide produk: tampilkan hasil model cepat (mis. M1, ~30 ms) lebih dulu, lalu koreksi dengan model akurat
(mis. M2, ~1,7 dtk) setelah selesai. Skrip ini menghitung akurasi tiap strategi dan seberapa sering
pengguna akan melihat koreksi.
  python gabung_model.py hasil/m1_int8_kw30.csv hasil/m2_finetune_full.csv [--uji80]"""
import sys
from hasil_io import read_rows
from metrics import score, latency_stats

args = [x for x in sys.argv[1:] if not x.startswith("--")]
if len(args) != 2:
    raise SystemExit("Pakai: python gabung_model.py cepat.csv lambat.csv [--uji80]")
sets = lambda s: set(x for x in s.split("|") if x)
cepat = {(r["spk"], int(r["id"])): r for r in read_rows(args[0])}
lambat = {(r["spk"], int(r["id"])): r for r in read_rows(args[1])}
keys = sorted(set(cepat) & set(lambat))
if "--uji80" in sys.argv:
    from splits import load_folds
    uji = {(s, int(i)) for f in load_folds() for s, i in f["test"]}
    keys = [k for k in keys if k in uji]
if not keys:
    raise SystemExit("Tidak ada rekaman yang sama di kedua berkas.")
print(f"{len(keys)} rekaman bersama | cepat: {list(cepat.values())[0]['model']} | lambat: {list(lambat.values())[0]['model']}\n")


def row(k, items, intent):
    return {"gold_intent": cepat[k]["jenis"], "pred_intent": intent,
            "gold_items": sets(cepat[k]["gold_items"]), "pred_items": items}


S = {
    "cepat saja": [row(k, sets(cepat[k]["pred_items"]), cepat[k]["pred_intent"]) for k in keys],
    "lambat saja": [row(k, sets(lambat[k]["pred_items"]), lambat[k]["pred_intent"]) for k in keys],
    "irisan (hanya yang disepakati)": [row(k, sets(cepat[k]["pred_items"]) & sets(lambat[k]["pred_items"]),
                                           lambat[k]["pred_intent"]) for k in keys],
    "gabungan (union)": [row(k, sets(cepat[k]["pred_items"]) | sets(lambat[k]["pred_items"]),
                             lambat[k]["pred_intent"]) for k in keys],
}
lc = latency_stats([float(cepat[k]["latency_ms"]) for k in keys]); ll = latency_stats([float(lambat[k]["latency_ms"]) for k in keys])
print(f"{'strategi':34s}{'itemF1':>8s}{'intent':>8s}{'benar':>8s}")
for nama, rows in S.items():
    s = score(rows); print(f"{nama:34s}{s['item_F1']:8.3f}{s['intent_acc']:8.1%}{s['kalimat_benar']:8.1%}")
print(f"\nKaskade (tampil cepat, lalu diganti hasil lambat): hasil pertama ~{lc['p50']:.0f} ms (p95 {lc['p95']:.0f}), "
      f"hasil final ~{ll['p50']:.0f} ms (p95 {ll['p95']:.0f}). Akurasi akhir = 'lambat saja'.")

# seberapa sering pengguna melihat koreksi
ok = lambda r: r["gold_intent"] == r["pred_intent"] and r["gold_items"] == r["pred_items"]
awal, akhir = S["cepat saja"], S["lambat saja"]
berubah = sum(1 for a, b in zip(awal, akhir) if a["pred_items"] != b["pred_items"] or a["pred_intent"] != b["pred_intent"])
membaik = sum(1 for a, b in zip(awal, akhir) if not ok(a) and ok(b))
memburuk = sum(1 for a, b in zip(awal, akhir) if ok(a) and not ok(b))
tambah = sum(len(b["pred_items"] - a["pred_items"]) for a, b in zip(awal, akhir))
buang = sum(len(a["pred_items"] - b["pred_items"]) for a, b in zip(awal, akhir))
n = len(keys)
print(f"\nDampak koreksi pada pengguna (dari {n} kalimat):")
print(f"  kalimat yang tampilannya berubah setelah koreksi : {berubah} ({berubah / n:.0%})")
print(f"  salah -> benar (diperbaiki)                      : {membaik}")
print(f"  benar -> salah (dirusak koreksi)                 : {memburuk}")
print(f"  barang DITAMBAHKAN oleh koreksi                  : {tambah}")
print(f"  barang DIHAPUS oleh koreksi (sudah sempat tampil): {buang}")
