"""Buat dokumentasi hasil 4 model dari berkas hasil yang sudah ada -> dokumentasi_hasil.md + perbandingan_80.csv
  python buat_dokumentasi.py
  python buat_dokumentasi.py --m1 hasil/m1_int8_kw30.csv --contoh 10
Semua model dibandingkan pada 80 rekaman uji (pembicara dan kalimat yang sama)."""
import argparse, collections, csv, datetime, random
from hasil_io import read_rows
from metrics import score, latency_stats
from splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--m0", default="hasil/m0_tiny_prompt_nlu2.csv")
ap.add_argument("--m1", default="hasil/m1_int8_kw30.csv")
ap.add_argument("--m2", default="hasil/m2_finetune_full.csv")
ap.add_argument("--m3", default="hasil/m3_bcresnet.csv")
ap.add_argument("--contoh", type=int, default=8)
ap.add_argument("--keluar", default="dokumentasi_hasil.md")
a = ap.parse_args()

NAMA = {"M0": "M0 Whisper tiny (+prompt)", "M1": "M1 Zipformer streaming (int8)",
        "M2": "M2 wav2vec2 XLS-R Indonesia (fine-tune)", "M3": "M3 BC-ResNet (kosakata tertutup)"}
RAW = {"M0": "hyp", "M1": "hyp_ph", "M2": "hyp", "M3": "hyp"}          # kolom keluaran mentah
JENIS_RAW = {"M0": "teks Whisper", "M1": "urutan fonem", "M2": "teks CTC", "M3": "kata kunci terdeteksi (probabilitas)"}
uji = {(s, int(i)) for f in load_folds() for s, i in f["test"]}
sets = lambda s: set(x for x in s.split("|") if x)

data = {}
for m in NAMA:
    try:
        rows = [r for r in read_rows(getattr(a, m.lower())) if (r["spk"], int(r["id"])) in uji]
    except FileNotFoundError:
        print(f"PERINGATAN: berkas {m} tidak ada, dilewati"); continue
    data[m] = {(r["spk"], int(r["id"])): r for r in rows}
    print(f"{m}: {len(rows)} rekaman uji")
if not data:
    raise SystemExit("Tidak ada berkas hasil yang terbaca.")
ada = list(data)
kunci = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"{len(kunci)} rekaman uji dimiliki semua model")
prep = lambda r: {"gold_intent": r["jenis"], "pred_intent": r["pred_intent"], "gold_items": sets(r["gold_items"]), "pred_items": sets(r["pred_items"])}
benar = lambda r: r["jenis"] == r["pred_intent"] and sets(r["gold_items"]) == sets(r["pred_items"])
esc = lambda t: (t or "").replace("|", "\\|")
L = []
L.append(f"# Dokumentasi Hasil Uji Empat Model\n\nDibuat otomatis {datetime.datetime.now():%Y-%m-%d %H:%M}. "
         f"Semua model dievaluasi pada **{len(kunci)} rekaman uji yang sama** (pembicara dan kalimat yang belum dilihat model yang dilatih).\n")

L.append("## 1. Ringkasan metrik\n")
L.append("| Model | Item F1 | Presisi | Recall | Intent | Kalimat benar | Latensi p50 / p95 (ms) |\n|---|---|---|---|---|---|---|")
for m in ada:
    rows = [data[m][k] for k in kunci]; s = score([prep(r) for r in rows]); lt = latency_stats([float(r["latency_ms"]) for r in rows])
    L.append(f"| {NAMA[m]} | {s['item_F1']:.3f} | {s['item_P']:.3f} | {s['item_R']:.3f} | {s['intent_acc']:.1%} | {s['kalimat_benar']:.1%} | {lt['p50']:.0f} / {lt['p95']:.0f} |")
L.append("\nCatatan latensi: ditulis apa adanya dari berkas hasil (M2 diukur di CPU Colab; model lain di laptop). "
         "Lihat bagian keterbatasan.\n")

L.append("## 2. Akurasi per jenis perintah (kalimat benar)\n")
L.append("| Model | tambah | centang | hapus |\n|---|---|---|---|")
for m in ada:
    cells = []
    for j in ("tambah", "centang", "hapus"):
        sel = [data[m][k] for k in kunci if data[m][k]["jenis"] == j]
        cells.append(f"{sum(benar(r) for r in sel)}/{len(sel)}")
    L.append(f"| {NAMA[m]} | " + " | ".join(cells) + " |")

L.append("\n## 3. Barang yang paling sering terlewat\n")
for m in ada:
    miss = collections.Counter()
    for k in kunci:
        r = data[m][k]; miss.update(sets(r["gold_items"]) - sets(r["pred_items"]))
    L.append(f"- **{NAMA[m]}**: " + (", ".join(f"{n} ({c})" for n, c in miss.most_common(6)) or "tidak ada"))

L.append("\n## 4. Contoh keluaran per model\n")
L.append("Setiap contoh menampilkan **keluaran mentah model**, lalu **hasil setelah pencocokan** (intent dan barang). "
         "Tanda ✓ berarti intent dan seluruh barang benar.\n")
rng = random.Random(0); pilih = []
for j, n in (("tambah", 0.5), ("centang", 0.375), ("hapus", 0.125)):
    pool = [k for k in kunci if data[ada[0]][k]["jenis"] == j]; rng.shuffle(pool)
    pilih += pool[:max(1, round(a.contoh * n))]
for k in pilih:
    r0 = data[ada[0]][k]
    L.append(f"### {k[0]} kalimat {k[1]} ({r0['jenis']})\n")
    L.append(f"- **Naskah:** {r0['ref']}\n- **Jawaban benar:** intent `{r0['jenis']}`, barang: {', '.join(sorted(sets(r0['gold_items']))) or '-'}\n")
    L.append("| Model | Keluaran mentah | Intent | Barang terbaca | |\n|---|---|---|---|---|")
    for m in ada:
        r = data[m][k]
        L.append(f"| {m} | `{esc(r.get(RAW[m], ''))[:140]}` | {r['pred_intent']} | {', '.join(sorted(sets(r['pred_items']))) or '-'} | {'✓' if benar(r) else '✗'} |")
    L.append("")
L.append("Jenis keluaran mentah: " + "; ".join(f"{m} = {JENIS_RAW[m]}" for m in ada) + ".\n")

L.append("## 5. Kasus ringkas\n")
sem = sum(all(benar(data[m][k]) for m in ada) for k in kunci); nol = sum(not any(benar(data[m][k]) for m in ada) for k in kunci)
L.append(f"- Kalimat yang benar di **semua** model: {sem} dari {len(kunci)}\n- Kalimat yang salah di **semua** model: {nol} dari {len(kunci)}")
if "M2" in ada:
    hanya = sum(benar(data["M2"][k]) and not any(benar(data[m][k]) for m in ada if m != "M2") for k in kunci)
    L.append(f"- Kalimat yang hanya benar di M2: {hanya}")

L.append("\n## 6. Keterbatasan yang perlu disebut\n")
L.append("- Set uji hanya 80 rekaman; selisih kecil antar model tidak pasti.\n"
         "- Naskah (pola kalimat dan 50 barang) sama untuk data latih dan uji; hasil berlaku untuk domain tertutup ini.\n"
         "- Latensi tidak diukur pada satu mesin yang sama untuk semua model; latensi M2 di Colab jauh di atas laptop.\n"
         "- Pencocokan teks/fonem ke barang dirancang setelah melihat kesalahan pada data yang sama (M0 dan M1 sedikit optimistis).\n"
         "- Belum diuji: noise, perangkat target (HP), barang di luar 50 item.")
open(a.keluar, "w", encoding="utf-8").write("\n".join(L) + "\n")

with open("perbandingan_80.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["pembicara", "kalimat_id", "jenis", "naskah", "barang_benar"] + [x for m in ada for x in (f"{m}_keluaran_mentah", f"{m}_intent", f"{m}_barang", f"{m}_benar")])
    for k in kunci:
        r0 = data[ada[0]][k]
        w.writerow([k[0], k[1], r0["jenis"], r0["ref"], r0["gold_items"]] +
                   [x for m in ada for x in (data[m][k].get(RAW[m], ""), data[m][k]["pred_intent"], data[m][k]["pred_items"], int(benar(data[m][k])))])
print(f"\nTertulis: {a.keluar} dan perbandingan_80.csv")
