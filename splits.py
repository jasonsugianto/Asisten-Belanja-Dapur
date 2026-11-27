"""Pembagian untuk model yang DILATIH (M1-finetune, M2, M3).

Semua pembicara membaca naskah yang sama, jadi memisahkan pembicara saja tidak cukup: model bisa
menghafal kalimat dari pembicara lain. Di sini uji = pembicara baru DAN kalimat baru.

  - 80 kalimat dibagi 5 kelompok (16 kalimat), seimbang per jenis perintah, dan kemunculan
    satu barang disebar ke kelompok berbeda supaya barang tidak hilang dari data latih.
  - Putaran i : uji  = pembicara i  x kelompok i
                val  = pembicara i+1 x kelompok i+1
                latih = 3 pembicara lain x 3 kelompok lain (sekitar 48 kalimat x 3 orang)
  - Kelima putaran bersama-sama menguji 80 rekaman (tiap pasangan pembicara-kalimat tepat sekali).

Jalankan `python splits.py` untuk membuat splits.json (BEKUKAN; jangan diubah setelah pelatihan)."""
import json, random
from pathlib import Path
from data import load_script, load_dataset, BASE

K = 5
SPLIT_FILE = BASE / "splits.json"


def sentence_groups(script, k=K, seed=0):
    rng = random.Random(seed)
    freq = {}
    for r in script.values():
        for it in r["items"]:
            freq[it] = freq.get(it, 0) + 1
    ids = sorted(script)
    groups = [[] for _ in range(k)]
    cap = {}
    for j in ("tambah", "centang", "hapus"):
        n = sum(1 for i in ids if script[i]["jenis"] == j)
        cap[j] = [n // k + (1 if g < n % k else 0) for g in range(k)]      # kuota per kelompok
    # kalimat dengan barang paling langka dulu; acak ringan agar tidak terikat urutan nomor
    order = sorted(ids, key=lambda i: (min(freq[x] for x in script[i]["items"]), rng.random()))
    items_in = [dict() for _ in range(k)]
    for i in order:
        j, its = script[i]["jenis"], script[i]["items"]
        ok = [g for g in range(k) if sum(1 for x in groups[g] if script[x]["jenis"] == j) < cap[j][g]]
        # pilih kelompok yang paling sedikit sudah memuat barang-barang kalimat ini
        g = min(ok, key=lambda g: (sum(items_in[g].get(x, 0) for x in its), len(groups[g]), rng.random()))
        groups[g].append(i)
        for x in its:
            items_in[g][x] = items_in[g].get(x, 0) + 1
    return [sorted(g) for g in groups]


def make_folds(speakers, groups):
    n, folds = len(speakers), []
    for i in range(n):
        t, v = i % n, (i + 1) % n
        used = {i % K, (i + 1) % K}                      # kelompok uji dan val
        train = [(s, sid) for si, s in enumerate(speakers) if si not in (t, v)
                 for g in range(K) if g not in used for sid in groups[g]]
        folds.append({"fold": i,
                      "test": [(speakers[t], sid) for sid in groups[i % K]],
                      "val": [(speakers[v], sid) for sid in groups[(i + 1) % K]],
                      "train": train})
    return folds


def load_folds(speakers=None):
    """Baca splits.json (buat bila belum ada). Mengembalikan daftar putaran."""
    if SPLIT_FILE.exists():
        d = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        return [{k: ([tuple(x) for x in v] if isinstance(v, list) else v) for k, v in f.items()}
                for f in d["folds"]]
    script = load_script()
    spks = speakers or sorted({r["spk"] for r in load_dataset("processed")})
    groups = sentence_groups(script)
    folds = make_folds(spks, groups)
    SPLIT_FILE.write_text(json.dumps({"speakers": spks, "groups": groups, "folds": folds},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return folds


def report(folds, script):
    for f in folds:
        cnt = {}
        for _, sid in f["train"]:
            for it in script[sid]["items"]:
                cnt[it] = cnt.get(it, 0) + 1
        tr_sent = {sid for _, sid in f["train"]}
        test_items = {it for _, sid in f["test"] for it in script[sid]["items"]}
        unseen = sorted(it for it in test_items if cnt.get(it, 0) == 0)
        jarang = sorted(it for it in test_items if 0 < cnt.get(it, 0) <= 3)
        print(f"putaran {f['fold']}: latih {len(f['train'])} rekaman ({len(tr_sent)} kalimat), "
              f"val {len(f['val'])}, uji {len(f['test'])} | barang uji tak ada di latih: {unseen or '-'} "
              f"| hanya 1 kalimat latih: {len(jarang)}")
    allt = [x for f in folds for x in f["test"]]
    print(f"\nrekaman uji total {len(allt)}, unik {len(set(allt))}")
    # tidak boleh ada kalimat uji yang muncul di data latih putaran yang sama
    bad = sum(1 for f in folds if {sid for _, sid in f['test']} & {sid for _, sid in f['train']})
    print("kebocoran kalimat uji->latih:", "TIDAK ADA" if bad == 0 else f"{bad} putaran!")


if __name__ == "__main__":
    fl = load_folds()
    report(fl, load_script())
    print("tersimpan:", SPLIT_FILE)
