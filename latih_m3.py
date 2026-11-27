"""M3: BC-ResNet (kosakata tertutup). Jendela 1,2 dtk -> 52 keluaran sigmoid (50 barang + hapus + centang),
MULTI-LABEL: sebuah jendela bisa memuat beberapa kata kunci sekaligus, dan keluaran semua-nol berarti
'tidak ada kata kunci' (kata lain / senyap). Pada kalimat utuh, jendela digeser tiap 0,1 dtk dan sebuah
kelas dianggap hadir bila probabilitasnya >= ambang pada beberapa jendela berurutan.
Dilatih dari nol dengan pembagian splits.py. Prasyarat: segmen.json (python potong_kata.py).
  python latih_m3.py --uji-cepat     <- dulu (1-3 menit)
  python latih_m3.py                 <- 5 putaran
Keluaran: hasil/m3_bcresnet.csv (80 rekaman uji). Latensi = waktu memproses 2 jendela terakhir + dekode
setelah potongan audio terakhir (di --eval-device, bawaan cpu)."""
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--folds", nargs="*", type=int)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--per-epoch", type=int, default=2048)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--tau", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos-weight", type=float, default=4.0)
ap.add_argument("--eval-device", default="cpu")
ap.add_argument("--audio", default="processed")
ap.add_argument("--segmen", default="segmen.json")
ap.add_argument("--tag", default="bcresnet")
ap.add_argument("--uji-cepat", action="store_true")
a = ap.parse_args()

try:
    import torch, torch.nn.functional as F
except ImportError:
    raise SystemExit("Butuh torch: pip install torch")
from data import load_dataset
from splits import load_folds
from hasil_io import open_writer
from m3_utils import locate, decode, SR
from metrics import score
from nlu import normalize
from augment import add_noise, speed_perturb
from bcresnet import BCResNet

random.seed(0); np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("perangkat latih:", dev, "| perangkat ukur latensi:", a.eval_device)

# ---- data ----------------------------------------------------------------------------------------
meta = {(d["spk"], d["id"]): d for d in load_dataset(a.audio)}
wav = {}
for k, d in meta.items():
    y, sr = sf.read(d["path"], dtype="float32"); assert sr == SR, f"{d['path']} bukan 16 kHz"
    wav[k] = y if y.ndim == 1 else y.mean(axis=1)
if not Path(a.segmen).exists():
    raise SystemExit(f"{a.segmen} belum ada. Jalankan dulu: python potong_kata.py")
seg = json.load(open(a.segmen, encoding="utf-8"))
items = sorted({it for d in meta.values() for it in d["items"]})
classes = items + ["hapus", "centang"]
cid = {c: i for i, c in enumerate(classes)}
C = len(classes)
WIN, HALF, HOP = 1.2, 0.6, 0.1
print(f"{C} kelas ({len(items)} barang + hapus + centang), jendela {WIN} dtk, langkah {HOP} dtk")


def events_for(key):
    """(kata_kunci [(kelas,awal,akhir)], kata_lain [(awal,akhir)], celah [(awal,akhir)]) atau None."""
    d = meta[key]; words = normalize(d["transcript"]).split()
    sg = seg.get(f"{key[0]}/{key[1]}")
    if sg is None or len(sg) != len(words):
        return None
    kw, covered = [], set()
    for i, n, c in locate(words, set(d["items"])):
        kw.append((c, sg[i]["s"], sg[i + n - 1]["e"])); covered.update(range(i, i + n))
    other = [(sg[i]["s"], sg[i]["e"]) for i in range(len(words)) if i not in covered]
    dur = len(wav[key]) / SR
    gaps = [(0.0, sg[0]["s"])] + [(sg[i]["e"], sg[i + 1]["s"]) for i in range(len(sg) - 1)] + [(sg[-1]["e"], dur)]
    return kw, other, [(s, e) for s, e in gaps if e - s >= 0.12]


def crop(y, center_idx, n):
    out = np.zeros(n, np.float32); s = int(center_idx) - n // 2
    lo, hi = max(s, 0), min(s + n, len(y))
    if hi > lo: out[lo - s:hi - s] = y[lo:hi]
    return out


def targets(kw, w0, w1):
    """Label multi-label untuk jendela yang menutup waktu asli [w0,w1]. Kata terpotong sebagian diabaikan (mask 0)."""
    y = np.zeros(C, np.float32); m = np.ones(C, np.float32)
    for c, s, e in kw:
        frac = max(0.0, min(e, w1) - max(s, w0)) / max(min(e - s, WIN), 1e-3)
        i = cid[c]
        if frac >= 0.75: y[i] = 1.0
        elif frac > 0.25 and y[i] == 0.0: m[i] = 0.0
    m[y > 0] = 1.0
    return y, m


def make_sample(key, kind, s, e, rng):
    """kind: 'kw' (berpusat pada kata kunci), 'other' (kata lain), 'gap' (senyap/celah), 'noise' (derau saja)."""
    y, n = wav[key], int(WIN * SR); kw = EV[key][0]
    if kind == "noise":
        return (10 ** rng.uniform(-5, -2)) * np.random.randn(n).astype(np.float32), np.zeros(C, np.float32), np.ones(C, np.float32)
    if kind == "gap":
        x = np.zeros(n, np.float32); g = y[int(s * SR):int(e * SR)][:n]; o = (n - len(g)) // 2; x[o:o + len(g)] = g
        return x + (10 ** rng.uniform(-5, -2)) * np.random.randn(n).astype(np.float32), np.zeros(C, np.float32), np.ones(C, np.float32)
    center = (s + e) / 2 + rng.uniform(-0.25, 0.25) if kind == "kw" else (s + e) / 2 + rng.uniform(-0.15, 0.15)
    f = rng.uniform(0.9, 1.1) if rng.random() < 0.5 else 1.0
    if f != 1.0:
        ex = speed_perturb(crop(y, center * SR, int(1.8 * SR)), f).astype(np.float32)
        x = crop(ex, len(ex) / 2, n)
    else:
        x = crop(y, center * SR, n)
    lab, mask = targets(kw, center - HALF * f, center + HALF * f)
    x = x * rng.uniform(0.4, 1.6)
    if rng.random() < 0.7:
        x = add_noise(x, np.random.randn(n).astype(np.float32), rng.uniform(5, 30)).astype(np.float32)
    return x, lab, mask


def window_probs(model, y, device, lo=0, hi=None):
    """Probabilitas (sigmoid) tiap kelas untuk jendela ke-lo..hi-1 (pusat tiap HOP detik)."""
    N = int(len(y) / SR / HOP) + 1; hi = N if hi is None else hi
    W = np.stack([crop(y, i * HOP * SR, int(WIN * SR)) for i in range(lo, hi)])
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(W).to(device))).cpu().numpy()


def run_sentence(model, key, device, params):
    y = wav[key]; N = int(len(y) / SR / HOP) + 1; k = max(N - 2, 0)
    P0 = window_probs(model, y, device, 0, k) if k > 0 else np.zeros((0, C), np.float32)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()                                      # mulai hitung: potongan terakhir tiba
    P = np.concatenate([P0, window_probs(model, y, device, k, N)])
    intent, its, best = decode(P, classes, *params)
    if device == "cuda": torch.cuda.synchronize()
    return intent, its, best, (time.perf_counter() - t0) * 1000


def val_score(model, keys, device, params=(0.5, 0.5, 2), cache=None):
    P = []
    for k in keys:
        if cache is not None and k in cache: pr = cache[k]
        else:
            pr = window_probs(model, wav[k], device)
            if cache is not None: cache[k] = pr
        intent, its, _ = decode(pr, classes, *params)
        P.append({"gold_intent": meta[k]["jenis"], "pred_intent": intent,
                  "gold_items": set(meta[k]["items"]), "pred_items": set(its)})
    return score(P)


# ---- putaran -------------------------------------------------------------------------------------
folds = load_folds()
sel = a.folds if a.folds is not None else ([0] if a.uji_cepat else range(len(folds)))
akhiran = "_f" + "_".join(map(str, sel)) if (a.folds is not None and not a.uji_cepat) else ""
outdir = "hasil_uji" if a.uji_cepat else "hasil"
fo, wo = open_writer(f"{outdir}/m3_{a.tag}{akhiran}.csv")
epochs = 2 if a.uji_cepat else a.epochs
per_epoch = 128 if a.uji_cepat else a.per_epoch
steps_per_epoch = max(per_epoch // a.batch, 1)
pw = torch.full((C,), a.pos_weight, device=dev)
summary, EV = [], {}

for fi in sel:
    fold = folds[fi]
    tr = [k for k in fold["train"] if k in meta]; va = [k for k in fold["val"] if k in meta]
    te = [k for k in fold["test"] if k in meta]
    if a.uji_cepat: tr, va, te = tr[:24], va[:4], te[:4]
    pools = {"kw": [], "other": [], "gap": []}; no_seg = 0; per_class = {c: 0 for c in classes}
    for k in tr:
        ev = events_for(k)
        if ev is None: no_seg += 1; continue
        EV[k] = ev
        for c, s, e in ev[0]: pools["kw"].append((k, s, e)); per_class[c] += 1
        pools["other"] += [(k, s, e) for s, e in ev[1]]; pools["gap"] += [(k, s, e) for s, e in ev[2]]
    cnt = list(per_class.values()); kosong = [c for c, v in per_class.items() if v == 0]
    print(f"\n=== putaran {fi}: latih {len(tr)} rekaman ({no_seg} tanpa segmen), val {len(va)}, uji {len(te)} ===")
    print(f"  klip per kelas (min/median/maks): {min(cnt)} {int(np.median(cnt))} {max(cnt)} | kata lain {len(pools['other'])} "
          f"| celah {len(pools['gap'])} | kelas tanpa klip: {kosong or '-'}")
    # pemilihan jangkar: kata kunci seimbang antar kelas, sisanya kata lain / senyap
    by_class = {}
    for k, s, e in pools["kw"]:
        c = next(c for c, s2, e2 in EV[k][0] if s2 == s and e2 == e); by_class.setdefault(c, []).append((k, s, e))
    kc = list(by_class)

    def draw(rng):
        r = rng.random()
        if r < 0.62 and kc:
            k, s, e = rng.choice(by_class[rng.choice(kc)]); return k, "kw", s, e
        if r < 0.90 and pools["other"]:
            k, s, e = rng.choice(pools["other"]); return k, "other", s, e
        if r < 0.97 and pools["gap"]:
            k, s, e = rng.choice(pools["gap"]); return k, "gap", s, e
        return rng.choice(tr[:max(len(tr), 1)]), "noise", 0.0, 0.0

    model = BCResNet(C, tau=a.tau).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    rng = random.Random(fi)
    best, best_f1, best_ep = copy.deepcopy(model.state_dict()), -1.0, 0
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for _ in range(steps_per_epoch):
            xs, ys, ms = [], [], []
            for _ in range(a.batch):
                k, kind, s, e = draw(rng)
                x, lab, mask = make_sample(k, kind, s, e, rng); xs.append(x); ys.append(lab); ms.append(mask)
            x = torch.from_numpy(np.stack(xs)).to(dev)
            y = torch.from_numpy(np.stack(ys)).to(dev); m = torch.from_numpy(np.stack(ms)).to(dev)
            l = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw, reduction="none")
            loss = (l * m).sum(1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sch.step(); tot += float(loss.detach())
        if ep % 5 == 0 or ep == epochs:
            model.eval(); mt = val_score(model, va, dev)
            mark = ""
            if mt["item_F1"] > best_f1:
                best_f1, best_ep, best = mt["item_F1"], ep, copy.deepcopy(model.state_dict()); mark = "  <- terbaik"
            print(f"  epoch {ep:3d}: loss {tot / steps_per_epoch:.3f} | val item F1 {mt['item_F1']:.3f} "
                  f"(P {mt['item_P']:.2f} R {mt['item_R']:.2f}) intent {mt['intent_acc']:.0%} | {time.time() - t_start:.0f} dtk{mark}")
    model.load_state_dict(best); model.eval().to(dev)

    # sesuaikan ambang barang, ambang perintah, dan jumlah jendela berurutan pada val saja
    bp, bs, cache = (0.5, 0.5, 2), (-1.0, -1.0), {}
    for ti in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for tc in (0.3, 0.5, 0.7, 0.9):
            for run in (1, 2, 3):
                mt = val_score(model, va, dev, (ti, tc, run), cache)
                if (mt["item_F1"], mt["intent_acc"]) > bs: bs, bp = (mt["item_F1"], mt["intent_acc"]), (ti, tc, run)
    print(f"  dipakai: epoch {best_ep}, ambang barang {bp[0]}, perintah {bp[1]}, jendela berurutan {bp[2]} (val item F1 {bs[0]:.3f})")

    model.to(a.eval_device).eval()
    if te: run_sentence(model, te[0], a.eval_device, bp)                       # pemanasan
    for k in te:
        intent, its, best_p, ms = run_sentence(model, k, a.eval_device, bp)
        d = meta[k]
        wo.writerow({"model": f"M3_{a.tag}", "spk": d["spk"], "id": d["id"], "jenis": d["jenis"], "ref": d["transcript"],
                     "gold_items": "|".join(d["items"]), "hyp": " ".join(f"{n}({p:.2f})" for n, p in best_p.items()),
                     "pred_intent": intent, "pred_items": "|".join(its), "latency_ms": round(ms, 1)})
    fo.flush(); summary.append((fi, best_ep, bp, round(bs[0], 3)))
    Path("models").mkdir(exist_ok=True)
    torch.save({"classes": classes, "state": best, "tau": a.tau, "params": bp}, f"models/m3_{a.tag}_f{fi}.pt")
    del model, opt, sch
    if dev == "cuda": torch.cuda.empty_cache()

fo.close()
print("\nRingkasan (putaran, epoch, (ambang barang, ambang perintah, jendela), val item F1):")
for r in summary: print("  ", r)
print("\nUJI CEPAT SELESAI. Hasil di hasil_uji/ (jangan dipakai untuk laporan)." if a.uji_cepat else
      f"\nTersimpan: {outdir}/m3_{a.tag}{akhiran}.csv  ->  python evaluasi.py hasil/m3_{a.tag}.csv --uji80")
