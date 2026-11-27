"""M2: wav2vec2 (XLS-R, sudah di-fine-tune untuk bahasa Indonesia) + CTC, di-fine-tune dengan data Anda.
Pembagian: splits.py (uji = pembicara baru DAN kalimat baru). Jalankan di GPU (Google Colab).

  python latih_m2.py --uji-cepat     <- WAJIB dulu (sekitar 3 menit): memeriksa seluruh jalur kode
  python latih_m2.py                 <- 5 putaran penuh

Keluaran (80 rekaman uji, satu kali per pasangan pembicara-kalimat):
  hasil/m2_zeroshot.csv   model asli TANPA dilatih (untuk melihat manfaat data Anda)
  hasil/m2_finetune.csv   model setelah fine-tune (epoch terbaik menurut val)
Latensi diukur per rekaman (batch 1) di --eval-device (bawaan cpu)."""
import argparse, random, time
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="indonesian-nlp/wav2vec2-large-xlsr-indonesian")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--folds", nargs="*", type=int, help="mis. --folds 0 1 (bawaan: semua)")
ap.add_argument("--eval-device", default="cpu")
ap.add_argument("--audio", default="processed")
ap.add_argument("--tag", default="xlsr_id")
ap.add_argument("--uji-cepat", action="store_true")
ap.add_argument("--izinkan-cpu", action="store_true", help="paksa jalan di CPU (sangat lambat)")
a = ap.parse_args()

try:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    raise SystemExit("Butuh torch dan transformers: pip install torch transformers")

from data import load_dataset
from splits import load_folds
from hasil_io import open_writer
from metrics import edit_distance
from nlu import parse, normalize
from augment import add_noise, speed_perturb

random.seed(0); np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = dev == "cuda"
print("torch", torch.__version__, "| perangkat latih:", dev, "| perangkat ukur latensi:", a.eval_device)
if dev == "cpu" and not (a.uji_cepat or a.izinkan_cpu):
    raise SystemExit("GPU tidak terdeteksi, dihentikan agar tidak membuang waktu berjam-jam.\n"
                     "Cek '!nvidia-smi', sambungkan ulang runtime ke GPU, lalu jalankan lagi.\n"
                     "(Paksa CPU: tambahkan --izinkan-cpu)")

# ---- data -------------------------------------------------------------------------------------
meta = {(d["spk"], d["id"]): d for d in load_dataset(a.audio)}
wav = {}
for k, d in meta.items():
    y, sr = sf.read(d["path"], dtype="float32")
    assert sr == 16000, f"{d['path']} bukan 16 kHz"
    wav[k] = y if y.ndim == 1 else y.mean(axis=1)

proc = Wav2Vec2Processor.from_pretrained(a.model)
label_ids = {k: proc.tokenizer(normalize(meta[k]["transcript"])).input_ids for k in meta}
norm = lambda y: ((y - y.mean()) / (y.std() + 1e-7)).astype(np.float32)


def augment(y, rng):
    if rng.random() < 0.5:
        y = speed_perturb(y, rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        n = np.random.default_rng(rng.randrange(10 ** 9)).standard_normal(len(y)).astype(np.float32)
        y = add_noise(y, n, rng.uniform(15, 35))
    return y.astype(np.float32)


def collate(keys, train, rng, use_mask):
    xs = [norm(augment(wav[k], rng) if train else wav[k]) for k in keys]
    n = max(len(x) for x in xs)
    X = torch.zeros(len(xs), n); M = torch.zeros(len(xs), n, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = torch.from_numpy(x); M[i, :len(x)] = 1
    L = max(len(label_ids[k]) for k in keys)
    Y = torch.full((len(keys), L), -100, dtype=torch.long)
    for i, k in enumerate(keys):
        Y[i, :len(label_ids[k])] = torch.tensor(label_ids[k])
    return X, (M if use_mask else None), Y


def predict(model, keys, device, timing=False):
    model.eval(); model.to(device)
    hyps, lat = [], []
    with torch.no_grad():
        if timing and keys:                                   # pemanasan, tidak dihitung
            model(torch.from_numpy(norm(wav[keys[0]])).unsqueeze(0).to(device))
        for k in keys:
            x = torch.from_numpy(norm(wav[k])).unsqueeze(0).to(device)
            if device == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            ids = model(x).logits.argmax(-1)
            hyp = proc.batch_decode(ids)[0]
            if device == "cuda": torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1000); hyps.append(hyp)
    return hyps, lat


def cer(model, keys, device):
    hyps, _ = predict(model, keys, device)
    err = sum(edit_distance(list(normalize(meta[k]["transcript"])), list(h)) for k, h in zip(keys, hyps))
    return err / max(sum(len(normalize(meta[k]["transcript"])) for k in keys), 1)


def write_rows(w, name, keys, hyps, lat):
    for k, h, ms in zip(keys, hyps, lat):
        d, p = meta[k], parse(h)
        w.writerow({"model": name, "spk": d["spk"], "id": d["id"], "jenis": d["jenis"], "ref": d["transcript"],
                    "gold_items": "|".join(d["items"]), "hyp": h, "pred_intent": p["intent"],
                    "pred_items": "|".join(p["items"]), "latency_ms": round(ms, 1)})


# ---- putaran ----------------------------------------------------------------------------------
folds = load_folds()
sel = a.folds if a.folds is not None else ([0] if a.uji_cepat else range(len(folds)))
outdir = "hasil_uji" if a.uji_cepat else "hasil"
akhiran = "_f" + "_".join(map(str, sel)) if (a.folds is not None and not a.uji_cepat) else ""
f0, w0 = open_writer(f"{outdir}/m2_zeroshot{akhiran}.csv")
f1, w1 = open_writer(f"{outdir}/m2_finetune{akhiran}.csv")
epochs = 2 if a.uji_cepat else a.epochs
summary = []

for fi in sel:
    fold = folds[fi]
    tr = [k for k in fold["train"] if k in meta]
    va = [k for k in fold["val"] if k in meta]
    te = [k for k in fold["test"] if k in meta]
    if a.uji_cepat:
        tr, va, te = tr[:8], va[:4], te[:4]
    print(f"\n=== putaran {fi}: latih {len(tr)}, val {len(va)}, uji {len(te)} ===")
    model = Wav2Vec2ForCTC.from_pretrained(a.model, ctc_loss_reduction="mean", ctc_zero_infinity=True,
                                           mask_time_prob=0.05, layerdrop=0.0)
    (model.freeze_feature_encoder if hasattr(model, "freeze_feature_encoder")
     else model.freeze_feature_extractor)()
    use_mask = model.config.feat_extract_norm == "layer"
    model.to(dev)

    cer0 = cer(model, va, dev)                                 # epoch 0 = tanpa dilatih
    hy, lt = predict(model, te, a.eval_device, timing=True)
    write_rows(w0, f"M2_{a.tag}_zeroshot", te, hy, lt); f0.flush()
    model.to(dev)
    print(f"  tanpa latih: val CER {cer0:.3f}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = epochs * ((len(tr) + a.batch - 1) // a.batch)
    warm = max(1, steps // 10)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(steps - warm, 1))))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = random.Random(fi)
    best, best_ep, best_cer = None, 0, cer0

    for ep in range(1, epochs + 1):
        model.train(); order = tr[:]; rng.shuffle(order); tot = 0.0
        for i in range(0, len(order), a.batch):
            X, M, Y = collate(order[i:i + a.batch], True, rng, use_mask)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_values=X.to(dev), attention_mask=None if M is None else M.to(dev),
                             labels=Y.to(dev)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sch.step(); tot += float(loss.detach())
        c = cer(model, va, dev)
        mark = ""
        if c < best_cer:
            best_cer, best_ep = c, ep
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- terbaik"
        print(f"  epoch {ep:2d}: loss {tot / max(len(order) / a.batch, 1):.3f}  val CER {c:.3f}{mark}")

    if best is not None:
        model.load_state_dict(best)
    print(f"  dipakai: epoch {best_ep} (val CER {best_cer:.3f}; tanpa latih {cer0:.3f})")
    hy, lt = predict(model, te, a.eval_device, timing=True)
    write_rows(w1, f"M2_{a.tag}", te, hy, lt); f1.flush()
    summary.append((fi, cer0, best_cer, best_ep))
    del model, opt, sch, scaler, params, best     # sch memegang opt, opt memegang bobot
    import gc; gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()

f0.close(); f1.close()
print("\nRingkasan val CER (tanpa latih -> dipakai):")
for fi, c0, c1, ep in summary:
    print(f"  putaran {fi}: {c0:.3f} -> {c1:.3f} (epoch {ep})")
if a.uji_cepat:
    print("\nUJI CEPAT SELESAI tanpa error. Hasil ada di hasil_uji/ (jangan dipakai untuk laporan).")
else:
    print(f"\nTersimpan: {outdir}/m2_zeroshot{akhiran}.csv dan {outdir}/m2_finetune{akhiran}.csv")
    print("Bandingkan: python evaluasi.py hasil/m2_zeroshot.csv hasil/m2_finetune.csv --uji80")
