"""M1 fine-tune: Zipformer streaming fonem (bookbot, resep icefall pruned_transducer_stateless7_streaming)
dilatih lanjut dengan data Anda. Pembagian: splits.json (sama dengan M2/M3: uji = pembicara baru DAN kalimat baru).
Jalankan di GPU (Colab). Langkah lengkap: PANDUAN_FINETUNE_M1.md

  python latih_m1.py --resep $RESEP --cek          <- 1. cek arsitektur cocok dengan checkpoint (tanpa melatih)
  python latih_m1.py --resep $RESEP --uji-cepat    <- 2. jalur kode lengkap, 5 epoch, data kecil
  python latih_m1.py --resep $RESEP                <- 3. 5 putaran penuh

$RESEP = <icefall>/egs/librispeech/ASR/pruned_transducer_stateless7_streaming
Keluaran: exp_m1/fold{i}/epoch-{n}.pt (epoch terbaik menurut loss val), exp_m1/fold{i}/info.json,
          exp_m1/tokens_ekspor.txt (untuk export-onnx.py). Lanjutkan dengan ekspor_m1.py."""
import argparse, json, math, random, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--resep", required=True, help="folder resep pruned_transducer_stateless7_streaming di icefall")
ap.add_argument("--pretrained", default="models/m1_pytorch/exp/pretrained.pt")
ap.add_argument("--tokens", default="models/zipformer_id/tokens.txt")
ap.add_argument("--model-args", default="", help="argumen arsitektur bila checkpoint tidak memakai bawaan resep, "
                                                  "mis. \"--encoder-dims 256,256,256,256,256\"")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.004, help="base lr ScaledAdam (pelatihan awal resep: 0.05)")
ap.add_argument("--batch-count", type=int, default=100000, help="anggap model sudah lewat masa warm-up")
ap.add_argument("--folds", nargs="*", type=int)
ap.add_argument("--audio", default="processed")
ap.add_argument("--exp-dir", default="exp_m1")
ap.add_argument("--fp16", type=int, default=1)
ap.add_argument("--uji-cepat", action="store_true")
ap.add_argument("--cek", action="store_true")
ap.add_argument("--izinkan-cpu", action="store_true")
a = ap.parse_args()

try:
    import torch, k2
    from lhotse import Fbank, FbankConfig
    from g2p_id import G2p
except ImportError as e:
    raise SystemExit(f"Paket belum terpasang ({e}). Lihat bagian instalasi di PANDUAN_FINETUNE_M1.md")

resep = Path(a.resep).resolve()
if not (resep / "train.py").exists():
    raise SystemExit(f"train.py tidak ada di {resep}")
sys.path.insert(0, str(resep.parents[3]))                     # akar icefall
sys.path.insert(0, str(resep))
import train as R                                             # noqa: E402  (resep icefall)
from optim import ScaledAdam, Eden                            # noqa: E402

from data import load_dataset
from splits import load_folds
from augment import add_noise, speed_perturb

random.seed(0); np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = bool(a.fp16) and dev == "cuda"
print("torch", torch.__version__, "| k2 terpasang | perangkat:", dev)
if dev == "cpu" and not (a.cek or a.uji_cepat or a.izinkan_cpu):
    raise SystemExit("GPU tidak terdeteksi. Sambungkan runtime ke GPU (paksa CPU: --izinkan-cpu).")

# ---- token fonem & target --------------------------------------------------------------------
TOK = {}
for line in Path(a.tokens).read_text(encoding="utf-8").splitlines():
    if line.strip():
        s, i = line.split(); TOK[s] = int(i)
n_tok = max(i for s, i in TOK.items() if not re.fullmatch(r"#\d+", s)) + 1      # tanpa simbol #0
g2p = G2p()


def target_phonemes(text):
    """Teks -> fonem dengan '|' di antara kata (format yang sama dengan keluaran model M1)."""
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    out = []
    for i, w in enumerate(g2p(" ".join(words))):
        if i: out.append("|")
        out.extend(w)
    return out


meta = {(d["spk"], d["id"]): d for d in load_dataset(a.audio)}
labels, asing = {}, {}
for k, d in meta.items():
    phs = target_phonemes(d["transcript"])
    bad = [p for p in phs if p not in TOK]
    for p in bad: asing[p] = asing.get(p, 0) + 1
    labels[k] = [TOK[p] for p in phs if p in TOK]
if asing:
    raise SystemExit(f"Fonem g2p tidak ada di {a.tokens}: {asing}. Jalankan cek_fonem.py.")
print(f"{len(meta)} rekaman, {n_tok} token (blank=0). Contoh target {meta[next(iter(meta))]['transcript']!r}:",
      " ".join(target_phonemes(meta[next(iter(meta))]["transcript"])))

wav = {}
for k, d in meta.items():
    y, sr = sf.read(d["path"], dtype="float32")
    assert sr == 16000, f"{d['path']} bukan 16 kHz"
    wav[k] = y if y.ndim == 1 else y.mean(axis=1)

# ---- fitur: fbank 80 dimensi, sama dengan pelatihan awal (lhotse) dan sherpa-onnx (dither 0) ----
try:
    fbank = Fbank(FbankConfig(num_filters=80))
except TypeError:
    fbank = Fbank(FbankConfig(num_mel_bins=80))
LOG_EPS = math.log(1e-10)                                     # nilai padding fitur di lhotse


def spec_augment(f, rng):
    f = f.copy(); T, D = f.shape; isi = f.mean()
    for _ in range(2):                                        # 2 mask frekuensi, lebar <= 15
        w = rng.randint(0, 15); s = rng.randint(0, D - w); f[:, s:s + w] = isi
    for _ in range(2):                                        # 2 mask waktu, lebar <= 10% panjang
        w = rng.randint(0, max(1, min(20, T // 10))); s = rng.randint(0, max(T - w, 0)); f[s:s + w] = isi
    return f


def batch_of(keys, train, rng):
    feats = []
    for k in keys:
        y = wav[k]
        if train:
            if rng.random() < 0.5: y = speed_perturb(y, rng.uniform(0.9, 1.1)).astype(np.float32)
            if rng.random() < 0.5:
                n = np.random.default_rng(rng.randrange(10 ** 9)).standard_normal(len(y)).astype(np.float32)
                y = add_noise(y, n, rng.uniform(15, 35)).astype(np.float32)
        y = y.astype(np.float32)
        try:
            f = fbank.extract(y.reshape(1, -1), 16000)
        except Exception:                                     # versi lhotse yang meminta array 1 dimensi
            f = fbank.extract(y, 16000)
        f = np.asarray(f, dtype=np.float32)
        feats.append(spec_augment(f, rng) if train else f)
    T = max(len(f) for f in feats)
    X = torch.full((len(feats), T, 80), LOG_EPS)
    for i, f in enumerate(feats): X[i, :len(f)] = torch.from_numpy(f)
    lens = torch.tensor([len(f) for f in feats], dtype=torch.int64)
    return X, lens, k2.RaggedTensor([labels[k] for k in keys])


# ---- model -------------------------------------------------------------------------------------
def load_state(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                         # torch lama tanpa weights_only
        ck = torch.load(path, map_location="cpu")
    except RuntimeError:                                      # TorchScript (cpu_jit.pt)
        ck = torch.jit.load(path, map_location="cpu")
    if isinstance(ck, dict):
        return ck.get("model", ck)
    return ck.state_dict()


if not Path(a.pretrained).exists():
    raise SystemExit(f"{a.pretrained} tidak ada. Jalankan unduh_m1_pytorch.py dan pakai path yang dicetaknya.")
PRE = load_state(a.pretrained)
vocab = PRE["decoder.embedding.weight"].shape[0]
if vocab != n_tok:
    print(f"PERINGATAN: checkpoint punya {vocab} token, tokens.txt {n_tok}. Pastikan tokens.txt dari model yang sama.")


def build_model():
    # parser lengkap resep (bukan hanya add_model_arguments): context_size dll. didefinisikan di get_parser()
    p = R.get_parser() if hasattr(R, "get_parser") else argparse.ArgumentParser()
    if not hasattr(R, "get_parser"): R.add_model_arguments(p)
    params = R.get_params(); params.update(vars(p.parse_args(a.model_args.split())))
    params.vocab_size, params.blank_id = vocab, 0
    model = R.get_transducer_model(params)
    try:
        miss, extra = model.load_state_dict(PRE, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"Ukuran bobot tidak cocok dengan arsitektur bawaan resep:\n{str(e)[:2000]}\n"
                         "-> checkpoint memakai argumen arsitektur lain; kirim pesan ini, lalu isi --model-args.")
    if miss:
        raise SystemExit(f"{len(miss)} bobot tidak ada di checkpoint, mis. {miss[:5]}. Checkpoint bukan model yang sama.")
    if extra:
        print(f"  (diabaikan: {len(extra)} kunci tambahan di checkpoint, mis. {extra[:3]})")
    for m in model.modules():                                 # balancer/bypass: perilaku pasca warm-up
        if hasattr(m, "batch_count"): m.batch_count = a.batch_count
    return model.to(dev), params


def loss_of(model, X, lens, Y):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        simple, pruned = model(x=X.to(dev), x_lens=lens.to(dev), y=Y.to(dev), prune_range=5, am_scale=0.0, lm_scale=0.25)
    return 0.5 * simple + pruned                              # skala resep setelah warm-up


def val_loss(model, keys):
    model.eval(); tot = fr = 0.0; rng = random.Random(0)
    with torch.no_grad():
        for i in range(0, len(keys), a.batch):
            X, lens, Y = batch_of(keys[i:i + a.batch], False, rng)
            tot += float(loss_of(model, X, lens, Y)); fr += float((lens // 4).sum())
    return tot / max(fr, 1)                                   # loss per frame keluaran encoder


if a.cek:
    model, params = build_model()
    print(f"Arsitektur cocok. {sum(p.numel() for p in model.parameters()) / 1e6:.1f} juta parameter, vocab {vocab}, "
          f"decode_chunk_len {params.get('decode_chunk_len', '?')}")
    keys = [k for k in load_folds()[0]["val"] if k in meta][:8]
    print(f"Loss val pretrained (8 rekaman, per frame): {val_loss(model, keys):.3f}  "
          "(wajar < 1; jauh lebih besar = fitur/token tidak cocok)")
    sys.exit(0)

# ---- putaran ------------------------------------------------------------------------------------
folds = load_folds()
sel = a.folds if a.folds is not None else ([0] if a.uji_cepat else range(len(folds)))
epochs = 5 if a.uji_cepat else a.epochs
root = Path(a.exp_dir + ("_uji" if a.uji_cepat else ""))
root.mkdir(parents=True, exist_ok=True)
(root / "tokens_ekspor.txt").write_text(                      # export-onnx.py mencari <blk> dan <unk>
    "".join(f"{ {'<eps>': '<blk>', '<UNK>': '<unk>'}.get(s, s)} {i}\n" for s, i in sorted(TOK.items(), key=lambda x: x[1])),
    encoding="utf-8")
ringkas = []

for fi in sel:
    fold = folds[fi]
    tr = [k for k in fold["train"] if k in meta]; va = [k for k in fold["val"] if k in meta]
    if a.uji_cepat: tr, va = tr[:16], va[:4]
    print(f"\n=== putaran {fi}: latih {len(tr)}, val {len(va)} | uji {fold['test'][0][0]} (tidak disentuh di sini) ===")
    model, params = build_model()
    v0 = val_loss(model, va); print(f"  tanpa latih: val loss {v0:.3f}")
    names = [[n for n, _ in model.named_parameters()]]
    try:
        opt = ScaledAdam(model.parameters(), lr=a.lr, clipping_scale=2.0, parameters_names=names)
    except TypeError:
        opt = ScaledAdam(model.parameters(), lr=a.lr, clipping_scale=2.0)
    sch = Eden(opt, lr_batches=100000, lr_epochs=100)          # praktis konstan setelah warm-up singkat
    # init_scale=1.0 seperti train.py icefall; bawaan 65536 membuat langkah awal dilewati (overflow fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1.0)
    rng, step = random.Random(fi), 0
    best, best_ep, best_v, t0 = None, 0, float("inf"), time.time()
    for ep in range(1, epochs + 1):
        sch.step_epoch(ep - 1)
        model.train(); order = tr[:]; rng.shuffle(order); tot = fr = 0.0; lewat = 0
        for i in range(0, len(order), a.batch):
            X, lens, Y = batch_of(order[i:i + a.batch], True, rng)
            loss = loss_of(model, X, lens, Y)
            opt.zero_grad(); scaler.scale(loss).backward(); sc = scaler.get_scale()
            scaler.step(opt); scaler.update()
            lewat += amp and scaler.get_scale() < sc               # skala turun = langkah dilewati (inf/nan)
            step += 1; sch.step_batch(step)
            tot += float(loss.detach()); fr += float((lens // 4).sum())
        v = val_loss(model, va); mark = ""
        if v < best_v:
            best_v, best_ep, mark = v, ep, "  <- terbaik"
            best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"  epoch {ep:2d}: loss latih {tot / max(fr, 1):.3f}  val {v:.3f}  ({time.time() - t0:.0f} dtk)"
              f"{f'  [{lewat} langkah dilewati]' if lewat else ''}{mark}")
    out = root / f"fold{fi}"; out.mkdir(exist_ok=True)
    torch.save({"model": best}, out / f"epoch-{best_ep}.pt")
    info = {"fold": fi, "epoch": best_ep, "val_loss": best_v, "val_loss_tanpa_latih": v0,
            "uji_pembicara": fold["test"][0][0], "model_args": a.model_args, "lr": a.lr, "epochs": epochs}
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"  tersimpan {out / f'epoch-{best_ep}.pt'} (val {v0:.3f} -> {best_v:.3f})")
    if best_v >= v0: print("  PERINGATAN: fine-tune tidak menurunkan loss val; coba --lr lebih kecil.")
    ringkas.append((fi, v0, best_v, best_ep))
    del model, opt, sch, scaler, best
    if dev == "cuda": torch.cuda.empty_cache()

print("\nRingkasan val loss (tanpa latih -> terbaik, epoch):")
for fi, v0, v1, ep in ringkas: print(f"  putaran {fi}: {v0:.3f} -> {v1:.3f} (epoch {ep})")
print("\nUJI CEPAT SELESAI tanpa error." if a.uji_cepat else f"\nLanjut: python ekspor_m1.py --resep $RESEP --exp-dir {root}")
