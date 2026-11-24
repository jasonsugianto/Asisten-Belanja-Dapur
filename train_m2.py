"""M2: latih wav2vec2 + BiLSTM + CTC (fonem) dengan skema leave-one-speaker-out.

Menghasilkan satu checkpoint per fold di models/m2/fold_<test-spk>.pt,
ditambah vocab.json bersama. Dipakai oleh run_m2_wav2vec2.py untuk inferensi.

Contoh:
  python train_m2.py                        # 5 fold LOSO, default 15 epoch
  python train_m2.py --epochs 20 --augment  # + augmentasi noise/speed
  python train_m2.py --folds spk05          # latih satu fold saja (uji cepat)

Butuh: torch, transformers, soundfile, numpy, g2p_id_py
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from data import load_dataset, loso_folds, BASE
from m2_model import Wav2Vec2BiLSTMCTC, build_vocab, transcript_to_ids
from augment import add_noise, speed_perturb

SR = 16000
OUT_DIR = BASE / "models" / "m2"


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} bukan 16 kHz; jalankan konversi.py"
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


class SpeechDataset(Dataset):
    def __init__(self, items, vocab, augment=False, noise_pool=None, seed=0):
        self.items, self.vocab = items, vocab
        self.augment, self.noise_pool = augment, noise_pool or []
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        y = load_audio(it["path"])
        if self.augment:
            if self.rng.random() < 0.5:
                y = speed_perturb(y, float(self.rng.uniform(0.9, 1.1))).astype("float32")
            if self.noise_pool and self.rng.random() < 0.5:
                noise = self.noise_pool[self.rng.integers(len(self.noise_pool))]
                y = add_noise(y, noise, float(self.rng.uniform(5, 20)), self.rng).astype("float32")
        labels = transcript_to_ids(it["transcript"], self.vocab)
        return torch.from_numpy(np.ascontiguousarray(y)), torch.tensor(labels, dtype=torch.long)


def collate(batch):
    wavs, labels = zip(*batch)
    in_lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    lab_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    maxlen = int(in_lens.max())
    padded = torch.zeros(len(wavs), maxlen, dtype=torch.float32)
    mask = torch.zeros(len(wavs), maxlen, dtype=torch.long)
    for i, w in enumerate(wavs):
        padded[i, : len(w)] = w
        mask[i, : len(w)] = 1
    labels_cat = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
    return padded, mask, in_lens, labels_cat, lab_lens


def load_noise_pool(noise_dir):
    pool = []
    if not noise_dir:
        return pool
    for p in Path(noise_dir).glob("**/*.wav"):
        n, sr = sf.read(p, dtype="float32")
        if sr == SR:
            pool.append(n.mean(axis=1) if n.ndim > 1 else n)
    print(f"  noise pool: {len(pool)} berkas dari {noise_dir}")
    return pool


def train_fold(fold, vocab, args, device, noise_pool):
    train_items = load_dataset(args.audio, fold["train"], args.limit)
    val_items = load_dataset(args.audio, fold["val"], args.limit)
    ds = SpeechDataset(train_items, vocab, args.augment, noise_pool)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    num_workers=args.workers)

    model = Wav2Vec2BiLSTMCTC(len(vocab), base_model=args.base).to(device)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for padded, mask, in_lens, labels, lab_lens in dl:
            padded, mask = padded.to(device), mask.to(device)
            labels, lab_lens = labels.to(device), lab_lens.to(device)
            log_probs = model(padded, attention_mask=mask)          # (T', B, V)
            out_lens = model.feat_lengths(in_lens.to(device))
            loss = ctc(log_probs, labels, out_lens, lab_lens)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"    epoch {ep:02d}/{args.epochs}  ctc_loss {tot / max(nb, 1):.3f}  "
              f"(train {len(train_items)}, val {len(val_items)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUT_DIR / f"fold_{fold['test'][0]}.pt"
    torch.save({"state_dict": model.state_dict(), "base": args.base,
                "test_spk": fold["test"], "val_spk": fold["val"]}, ckpt)
    print(f"    tersimpan: {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/wav2vec2-base")
    ap.add_argument("--audio", default="processed")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--noise-dir", default=None, help="folder wav noise 16 kHz untuk augmentasi")
    ap.add_argument("--folds", nargs="*", help="batasi ke test-speaker tertentu, mis. spk05")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    vocab = build_vocab()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"vocab fonem: {len(vocab)} simbol (termasuk blank) -> {OUT_DIR / 'vocab.json'}")

    speakers = sorted({d["spk"] for d in load_dataset(args.audio)})
    folds = loso_folds(speakers)
    if args.folds:
        folds = [f for f in folds if f["test"][0] in args.folds]
    noise_pool = load_noise_pool(args.noise_dir) if args.augment else None

    t0 = time.perf_counter()
    for f in folds:
        print(f"=== fold: test={f['test']} val={f['val']} train={f['train']} ===")
        train_fold(f, vocab, args, device, noise_pool)
    print(f"Selesai {len(folds)} fold dalam {time.perf_counter() - t0:.0f} s. "
          f"Lanjut: python run_m2_wav2vec2.py")


if __name__ == "__main__":
    main()
