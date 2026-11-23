"""Pemuat dataset + pembagian leave-one-speaker-out. Dipakai semua skrip model."""
import csv, re
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_script():
    with open(BASE / "script.csv", encoding="utf-8") as f:
        return {int(r["id"]): {"jenis": r["jenis"], "transcript": r["transcript"],
                               "items": [x for x in r["items"].split("|") if x]}
                for r in csv.DictReader(f)}


def _wav_by_id(folder):
    """{nomor: path}; nomor diambil dari angka di nama file (1.wav / 001.wav sama saja)."""
    out = {}
    for p in Path(folder).glob("*.wav"):
        d = re.sub(r"\D", "", p.stem)
        if d:
            out[int(d)] = p
    return out


def load_dataset(audio_dir="processed", speakers=None, limit=None):
    """Daftar dict: spk, id, jenis, transcript, items, path. limit = jumlah kalimat per pembicara."""
    script, root = load_script(), BASE / audio_dir
    spks = sorted(d.name for d in root.iterdir() if d.is_dir())
    if speakers:
        spks = [s for s in spks if s in speakers]
    data = []
    for spk in spks:
        files = _wav_by_id(root / spk)
        for sid in sorted(script)[:limit]:
            if sid in files:
                data.append({"spk": spk, "id": sid, **script[sid], "path": str(files[sid])})
    return data


def loso_folds(speakers):
    """5 putaran: 1 pembicara test, pembicara berikutnya val, sisanya train."""
    n, folds = len(speakers), []
    for i, test in enumerate(speakers):
        val = speakers[(i + 1) % n]
        folds.append({"test": [test], "val": [val],
                      "train": [s for s in speakers if s not in (test, val)]})
    return folds
