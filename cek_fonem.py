"""Cek apakah himpunan fonem g2p (dipakai nlu_phoneme) cocok dengan tokens.txt model M1."""
from pathlib import Path
from nlu_phoneme import PATTERNS, _SYM, to_phonemes
from data import load_script

tok = Path(__file__).resolve().parent / "models" / "zipformer_id" / "tokens.txt"
vocab = {line.split()[0] for line in tok.read_text(encoding="utf-8").splitlines() if line.strip()}
print(f"{len(vocab)} token model:", " ".join(sorted(vocab)))

dipakai = set(_SYM)                                        # fonem dari lexicon + kata perintah
for r in load_script().values():
    dipakai |= set(to_phonemes(r["transcript"]))
hilang = sorted(dipakai - vocab)
print("\nFonem g2p yang TIDAK ada di model:", hilang or "tidak ada (cocok)")
print("Fonem model yang tidak dipakai g2p:", sorted(vocab - dipakai))
