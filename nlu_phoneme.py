"""NLU untuk keluaran FONEM (model M1: Zipformer RNN-T ID menghasilkan fonem IPA, bukan kata).
Alur: fonem hasil ASR -> cocokkan ke fonem tiap nama barang/kata perintah (edit distance lokal)
      -> pilih kecocokan terbaik yang tidak tumpang tindih -> (intent, item).
Butuh: pip install g2p_id_py"""
import re, time
from functools import lru_cache
from g2p_id import G2p
from nlu import LEXICON, INTENT_HAPUS, INTENT_CENTANG

_g2p = G2p()
BUKAN_FONEM = {"|", "", "<blk>", "<unk>", "<sos/eos>", "▁"}   # model M1 memberi "|" antar kata


def bersih(phs):
    """Buang pembatas kata dan token khusus; sisakan fonem saja."""
    return [p for p in phs if p not in BUKAN_FONEM]


@lru_cache(maxsize=None)
def to_phonemes(text):
    """teks -> daftar fonem datar (batas kata dibuang, karena ASR fonem tidak menjamin spasi)."""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return tuple(p for word in _g2p(text) for p in word)

# fonem multi-karakter (tʃ, dʒ, ...) dipetakan ke 1 karakter agar edit distance per-fonem
_SYM = {}
def enc(phs):
    return "".join(_SYM.setdefault(p, chr(0x100 + len(_SYM))) for p in phs)

# Pasangan yang di data Anda tertukar secara sistematis (beda konvensi, bukan salah dengar):
#   k<->ʔ (mis. "masuk" -> m a s u ʔ), h<->ʔ (teh), e<->ə. Biayanya 0.3, bukan 1.
CHEAP_COST = 0.3
_CHEAP = set()
for a_, b_ in [("k", "ʔ"), ("h", "ʔ"), ("e", "ə"),            # terlihat di data Anda
                ("p", "b"), ("t", "d"), ("k", "ɡ"), ("tʃ", "dʒ")]:  # pasangan bersuara/tak bersuara
    ca, cb = enc([a_])[0], enc([b_])[0]
    _CHEAP |= {(ca, cb), (cb, ca)}

# pola: (teks_pola, jenis, nilai) ; jenis: item | hapus | centang
PATTERNS = []
for canon, aliases in LEXICON.items():
    for a in [canon] + aliases:
        PATTERNS.append((enc(to_phonemes(a)), "item", canon))
for kw in INTENT_HAPUS:   PATTERNS.append((enc(to_phonemes(kw)), "hapus", kw))
for kw in INTENT_CENTANG: PATTERNS.append((enc(to_phonemes(kw)), "centang", kw))

def max_edits(m, rate=0.22):
    """Anggaran kesalahan: pola pendek harus persis, makin panjang makin longgar.
    (Anggaran tambahan untuk substitusi murah sempat dicoba dan menambah salah-tambah pada teks bersih.)"""
    return int(rate * m)

def local_matches(pat, text, k):
    """Semua (jarak, awal, akhir) di mana pola cocok dengan substring text dengan <= k edit."""
    m, n = len(pat), len(text)
    prev = [(0, j) for j in range(n + 1)]            # (jarak, posisi awal)
    for i in range(1, m + 1):
        cur = [(i, 0)]
        for j in range(1, n + 1):
            a_, b_ = pat[i - 1], text[j - 1]
            c = 0 if a_ == b_ else (CHEAP_COST if (a_, b_) in _CHEAP else 1)
            sub = prev[j - 1]; sub = (sub[0] + c, sub[1])
            dele = (prev[j][0] + 1, prev[j][1])
            ins = (cur[j - 1][0] + 1, cur[j - 1][1])
            cur.append(min(sub, dele, ins))
        prev = cur
    return [(d, s, j) for j, (d, s) in enumerate(prev) if d <= k + 1e-9 and j > s]

def parse_phonemes(phs, rate=0.22, kw_rate=None):
    """kw_rate: toleransi khusus kata perintah (hapus/centang); None = sama dengan rate."""
    text = enc(bersih(phs))
    cands = []
    for pat, kind, val in PATTERNS:
        m = len(pat); k = max_edits(m, (kw_rate or rate) if kind != "item" else rate)
        for d, s, e in local_matches(pat, text, k):
            cands.append((-(m - d), d, s, e, kind, val))   # utamakan kecocokan yang menutup fonem terbanyak
    cands.sort()                                     # panjang terselaraskan terbesar dulu; seri -> jarak kecil
    used, items, hapus, centang = set(), [], False, False
    for _, _, s, e, kind, val in cands:
        if any(p in used for p in range(s, e)): continue
        used.update(range(s, e))
        if kind == "item":
            if (s, val) not in items: items.append((s, val))
        elif kind == "hapus": hapus = True
        else: centang = True
    seen, out = set(), []
    for _, v in sorted(items):
        if v not in seen: seen.add(v); out.append(v)
    intent = "hapus" if hapus else "centang" if centang else ("tambah" if out else "none")
    return {"intent": intent, "items": out}

if __name__ == "__main__":
    for s in ["Tolong catat susu, bayem, telor, dan roti.", "Migor sudah diambil", "Centang tomat", "Batal beli cabe"]:
        t = time.perf_counter(); r = parse_phonemes(to_phonemes(s))
        print(f"{s!r} -> {r} ({(time.perf_counter()-t)*1000:.1f} ms)")
