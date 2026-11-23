"""Teks -> (intent, daftar item). Tanpa model berat; dipakai setelah ASR.
Hanya butuh Python standar."""
import re, time
from difflib import SequenceMatcher

# kanonik -> alias (sebutan lain / slang / salah ucap ASR yang umum)
LEXICON = {
    "susu": ["susu kotak"], "telur": ["telor"], "bayam": ["bayem"],
    "cabai": ["cabe", "lombok", "cabai merah", "cabe merah"],
    "bawang merah": ["bawang abang"], "bawang putih": [], "bawang bombay": ["bombay", "bawang bombai"],
    "minyak goreng": ["migor"], "gula pasir": ["gulpas"], "gula merah": [],
    "mie instan": ["mi instan", "indomie"], "timun": ["mentimun"], "kol": ["kubis"],
    "roti": [], "tomat": [], "wortel": [], "tempe": [], "tahu": [], "kentang": [],
    "ayam": [], "kangkung": [], "jeruk": [], "daging sapi": [], "ikan": [], "udang": [],
    "beras": [], "garam": [], "kecap manis": [], "kecap asin": [], "santan": [],
    "tepung beras": [], "tepung terigu": ["terigu"], "pisang": [], "semangka": [],
    "apel": [], "teh celup": [], "sabun cuci piring": [], "saus sambal": [],
    "mentega": [], "air mineral": [], "kopi": [], "jahe": [], "kunyit": [],
    "sawi": [], "brokoli": [], "buncis": [], "sirup": [], "tisu": [],
    "jus jeruk": [], "telur puyuh": [],
}
INTENT_HAPUS = ["hapus", "batal", "nggak jadi", "gak jadi", "ga jadi", "enggak jadi",
                "gak usah", "nggak usah", "ga usah", "enggak usah", "jangan jadi"]
INTENT_CENTANG = ["sudah", "udah", "udh", "centang", "ceklis", "checklist", "dapat", "dapet",
                  "ambil", "diambil", "keranjang", "troli", "trolley", "beres", "selesai"]

_CMD_WORDS = {w for kw in INTENT_HAPUS + INTENT_CENTANG for w in kw.split()}
_ALIAS = {}                       # alias -> kanonik
for canon, al in LEXICON.items():
    _ALIAS[canon] = canon
    for a in al: _ALIAS[a] = canon
_MAXN = max(len(a.split()) for a in _ALIAS)
_BYLEN = sorted(_ALIAS, key=lambda a: -len(a))


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _intent(norm):
    padded = f" {norm} "
    for kw in INTENT_HAPUS:
        if f" {kw} " in padded: return "hapus"
    for kw in INTENT_CENTANG:
        if f" {kw} " in padded: return "centang"
    return "tambah"


def extract_items(norm, fuzzy=0.85):
    toks, used, found = norm.split(), set(), []
    # 1) exact, frasa terpanjang dulu
    for n in range(_MAXN, 0, -1):
        for i in range(len(toks) - n + 1):
            if any(j in used for j in range(i, i + n)): continue
            ph = " ".join(toks[i:i + n])
            if ph in _ALIAS:
                used.update(range(i, i + n)); found.append((i, _ALIAS[ph]))
    # 2) fuzzy untuk sisa token (tahan salah ucap/ASR); hanya frasa >=4 huruf
    if fuzzy:
        for n in range(_MAXN, 0, -1):
            for i in range(len(toks) - n + 1):
                if any(j in used for j in range(i, i + n)): continue
                if any(t in _CMD_WORDS for t in toks[i:i + n]): continue
                ph = " ".join(toks[i:i + n])
                if len(ph) < 4: continue
                best = max(_BYLEN, key=lambda a: SequenceMatcher(None, ph, a).ratio())
                if len(best) >= 4 and SequenceMatcher(None, ph, best).ratio() >= fuzzy:
                    used.update(range(i, i + n)); found.append((i, _ALIAS[best]))
    seen, out = set(), []
    for _, c in sorted(found):
        if c not in seen: seen.add(c); out.append(c)
    return out


def parse(text, fuzzy=0.85):
    norm = normalize(text)
    items = extract_items(norm, fuzzy)
    it = _intent(norm)       # hapus / centang / tambah (bawaan)
    intent = it if it in ("hapus", "centang") else ("tambah" if items else "none")
    return {"intent": intent, "items": items}


if __name__ == "__main__":
    for s in ["Tolong catat susu, bayem, telor, dan roti.", "Migor sudah diambil", "Batal beli cabe", "hmm apa lagi ya"]:
        t = time.perf_counter(); r = parse(s)
        print(f"{s!r} -> {r}  ({(time.perf_counter()-t)*1000:.2f} ms)")
