"""Fungsi pendukung M3 (hanya numpy; mudah diuji): penjajaran CTC, lokasi kata, fitur mel, dekode."""
import numpy as np
from nlu import normalize, _ALIAS, INTENT_HAPUS, INTENT_CENTANG

SR = 16000
CAT_UNK, CAT_DIAM = "unknown", "diam"


# ---------------------------------------------------------------- penjajaran CTC (Viterbi) ----------
def ctc_align(logp, targets, blank=0):
    """Penjajaran paksa CTC. logp [T,V] log-probabilitas, targets daftar id (tanpa blank).
    Mengembalikan [(frame_awal, frame_akhir_eksklusif)] satu per target, atau None bila mustahil."""
    T, L = logp.shape[0], len(targets)
    S = 2 * L + 1
    lab = np.full(S, blank, dtype=int); lab[1::2] = targets
    if T < L + sum(1 for i in range(1, L) if targets[i] == targets[i - 1]):
        return None
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, blank]
    if S > 1: dp[0, 1] = logp[0, lab[1]]
    can_skip = np.zeros(S, dtype=bool)
    for s in range(3, S, 2):
        can_skip[s] = lab[s] != lab[s - 2]
    for t in range(1, T):
        stay = dp[t - 1]
        prev = np.concatenate(([NEG], dp[t - 1, :-1]))
        skip = np.concatenate(([NEG, NEG], dp[t - 1, :-2])); skip = np.where(can_skip, skip, NEG)
        cand = np.stack([stay, prev, skip])
        best = cand.argmax(0)
        dp[t] = cand[best, np.arange(S)] + logp[t, lab]
        bp[t] = best
    s = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s]) if t > 0 else 0
    spans = [[None, None] for _ in range(L)]
    for t, st in enumerate(path):
        if st % 2 == 1:
            i = st // 2
            if spans[i][0] is None: spans[i][0] = t
            spans[i][1] = t + 1
    return [tuple(x) if x[0] is not None else None for x in spans]


def word_spans(spans, word_lens):
    """Gabungkan rentang per-huruf menjadi per-kata (frame). word_lens = jumlah huruf tiap kata."""
    out, k = [], 0
    for n in word_lens:
        seg = [s for s in spans[k:k + n] if s is not None]
        out.append((min(s[0] for s in seg), max(s[1] for s in seg)) if seg else None)
        k += n
    return out


# ---------------------------------------------------------------- lokasi barang & kata perintah -----
_CMD = {}
for _kw in INTENT_HAPUS: _CMD[_kw] = "hapus"
for _kw in INTENT_CENTANG: _CMD[_kw] = "centang"
_MAXN = max(len(a.split()) for a in list(_ALIAS) + list(_CMD))


def locate(words, gold_items):
    """Kembalikan [(awal_kata, jumlah_kata, kelas)] untuk barang gold dan kata perintah.
    Kata lain tidak dikembalikan (dianggap 'unknown')."""
    used, out = set(), []
    for n in range(_MAXN, 0, -1):
        for i in range(len(words) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            ph = " ".join(words[i:i + n])
            if ph in _CMD:
                used.update(range(i, i + n)); out.append((i, n, _CMD[ph]))
            elif ph in _ALIAS and _ALIAS[ph] in gold_items:
                canon = _ALIAS[ph]
                if ph != canon and ph.startswith(canon + " "):     # mis. "susu kotak" -> hanya "susu"
                    n2 = len(canon.split())
                    used.update(range(i, i + n2)); out.append((i, n2, canon))
                else:
                    used.update(range(i, i + n)); out.append((i, n, canon))
    return sorted(out)


# ---------------------------------------------------------------- fitur log-mel ---------------------
def mel_filterbank(n_mels=40, n_fft=512, sr=SR, fmin=20.0, fmax=7600.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c): fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r): fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


# ---------------------------------------------------------------- dekode kehadiran ---------------------
def present(probs, thr, min_run=2):
    """probs [N,C]; kelas dianggap hadir bila probabilitasnya >= thr pada min_run jendela BERURUTAN.
    (Kata yang utuh berada di dalam jendela pada sekitar 0,6-0,8 detik, jadi plato lebar.)"""
    N, C = probs.shape
    thr = np.broadcast_to(np.asarray(thr, dtype=float), (C,))
    ok = probs >= thr
    out = np.zeros(C, dtype=bool)
    for c in range(C):
        run = 0
        for n in range(N):
            run = run + 1 if ok[n, c] else 0
            if run >= min(min_run, N):
                out[c] = True; break
    return out


def decode(probs, class_names, thr_item=0.5, thr_cmd=0.5, min_run=2):
    """Kembalikan (intent, daftar barang, {kelas: prob maksimum} untuk kelas hadir)."""
    thr = np.array([thr_cmd if n in ("hapus", "centang") else thr_item for n in class_names])
    hadir = present(probs, thr, min_run)
    items = [n for n, h in zip(class_names, hadir) if h and n not in ("hapus", "centang")]
    cmds = {n for n, h in zip(class_names, hadir) if h and n in ("hapus", "centang")}
    intent = "hapus" if "hapus" in cmds else "centang" if "centang" in cmds else ("tambah" if items else "none")
    best = {n: float(probs[:, i].max()) for i, (n, h) in enumerate(zip(class_names, hadir)) if h}
    return intent, items, best
