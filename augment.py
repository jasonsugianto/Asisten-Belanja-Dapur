import numpy as np

def add_noise(clean, noise, snr_db, rng=None):
    """Tambah noise pada SNR tertentu. noise: array 1D (mis. dari MUSAN/DEMAND/ESC-50)."""
    rng = rng or np.random.default_rng()
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    start = rng.integers(0, len(noise) - len(clean) + 1)
    noise = noise[start:start + len(clean)]
    p_c, p_n = (clean ** 2).mean(), (noise ** 2).mean() + 1e-12
    scale = np.sqrt(p_c / (p_n * 10 ** (snr_db / 10)))
    out = clean + scale * noise
    peak = np.abs(out).max()
    return out / peak * 0.99 if peak > 0.99 else out   # cegah clipping

def speed_perturb(y, factor):
    """Ubah kecepatan (dan pitch) dengan resampling linear. factor 0.9-1.1."""
    idx = np.arange(0, len(y), factor)
    return np.interp(idx, np.arange(len(y)), y)
