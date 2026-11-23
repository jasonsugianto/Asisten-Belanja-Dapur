import numpy as np


def edit_distance(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def error_rate(refs, hyps):
    """WER bila daftar kata, PER bila daftar fonem. refs/hyps: list of list."""
    err = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return err / max(sum(len(r) for r in refs), 1)


def score(rows):
    """rows: dict dengan gold_intent, pred_intent, gold_items(set), pred_items(set)."""
    tp = fp = fn = ok_intent = ok_all = 0
    for r in rows:
        g, p = r["gold_items"], r["pred_items"]
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        ok_intent += r["gold_intent"] == r["pred_intent"]
        ok_all += r["gold_intent"] == r["pred_intent"] and g == p
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    n = max(len(rows), 1)
    return {"n": len(rows), "intent_acc": ok_intent / n, "item_P": P, "item_R": R,
            "item_F1": 2 * P * R / max(P + R, 1e-9), "kalimat_benar": ok_all / n}


def latency_stats(ms):
    ms = np.array(ms, dtype=float)
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "max": float(ms.max())}
