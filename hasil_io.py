import csv
from pathlib import Path

FIELDS = ["model", "spk", "id", "jenis", "ref", "gold_items", "hyp", "pred_intent",
          "pred_items", "latency_ms", "ref_ph", "hyp_ph"]


def open_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    return f, w


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
