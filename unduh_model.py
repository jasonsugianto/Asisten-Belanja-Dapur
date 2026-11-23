"""Unduh model Zipformer fonem Indonesia (untuk M1) ke models/zipformer_id."""
from huggingface_hub import snapshot_download
from pathlib import Path

REPO = "bookbot/sherpa-onnx-pruned-transducer-stateless7-streaming-id"
dst = Path(__file__).resolve().parent / "models" / "zipformer_id"
snapshot_download(REPO, local_dir=str(dst))
print("Isi folder:")
for p in sorted(dst.rglob("*")):
    if p.is_file() and ".git" not in p.parts and ".cache" not in p.parts:
        print(f"  {p.relative_to(dst)}  {p.stat().st_size/1e6:.1f} MB")
