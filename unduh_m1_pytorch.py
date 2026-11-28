"""Unduh checkpoint PyTorch M1 (bukan ONNX) dari bookbot/pruned-transducer-stateless7-streaming-id.
ONNX di models/zipformer_id hanya bisa dipakai untuk inferensi; fine-tune butuh bobot PyTorch.
  python unduh_m1_pytorch.py            -> models/m1_pytorch/<berkas>
Urutan pilihan: pretrained.pt (state_dict) > epoch-*.pt terbesar > cpu_jit.pt (TorchScript).
Bila tidak satu pun ada, skrip berhenti dan menampilkan isi repo; kirim daftar itu."""
import re
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "bookbot/pruned-transducer-stateless7-streaming-id"
dst = Path(__file__).resolve().parent / "models" / "m1_pytorch"
files = list_repo_files(REPO)
print(f"{len(files)} berkas di {REPO}:")
for f in files:
    print("  ", f)

pt = [f for f in files if f.endswith(".pt")]
pilih = [f for f in pt if Path(f).name == "pretrained.pt"]
if not pilih:
    ep = sorted((f for f in pt if re.fullmatch(r"epoch-\d+\.pt", Path(f).name)),
                key=lambda f: int(re.findall(r"\d+", Path(f).name)[0]))
    pilih = ep[-1:]
if not pilih:
    pilih = [f for f in pt if Path(f).name == "cpu_jit.pt"]
if not pilih:
    raise SystemExit("\nTidak ada pretrained.pt / epoch-*.pt / cpu_jit.pt. Fine-tune M1 tidak bisa dari repo ini; "
                     "kirim daftar berkas di atas.")

p = hf_hub_download(REPO, pilih[0], local_dir=str(dst))
print(f"\nTerunduh: {p}  ({Path(p).stat().st_size / 1e6:.0f} MB)")
print(f"Pakai di latih_m1.py: --pretrained {Path(p).relative_to(Path(__file__).resolve().parent).as_posix()}")
