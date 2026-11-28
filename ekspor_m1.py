"""Ekspor hasil latih_m1.py ke ONNX (fp32 + int8) dalam format yang dibaca run_m1_zipformer.py.
  python ekspor_m1.py --resep $RESEP                 (semua exp_m1/fold*)
  python ekspor_m1.py --resep $RESEP --folds 0       (satu putaran)
Keluaran: models/m1_ft/fold{i}/{encoder,decoder,joiner}-*.onnx (+ .int8.onnx) dan tokens.txt"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--resep", required=True)
ap.add_argument("--exp-dir", default="exp_m1")
ap.add_argument("--tokens", default="models/zipformer_id/tokens.txt", help="tokens.txt asli untuk sherpa-onnx")
ap.add_argument("--out", default="models/m1_ft")
ap.add_argument("--decode-chunk-len", type=int, default=32, help="harus sama dengan model asli (bawaan resep 32)")
ap.add_argument("--folds", nargs="*", type=int)
a = ap.parse_args()

resep, exp, base = Path(a.resep).resolve(), Path(a.exp_dir).resolve(), Path(__file__).resolve().parent
icefall = resep.parents[3]
ENV = {**os.environ, "PYTHONPATH": str(icefall) + os.pathsep + os.environ.get("PYTHONPATH", "")}
bantuan = subprocess.run([sys.executable, "export-onnx.py", "--help"], cwd=resep, capture_output=True, text=True,
                         env=ENV)
if "--tokens" not in bantuan.stdout:
    raise SystemExit("export-onnx.py versi ini tidak punya --tokens (masih --bpe-model). Pakai icefall terbaru.\n"
                     + bantuan.stderr[-1500:])

# export-onnx.py mencari simbol <blk> (id 0) dan <unk>; tokens.txt bookbot memakai <eps> dan <UNK>.
# Hanya nama simbol yang diganti; id tetap sama, jadi tokens.txt asli tetap dipakai sherpa-onnx.
GANTI = {"<eps>": "<blk>", "<UNK>": "<unk>"}
tok_ekspor = exp / "tokens_ekspor.txt"
baris = [l.split() for l in (base / a.tokens).read_text(encoding="utf-8").splitlines() if l.strip()]
tok_ekspor.write_text("".join(f"{GANTI.get(s, s)} {i}\n" for s, i in baris), encoding="utf-8")
nama = {s for s, _ in (l.split() for l in tok_ekspor.read_text(encoding="utf-8").splitlines())}
if not {"<blk>", "<unk>"} <= nama:
    raise SystemExit(f"{tok_ekspor} tidak memuat <blk> dan <unk>; periksa {a.tokens}")

dirs = sorted(exp.glob("fold*"))
if a.folds is not None:
    dirs = [d for d in dirs if int(d.name[4:]) in a.folds]
if not dirs:
    raise SystemExit(f"Tidak ada {exp}/fold*; jalankan latih_m1.py dulu.")

for d in dirs:
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    ep = info["epoch"]
    cmd = [sys.executable, "export-onnx.py", "--exp-dir", str(d), "--tokens", str(tok_ekspor),
           "--epoch", str(ep), "--avg", "1", "--use-averaged-model", "0",
           "--decode-chunk-len", str(a.decode_chunk_len)] + info.get("model_args", "").split()
    print(f"\n=== {d.name} (epoch {ep}, uji {info['uji_pembicara']}) ===\n", " ".join(cmd))
    r = subprocess.run(cmd, cwd=resep, env=ENV)
    if r.returncode:
        raise SystemExit(f"export-onnx.py gagal untuk {d.name}")
    dst = base / a.out / d.name; dst.mkdir(parents=True, exist_ok=True)
    for part in ("encoder", "decoder", "joiner"):
        plain = d / f"{part}-epoch-{ep}-avg-1.onnx"
        if not plain.exists():
            raise SystemExit(f"{plain.name} tidak terbentuk; lihat keluaran export-onnx.py di atas.")
        q = d / f"{part}-epoch-{ep}-avg-1.int8.onnx"
        if not q.exists():                                     # versi lama export-onnx.py tidak membuat int8
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(model_input=str(plain), model_output=str(q),
                             op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
        for f in (plain, q):
            shutil.copy(f, dst / f.name)
    shutil.copy(base / a.tokens, dst / "tokens.txt")
    (dst / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    mb = sum(f.stat().st_size for f in dst.glob("*.int8.onnx")) / 1e6
    print(f"  -> {dst} (int8 total {mb:.0f} MB)")

print(f"\nSelesai. Unduh folder {a.out}/ (cukup berkas *.int8.onnx, tokens.txt, info.json) ke laptop.")
