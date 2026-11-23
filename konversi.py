import os, subprocess
SRC, DST = "raw", "processed"
for spk in sorted(os.listdir(SRC)):
    os.makedirs(f"{DST}/{spk}", exist_ok=True)
    for f in os.listdir(f"{SRC}/{spk}"):
        if f.lower().endswith(".wav"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{SRC}/{spk}/{f}",
                            "-ar", "16000", "-ac", "1", f"{DST}/{spk}/{f}"], check=True)
print("Selesai")
