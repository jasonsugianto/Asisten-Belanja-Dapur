import os
import soundfile as sf

total, durasi, salah = 0, 0.0, []
for spk in sorted(os.listdir("processed")):
    files = [f for f in os.listdir(f"processed/{spk}") if f.endswith(".wav")]
    for f in files:
        i = sf.info(f"processed/{spk}/{f}")
        total += 1; durasi += i.duration
        if i.samplerate != 16000 or i.channels != 1:
            salah.append((spk, f, i.samplerate, i.channels))
    print(spk, len(files), "file")
print(f"total {total} file, {durasi/60:.1f} menit, {len(salah)} format salah")
for s in salah: print(" ", s)