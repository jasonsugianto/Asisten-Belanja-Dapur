"""Ubah naskah_asli.csv (format Google Sheet) menjadi script.csv yang bersih + laporan."""
import csv, collections

rows = list(csv.reader(open("naskah_asli.csv", encoding="utf-8-sig")))
data = [r for r in rows if r and r[0].strip().isdigit()]   # lewati judul & header

out, cnt_item, cnt_jenis = [], collections.Counter(), collections.Counter()
for r in data:
    sid, kalimat, jenis, items = int(r[0]), r[1].strip(), r[2].strip(), r[3].strip()
    lst = [x.strip() for x in items.split(",") if x.strip()]
    out.append([sid, jenis, kalimat, "|".join(lst)])
    cnt_jenis[jenis] += 1
    cnt_item.update(lst)

with open("script.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "jenis", "transcript", "items"])
    w.writerows(out)

print(f"{len(out)} kalimat -> script.csv")
print("Jenis perintah:", dict(cnt_jenis))
print(f"{len(cnt_item)} item unik. Paling jarang muncul:")
for k, v in sorted(cnt_item.items(), key=lambda x: x[1])[:5]:
    print(f"  {k}: {v}x per pembicara")
