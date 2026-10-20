# -*- coding: utf-8 -*-
"""
PIPELINE PHÂN LOẠI ẢNH TRÙNG
Tách toàn bộ ảnh gốc thành 2 folder:
  - ANH_TRUNG_YHET      : các bản TRÙNG Y HỆT (bị loại khỏi dataset)
  - ANH_KHONG_TRUNG_YHET: các ảnh giữ lại (mỗi nhóm trùng giữ 1 bản đại diện)
Chạy:  python phan_loai_anh_trung.py
"""
import os, glob, shutil
from collections import defaultdict
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(HERE, "Vehicles.v4-final_cuoi_cung.coco", "train")
OUT_TRUNG = os.path.join(HERE, "ANH_TRUNG_YHET")
OUT_KHONG = os.path.join(HERE, "ANH_KHONG_TRUNG_YHET")

for d in (OUT_TRUNG, OUT_KHONG):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

def avg_hash(path, size=8):
    im = Image.open(path).convert("L").resize((size, size))
    px = list(im.getdata()); avg = sum(px) / len(px)
    b = 0
    for i, p in enumerate(px):
        if p >= avg: b |= (1 << i)
    return b

# 1) Hash toàn bộ ảnh -> gom theo hash
imgs = sorted([p for p in glob.glob(os.path.join(COCO_DIR, "*"))
               if p.lower().endswith((".jpg", ".jpeg", ".png"))])
buckets = defaultdict(list)
for p in imgs:
    buckets[avg_hash(p)].append(p)

# 2) Mỗi nhóm: bản đầu = GIỮ (không trùng), các bản sau = TRÙNG (loại)
n_keep = n_dup = 0
manifest = []
for h, group in buckets.items():
    keep = group[0]
    shutil.copy2(keep, os.path.join(OUT_KHONG, os.path.basename(keep)))
    n_keep += 1
    for dup in group[1:]:
        shutil.copy2(dup, os.path.join(OUT_TRUNG, os.path.basename(dup)))
        n_dup += 1
        manifest.append((os.path.basename(dup), os.path.basename(keep)))

# 3) Ghi chú
with open(os.path.join(OUT_TRUNG, "_GHICHU.txt"), "w", encoding="utf-8") as f:
    f.write(f"ANH TRUNG Y HET (bi loai): {n_dup} anh\n")
    f.write("Moi dong: <anh trung bi loai>  <-- trung voi -->  <anh duoc giu>\n")
    f.write("="*70 + "\n")
    for dup, keep in manifest:
        f.write(f"{dup}\n   <-- trung voi --> {keep}\n\n")
with open(os.path.join(OUT_KHONG, "_GHICHU.txt"), "w", encoding="utf-8") as f:
    f.write(f"ANH KHONG TRUNG (giu lai): {n_keep} anh\n")
    f.write("Day la tap anh sach (moi noi dung chi giu 1 ban).\n")

print("="*55)
print(f"Tong anh goc            : {len(imgs)}")
print(f"ANH_KHONG_TRUNG_YHET    : {n_keep} anh  (giu lai)")
print(f"ANH_TRUNG_YHET          : {n_dup} anh  (loai)")
print(f"Kiem tra: {n_keep} + {n_dup} = {n_keep + n_dup}")
print("="*55)
print("Folder 1:", OUT_TRUNG)
print("Folder 2:", OUT_KHONG)
