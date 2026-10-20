# -*- coding: utf-8 -*-
"""
Tách 117 ảnh AUGMENT ROBOFLOW (.rf. — bản dư của cùng 1 cảnh) theo từng tập.
Đọc từ VEHICLE_DATASET_YOLO (đã chia sẵn) -> copy ra folder ANH_AUGMENT_THEO_TAP/<tap>/.
Lưu ý: chỉ xét ảnh GỐC (.rf.), BỎ ảnh _aug_ do Cell 6 tạo.
Chạy:  python tach_anh_augment_theo_tap.py
"""
import os, glob, shutil, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "VEHICLE_DATASET_YOLO")
OUT = os.path.join(HERE, "ANH_AUGMENT_THEO_TAP")
exts = (".jpg", ".jpeg", ".png")

if os.path.exists(OUT): shutil.rmtree(OUT)

def uuid(name):
    return os.path.splitext(name)[0].split('.rf.')[0]

assert os.path.isdir(DS), f"Khong thay {DS} — hay chay pipeline tao dataset truoc."

total = 0
report = {}
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    # chi lay anh GOC (.rf.), bo anh _aug_ cua Cell 6
    imgs = [p for p in glob.glob(os.path.join(DS, split, "images", "*"))
            if p.lower().endswith(exts) and "_aug_" not in os.path.basename(p)]
    groups = defaultdict(list)
    for p in imgs:
        groups[uuid(os.path.basename(p))].append(p)

    extra = 0
    manifest = []
    for u, files in groups.items():
        if len(files) <= 1:
            continue                      # canh chi co 1 ban -> khong phai augment du
        files = sorted(files)
        keep = files[0]                   # 1 ban dai dien (khong copy)
        for dup in files[1:]:             # cac ban DU = augment Roboflow
            shutil.copy2(dup, os.path.join(OUT, split, os.path.basename(dup)))
            manifest.append((os.path.basename(dup), os.path.basename(keep)))
            extra += 1
    report[split] = (len(groups), extra)
    total += extra
    with open(os.path.join(OUT, split, "_GHICHU.txt"), "w", encoding="utf-8") as f:
        f.write(f"Anh augment Roboflow (.rf.) DU trong tap {split.upper()}: {extra} anh\n")
        f.write("Moi dong: <ban augment di theo>  <-- cung canh -->  <ban dai dien>\n")
        f.write("="*70 + "\n")
        for dup, keep in manifest:
            f.write(f"{dup}\n   <-- cung canh --> {keep}\n\n")

print("="*55)
print("ANH AUGMENT ROBOFLOW (.rf.) DI THEO TUNG TAP")
print("="*55)
for split in ["train", "val", "test"]:
    ncanh, extra = report[split]
    print(f"  {split:5}: {extra:3} anh augment  (tren {ncanh} canh)")
print("-"*55)
print(f"  TONG : {total} anh   (khop con so 117)")
print("="*55)
print("Folder:", OUT)
