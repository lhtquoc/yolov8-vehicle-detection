# -*- coding: utf-8 -*-
"""
GIAO DIỆN NHẬN DIỆN & ĐẾM XE CỘ (YOLOv8) — NGƯỠNG ĐỘNG THEO ẢNH
- Nhập best.pt
- Nhấn NHẬN DIỆN: chạy 1 lần ở ngưỡng THẤP để lấy TẤT CẢ ứng viên trong ảnh
- Kéo slider (thấp -> cao) để lọc TỨC THÌ, không chạy lại model
- Tự gợi ý ngưỡng tối ưu theo TỪNG ảnh (tìm khoảng hụt độ tin cậy)
- Vẽ bbox kèm % + liệt kê từng xe + tổng số

Yêu cầu: customtkinter, ultralytics, opencv-python, pillow, numpy
"""
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image
from ultralytics import YOLO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Tự tìm best.pt cùng thư mục với app; nếu không có thì chọn file trong giao diện
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_HERE, "best.pt"),  # BẢN NỘP: model nằm chung thư mục với app
]
DEFAULT_MODEL = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])

LOW_CONF = 0.05         # chạy model 1 lần ở ngưỡng thấp -> lấy hết ứng viên
TRAIN_IMGSZ = 640       # model train ở 640: ảnh thường giữ đúng size này (phóng ảnh nhỏ lên to -> model TRƯỢT, ra 0 box!)
MAX_INFER_IMGSZ = 1280  # chỉ ảnh độ phân giải CAO mới lên tới mức này (để bắt xe nhỏ / ở xa)
START_CONF = 0.25       # ngưỡng hiển thị ban đầu

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_COLORS = {"Bus": (68, 68, 255), "Motorcycle": (68, 255, 68),
                "Car": (255, 136, 68), "Truck": (0, 170, 255)}
DEFAULT_COLOR = (200, 200, 200)


def suggest_threshold(scores):
    """Gợi ý ngưỡng theo ẢNH: tìm 'khoảng hụt' lớn nhất giữa các độ tin cậy (tách xe thật / nhiễu)."""
    if not scores:
        return START_CONF
    s = sorted(scores, reverse=True)
    if len(s) <= 2:
        return round(max(0.10, min(s) - 0.05), 2)
    best_gap, thr = 0.0, START_CONF
    for i in range(len(s) - 1):
        gap = s[i] - s[i + 1]
        mid = (s[i] + s[i + 1]) / 2
        if gap > best_gap and 0.12 <= mid <= 0.85:
            best_gap, thr = gap, mid
    if best_gap < 0.10:          # không có khoảng tách rõ -> giữ thấp để bắt nhiều
        thr = max(0.10, min(s))
    return round(thr, 2)


class VehicleDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nhận diện & Đếm Xe cộ - YOLOv8 (ngưỡng động theo ảnh)")
        self.geometry("1180x800")
        self.model = None
        self.image_path = None
        self.base_image = None        # ảnh gốc (cv2 BGR) để vẽ lại
        self.detections = []          # [(score, cls_name, x1, y1, x2, y2)] - chạy 1 lần ở LOW_CONF
        self._build_ui()
        if os.path.exists(DEFAULT_MODEL):
            self.path_entry.insert(0, DEFAULT_MODEL)
            self.load_model(DEFAULT_MODEL)

    def _build_ui(self):
        sb = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        sb.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(sb, text="YOLOv8\nVehicle Detector",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 16))

        ctk.CTkLabel(sb, text="1. Trọng số model (.pt):", anchor="w").pack(fill="x", padx=20)
        self.path_entry = ctk.CTkEntry(sb, placeholder_text="Đường dẫn best.pt")
        self.path_entry.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(sb, text="Chọn & Nạp Model", command=self.browse_model).pack(fill="x", padx=20, pady=5)
        self.model_status = ctk.CTkLabel(sb, text="Chưa nạp model", text_color="#ff7777", anchor="w",
                                         font=ctk.CTkFont(size=12))
        self.model_status.pack(fill="x", padx=20, pady=(2, 0))

        ctk.CTkLabel(sb, text="2. Ảnh cần nhận diện:", anchor="w").pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkButton(sb, text="Chọn Ảnh", command=self.browse_image).pack(fill="x", padx=20, pady=5)
        self.img_status = ctk.CTkLabel(sb, text="Chưa chọn ảnh", text_color="#aaaaaa", anchor="w",
                                       font=ctk.CTkFont(size=11))
        self.img_status.pack(fill="x", padx=20)

        ctk.CTkButton(sb, text="NHẬN DIỆN", height=44, font=ctk.CTkFont(size=16, weight="bold"),
                      fg_color="#28a745", hover_color="#218838",
                      command=self.run_detection).pack(fill="x", padx=20, pady=(16, 8))

        # Ngưỡng ĐỘNG - kéo để lọc tức thì
        self.conf_label = ctk.CTkLabel(sb, text=f"3. Ngưỡng (kéo thấp→cao): {START_CONF:.2f}", anchor="w")
        self.conf_label.pack(fill="x", padx=20, pady=(8, 0))
        self.conf_slider = ctk.CTkSlider(sb, from_=LOW_CONF, to=1.0, command=self._on_conf)
        self.conf_slider.set(START_CONF)
        self.conf_slider.pack(fill="x", padx=20, pady=5)
        self.conf_hint = ctk.CTkLabel(sb, text="(Nhận diện trước, rồi kéo slider để lọc tức thì)",
                                      anchor="w", font=ctk.CTkFont(size=11), text_color="#aaaaaa",
                                      wraplength=300, justify="left")
        self.conf_hint.pack(fill="x", padx=20)
        ctk.CTkButton(sb, text="🎯 Tự chọn ngưỡng theo ảnh", fg_color="#6f42c1", hover_color="#5a32a3",
                      command=self.auto_for_image).pack(fill="x", padx=20, pady=4)

        res = ctk.CTkFrame(sb); res.pack(fill="x", padx=20, pady=10)
        self.total_label = ctk.CTkLabel(res, text="Tổng số xe: 0",
                                        font=ctk.CTkFont(size=20, weight="bold"), text_color="#f39c12")
        self.total_label.pack(pady=(10, 4))
        self.breakdown_label = ctk.CTkLabel(res, text="", font=ctk.CTkFont(size=14), justify="left")
        self.breakdown_label.pack(pady=(0, 10))

        main = ctk.CTkFrame(self); main.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.image_display = ctk.CTkLabel(main, text="Ảnh xem trước sẽ hiển thị ở đây",
                                          font=ctk.CTkFont(size=18))
        self.image_display.pack(expand=True, fill="both", padx=10, pady=10)
        self.main_frame = main

    # ---------- Model / Ảnh ----------
    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt")])
        if p:
            self.path_entry.delete(0, "end"); self.path_entry.insert(0, p); self.load_model(p)

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.model_status.configure(
                text=f"✓ Đã nạp: {os.path.basename(path)} | Lớp: " + ", ".join(self.model.names.values()),
                text_color="#55dd77")
        except Exception as e:
            self.model = None
            self.model_status.configure(text="Lỗi nạp model!", text_color="#ff7777")
            messagebox.showerror("Lỗi", f"Không nạp được model:\n{e}")

    def browse_image(self):
        p = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if p:
            self.image_path = p
            self.base_image = cv2.imread(p)
            self.detections = []
            self.img_status.configure(text=os.path.basename(p))
            self.total_label.configure(text="Tổng số xe: 0"); self.breakdown_label.configure(text="")
            self._show(self.base_image)

    # ---------- Nhận diện 1 lần ở ngưỡng thấp ----------
    def run_detection(self):
        if self.model is None:
            messagebox.showwarning("Thiếu model", "Vui lòng nạp best.pt trước."); return
        if self.base_image is None:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn ảnh trước."); return
        # imgsz THÍCH ỨNG theo độ phân giải ảnh (cạnh dài, làm tròn bội số 32, kẹp trong [640, 1280]):
        #   ảnh ~640 -> 640 (đúng size train, nhận diện tốt); ảnh lớn -> tăng dần tới 1280 để bắt xe nhỏ.
        #   QUAN TRỌNG: phóng ảnh nhỏ (vd 640) lên 1280 sẽ làm model TRƯỢT, ra 0 box!
        long_side = max(self.base_image.shape[:2])
        imgsz = int(min(MAX_INFER_IMGSZ, max(TRAIN_IMGSZ, round(long_side / 32) * 32)))
        result = self.model(self.image_path, conf=LOW_CONF, imgsz=imgsz, verbose=False)[0]
        names = self.model.names
        self.detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            self.detections.append((float(b.conf[0]), names[int(b.cls[0])], x1, y1, x2, y2))
        # Tự gợi ý ngưỡng theo ẢNH này rồi vẽ
        thr = suggest_threshold([d[0] for d in self.detections])
        self.conf_slider.set(thr)
        self._on_conf(thr)
        self.conf_hint.configure(
            text=f"Đã lấy {len(self.detections)} ứng viên (imgsz={imgsz}). Gợi ý ngưỡng = {thr:.2f}. "
                 f"Kéo slider để lọc tức thì.", text_color="#55dd77")

    # ---------- Slider lọc tức thì ----------
    def _on_conf(self, v):
        conf = float(v)
        self.conf_label.configure(text=f"3. Ngưỡng (kéo thấp→cao): {conf:.2f}")
        if self.detections:
            self._render(conf)

    def auto_for_image(self):
        if not self.detections:
            self.conf_hint.configure(text="Hãy bấm NHẬN DIỆN trước.", text_color="#ff9966"); return
        thr = suggest_threshold([d[0] for d in self.detections])
        self.conf_slider.set(thr); self._on_conf(thr)

    def _render(self, conf):
        """Lọc các ứng viên đã cache theo ngưỡng, vẽ lại — KHÔNG chạy lại model."""
        img = self.base_image.copy()
        kept = [d for d in self.detections if d[0] >= conf]
        kept.sort(key=lambda x: -x[0])
        counts = {}
        for score, cls_name, x1, y1, x2, y2 in kept:
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            label = f"{cls_name} {score*100:.0f}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            counts[cls_name] = counts.get(cls_name, 0) + 1
        self.total_label.configure(text=f"Tổng số xe: {len(kept)}")
        self.breakdown_label.configure(
            text="\n".join(f"{i}. {n}  {s*100:.0f}%" for i, (s, n, *_ ) in enumerate(kept, 1))
            if kept else "(Không có xe nào ở ngưỡng này)")
        self._show(img)

    def _show(self, cv_bgr):
        pil = Image.fromarray(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))
        self.main_frame.update_idletasks()   # chỉ tính layout, KHÔNG xử lý lại event -> tránh đệ quy khi kéo slider
        fw = max(self.main_frame.winfo_width() - 20, 400)
        fh = max(self.main_frame.winfo_height() - 20, 400)
        pil.thumbnail((fw, fh), Image.Resampling.LANCZOS)
        ci = ctk.CTkImage(light_image=pil, dark_image=pil, size=(pil.width, pil.height))
        self.image_display.configure(image=ci, text=""); self.image_display.image = ci


if __name__ == "__main__":
    VehicleDetectionApp().mainloop()
