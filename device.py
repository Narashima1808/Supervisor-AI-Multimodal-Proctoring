# features/device.py  (FULLY PATCHED VERSION)
import cv2
import csv
from datetime import datetime
import os
import traceback
from ultralytics import YOLO
import numpy as np

# Globals
_model = None
_CONF_THRESH = 0.25
_IMG_SIZE = 960
LOG_FILE = "device_log.csv"

_ELECTRONIC_KEYWORDS = [
    "phone", "cell", "mobile", "laptop", "notebook", "computer",
    "monitor", "tv", "screen", "tablet", "remote", "keyboard", "mouse", "speaker"
]


# ---------------------------------------------------------
#  SAFE + ROBUST MODEL LOADING
# ---------------------------------------------------------
def init(config: dict = None):
    """
    Attempts to load the YOLO model safely.
    If primary weights fail → tries yolov8n.pt → tries CPU fallback.
    """
    global _model, _CONF_THRESH, _IMG_SIZE

    cfg = config or {}
    requested = cfg.get("weights", "yolov8m.pt")
    _CONF_THRESH = float(cfg.get("conf", 0.25))
    _IMG_SIZE = int(cfg.get("imgsz", 960))

    print(f"[device.init] Requested weights: {requested}")

    def try_load(weights, device=None):
        """Internal helper to safely load YOLO weights"""
        try:
            print(f"[device.init] Loading {weights}  device={device}")
            if device:
                m = YOLO(weights, device=device)
            else:
                m = YOLO(weights)
            # Access a property to ensure model loaded properly
            _ = m.names
            print(f"[device.init] Loaded {weights} successfully.")
            return m
        except Exception as e:
            print(f"[device.init] ❌ Failed to load {weights}: {e}")
            print(traceback.format_exc())
            return None

    # 1) Try requested model
    _model = try_load(requested)

    # 2) Fallback to small model
    if _model is None:
        print("[device.init] ⚠ Falling back to yolov8n.pt")
        _model = try_load("yolov8n.pt")

    # 3) Final fallback: CPU
    if _model is None:
        print("[device.init] ⚠ Falling back to CPU mode")
        _model = try_load("yolov8n.pt", device="cpu")

    # Create log file if not exists
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])

    # Final result
    if _model is None:
        print("[device.init] ❌ All loads failed. Device detector DISABLED.")
    else:
        print("[device.init] ✅ Device detector READY.")


# ---------------------------------------------------------
#  UTILITY: Is electronic?
# ---------------------------------------------------------
def _is_electronic(label):
    l = (label or "").lower()
    return any(k in l for k in _ELECTRONIC_KEYWORDS)


# ---------------------------------------------------------
#  SAFE RUN() — NEVER THROWS
# ---------------------------------------------------------
def run(frame=None, timestamp=None, **kwargs):
    """
    Always returns a dict:
      {
        'total_devices': int,
        'total_phones': int,
        'device_conf_max': float,
        'device_error': optional string
      }
    """
    global _model, _CONF_THRESH, _IMG_SIZE

    # If model failed loading → never crash
    if _model is None:
        return {
            "total_devices": 0,
            "total_phones": 0,
            "device_conf_max": 0.0,
            "device_error": "model_not_loaded"
        }

    try:
        if frame is None:
            return {"total_devices": 0, "total_phones": 0, "device_conf_max": 0.0}

        results = _model(frame, imgsz=_IMG_SIZE, conf=_CONF_THRESH, verbose=False)

        device_boxes = []
        phone_boxes = []
        max_conf = 0.0

        for r in results:
            boxes = getattr(r, 'boxes', [])
            if boxes is None:
                continue

            for b in boxes:
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                cls_id = int(b.cls[0]) if hasattr(b, "cls") else None

                label = _model.names.get(cls_id, str(cls_id)) if hasattr(_model, "names") else str(cls_id)

                if conf < _CONF_THRESH:
                    continue

                # Only electronic devices
                if _is_electronic(label):
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    device_boxes.append((x1, y1, x2, y2, label, conf))

                    if "phone" in label.lower():
                        phone_boxes.append((x1, y1, x2, y2, label, conf))

                    if conf > max_conf:
                        max_conf = conf

                    # Log to CSV
                    with open(LOG_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.utcnow().isoformat(),
                            label,
                            f"{conf:.3f}",
                            x1, y1, x2, y2
                        ])

        return {
            "total_devices": len(device_boxes),
            "total_phones": len(phone_boxes),
            "device_conf_max": max_conf
        }

    except Exception as e:
        # LOG FULL TRACEBACK for debugging
        os.makedirs("logs", exist_ok=True)
        with open("logs/device_traceback.log", "a") as f:
            f.write(f"\n[{datetime.utcnow().isoformat()}] Device run error:\n")
            f.write(traceback.format_exc())

        return {
            "total_devices": 0,
            "total_phones": 0,
            "device_conf_max": 0.0,
            "device_error": str(e)
        }

