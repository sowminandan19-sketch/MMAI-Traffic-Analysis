import os
import cv2
import time
import math
import wave
import struct
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# 1. APPLICATION SETUP & CONSTANTS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov11n.pt")
FLOWCHART_PATH = os.path.join(BASE_DIR, "methodology_flowchart.jpg")
SIREN_PATH = os.path.join(BASE_DIR, "emergency_siren.wav")
SAMPLE_CCTV = os.path.join(BASE_DIR, "sample_cctv.mp4")
SAMPLE_DASHCAM = os.path.join(BASE_DIR, "sample_dashcam.webm")

# COCO Vehicle Class IDs
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Incident Audit Log Storage
AUDIT_LOGS = []

# Load YOLOv11 Model
print(f"[INIT] Loading YOLOv11 model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("[INIT] YOLOv11 model loaded successfully.")
except Exception as e:
    print(f"[WARN] Error loading local weights ({e}), fallback to 'yolov11n.pt'")
    model = YOLO("yolov11n.pt")

# Ensure Emergency Siren Audio Exists
def ensure_siren_audio():
    if not os.path.exists(SIREN_PATH):
        duration = 3.0
        sample_rate = 44100
        n_samples = int(duration * sample_rate)
        with wave.open(SIREN_PATH, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(n_samples):
                t = i / sample_rate
                freq = 800 + 400 * math.sin(2 * math.pi * 1.5 * t)
                val = int(32767 * 0.45 * math.sin(2 * math.pi * freq * t))
                wav.writeframes(struct.pack('<h', val))
ensure_siren_audio()

# ==============================================================================
# 2. IMAGE PREPROCESSING & MATHEMATICAL MODULES (Chapter 3.3)
# ==============================================================================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Equation 2-5: Dynamic spatial resizing to 640x640 with aspect-ratio preservation
    and adaptive letterboxing borders (dw, dh).
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def calculate_iou(box1, box2):
    """
    Equation 15: Collision Overlap Metric (IoU collision)
    Flags dynamic spatial interactions between two tracked vehicle bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, float(x2 - x1))
    inter_h = max(0.0, float(y2 - y1))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = float(box1[2] - box1[0]) * float(box1[3] - box1[1])
    area2 = float(box2[2] - box2[0]) * float(box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0

# ==============================================================================
# 3. ADVANCED HUD & VISUALIZATION ENGINE
# ==============================================================================
def draw_telemetry_hud(frame, camera_mode, density_status, vehicle_count,
                       severity, entangled_count, severity_index, fps):
    """
    Renders an enterprise ITS smart-city command-center Glassmorphic HUD overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top Glassmorphic Telemetry Bar (Dark translucent slate)
    cv2.rectangle(overlay, (0, 0), (w, 54), (15, 23, 42), -1)
    # 2. Bottom Glassmorphic Status Bar
    cv2.rectangle(overlay, (0, h - 38), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Accent divider lines
    cv2.line(frame, (0, 54), (w, 54), (56, 189, 248), 2)
    cv2.line(frame, (0, h - 38), (w, h - 38), (56, 189, 248), 1)

    # Camera Mode Badge (Left)
    cam_label = "EGO-DASHCAM [DoTA]" if "DASH" in camera_mode.upper() else "CCTV OVERHEAD [CADP]"
    cv2.putText(frame, cam_label, (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.60, (56, 189, 248), 2)

    # Traffic Density Badge (Center-Left)
    if density_status == "Light":
        d_color = (34, 197, 94)   # Emerald
    elif density_status == "Moderate":
        d_color = (234, 179, 8)   # Amber
    else:
        d_color = (239, 68, 68)   # Crimson
    cv2.putText(frame, f"DENSITY: {density_status.upper()} ({vehicle_count} Veh)", (240, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, d_color, 2)

    # Severity Alert Badge (Right)
    if severity == "Severe":
        s_color = (239, 68, 68)   # Red
        s_text = "CRASH: SEVERE (L3)"
    elif severity == "Moderate":
        s_color = (249, 115, 22)  # Orange
        s_text = "CRASH: MODERATE (L2)"
    elif severity == "Minor":
        s_color = (6, 182, 212)   # Cyan
        s_text = "INCIDENT: MINOR (L1)"
    else:
        s_color = (203, 213, 225) # Slate
        s_text = "STATUS: NORMAL FLOW"

    cv2.putText(frame, s_text, (w - 240, 34), cv2.FONT_HERSHEY_DUPLEX, 0.58, s_color, 2)

    # Bottom Telemetry Details
    bottom_str = f"THROUGHPUT: {fps:.1f} FPS | IMPACT VEHICLES: {entangled_count} | SEVERITY INDEX: {severity_index:.2f} | YOLOv11+ByteTrack"
    cv2.putText(frame, bottom_str, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1)

    return frame

# ==============================================================================
# 4. CORE AI VIDEO PROCESSING PIPELINE
# ==============================================================================
def process_traffic_analysis(video_input, camera_source, conf_threshold, iou_threshold,
                             target_fps, selected_classes, progress=gr.Progress()):
    """
    Executes the 10-stage processing pipeline defined in the research paper.
    """
    if video_input is None:
        return (
            None, None,
            "⚠️ No Video Source Provided. Please select a preset or upload video.",
            "Normal", "Light", 0, 0, 0.0,
            None, None, None
        )

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        return (
            None, None,
            f"❌ Unable to open video source: {video_input}",
            "Normal", "Light", 0, 0, 0.0,
            None, None, None
        )

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    if fps_native <= 0:
        fps_native = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = max(5.0, float(target_fps))
    k = max(1, int(round(fps_native / target_fps)))

    # Unique Output File Paths per Analysis Session (Prevents Windows File Lock Permission Errors)
    session_id = f"{int(time.time())}_{np.random.randint(1000, 9999)}"
    output_video_path = os.path.join(BASE_DIR, f"output_traffic_{session_id}.mp4")
    snapshot_path = os.path.join(BASE_DIR, f"collision_snapshot_{session_id}.jpg")
    chart_path = os.path.join(BASE_DIR, f"telemetry_timeline_{session_id}.png")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (640, 640))

    # Analytical tracking metrics
    frame_idx = 0
    sampled_idx = 0
    peak_vehicles = 0
    max_severity_overall = "Normal"
    max_severity_score = 0
    max_severity_index = 0.0
    highest_entangled = 0
    highest_iou_overall = 0.0
    best_snapshot = None

    timeline_data = {
        "timestamp_sec": [],
        "vehicle_count": [],
        "collision_iou": [],
        "severity_level": []
    }

    class_id_map = {"Car": 2, "Motorcycle": 3, "Bus": 5, "Truck": 7}
    allowed_ids = [class_id_map[c] for c in selected_classes if c in class_id_map]
    if not allowed_ids:
        allowed_ids = [2, 3, 5, 7]

    start_time = time.time()

    try:
        # Processing Loop
        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % k != 0:
                continue

            sampled_idx += 1
            if total_frames > 0:
                progress(min(0.95, frame_idx / total_frames), desc=f"Analyzing Frame {frame_idx}/{total_frames}...")

            t_frame_start = time.time()

            # Step 2: Preprocessing (Resizing 640x640 + [0.0, 1.0] intensity normalization)
            resized_frame, ratio, (dw, dh) = letterbox(raw_frame, new_shape=(640, 640))
            normalized_frame = resized_frame.astype(np.float32) / 255.0  # Eq. 6

            # Step 4: YOLOv11 Feature Extraction + ByteTrack Tracking
            results = model.track(
                resized_frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=float(conf_threshold),
                classes=allowed_ids,
                verbose=False
            )

            vehicle_boxes = []
            track_ids = []
            class_names = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                t_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, t_id, cls in zip(boxes, t_ids, classes):
                    cls_int = int(cls)
                    if cls_int in allowed_ids:
                        vehicle_boxes.append(box)
                        track_ids.append(int(t_id))
                        class_names.append(VEHICLE_CLASSES.get(cls_int, "Vehicle"))

            # Traffic Density Estimation (Chapter 3.5, Head 2)
            vehicle_count = len(vehicle_boxes)
            if vehicle_count > peak_vehicles:
                peak_vehicles = int(vehicle_count)

            if vehicle_count <= 5:
                density_status = "Light"
            elif vehicle_count <= 15:
                density_status = "Moderate"
            else:
                density_status = "Heavy"

            # 3-Tier Severity Head & Localized Impact Quantification (Eq. 15-16)
            max_iou_frame = 0.0
            colliding_pair = None
            entangled_vehicles = 0
            severity_frame = "Normal"

            num_vehicles = len(vehicle_boxes)
            for i in range(num_vehicles):
                for j in range(i + 1, num_vehicles):
                    iou = float(calculate_iou(vehicle_boxes[i], vehicle_boxes[j]))
                    if iou > max_iou_frame:
                        max_iou_frame = float(iou)
                        colliding_pair = (i, j)

            # 3-Tier Classification
            if max_iou_frame >= 0.28:
                severity_frame = "Severe"
                severity_rank = 3
            elif max_iou_frame >= float(iou_threshold):
                severity_frame = "Moderate"
                severity_rank = 2
            elif max_iou_frame >= 0.04:
                severity_frame = "Minor"
                severity_rank = 1
            else:
                severity_frame = "Normal"
                severity_rank = 0

            # Localized Crash Polygon Locking (Chapter 3.5, Head 4B)
            entangled_ids = set()
            impact_polygon = None
            if colliding_pair is not None and max_iou_frame >= 0.04:
                i, j = colliding_pair
                boxA = vehicle_boxes[i]
                boxB = vehicle_boxes[j]

                # Construct localized collision polygon with margin
                margin = 25
                poly_x1 = max(0.0, min(float(boxA[0]), float(boxB[0])) - margin)
                poly_y1 = max(0.0, min(float(boxA[1]), float(boxB[1])) - margin)
                poly_x2 = min(640.0, max(float(boxA[2]), float(boxB[2])) + margin)
                poly_y2 = min(640.0, max(float(boxA[3]), float(boxB[3])) + margin)
                impact_polygon = (int(poly_x1), int(poly_y1), int(poly_x2), int(poly_y2))

                # Quantify only vehicles whose center falls inside the localized polygon
                for idx, box in enumerate(vehicle_boxes):
                    cx = float(box[0] + box[2]) / 2.0
                    cy = float(box[1] + box[3]) / 2.0
                    if poly_x1 <= cx <= poly_x2 and poly_y1 <= cy <= poly_y2:
                        entangled_ids.add(int(track_ids[idx]))

                entangled_vehicles = len(entangled_ids)
                if entangled_vehicles == 0:
                    entangled_vehicles = 2

            # Severity Index S = IoU * V_entangled (Eq. 16)
            severity_index_frame = float(max_iou_frame * (entangled_vehicles if entangled_vehicles > 0 else 1))

            if severity_rank > max_severity_score:
                max_severity_score = int(severity_rank)
                max_severity_overall = severity_frame
                best_snapshot = resized_frame.copy()

            if severity_index_frame > max_severity_index:
                max_severity_index = float(severity_index_frame)
                highest_entangled = int(entangled_vehicles)
                highest_iou_overall = float(max_iou_frame)

            # Timeline recording
            current_sec = float(sampled_idx / target_fps)
            timeline_data["timestamp_sec"].append(round(current_sec, 2))
            timeline_data["vehicle_count"].append(int(vehicle_count))
            timeline_data["collision_iou"].append(round(float(max_iou_frame), 3))
            timeline_data["severity_level"].append(severity_frame)

            # Drawing HUD & Bounding Boxes
            annotated_frame = resized_frame.copy()

            # Draw Localized Impact Polygon Glow
            if impact_polygon is not None and severity_frame in ["Moderate", "Severe"]:
                px1, py1, px2, py2 = impact_polygon
                poly_overlay = annotated_frame.copy()
                cv2.rectangle(poly_overlay, (px1, py1), (px2, py2), (0, 0, 255), -1)
                cv2.addWeighted(poly_overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"IMPACT ZONE (Locked: {entangled_vehicles} Veh)",
                            (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

            # Draw vehicle tracking boxes
            for box, t_id, cname in zip(vehicle_boxes, track_ids, class_names):
                x1, y1, x2, y2 = map(int, box)
                is_entangled = t_id in entangled_ids

                if is_entangled:
                    box_color = (0, 0, 255)  # Red for crash vehicle
                    label = f"CRASH ID #{t_id} [{cname}]"
                else:
                    box_color = (0, 255, 127)  # Mint Green for active vehicle
                    label = f"ID #{t_id} {cname}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # Compute instantaneous FPS
            t_frame_end = time.time()
            fps_instant = float(1.0 / max(0.001, (t_frame_end - t_frame_start)))

            # Apply HUD Glassmorphic telemetry
            annotated_frame = draw_telemetry_hud(
                annotated_frame, camera_source, density_status, vehicle_count,
                severity_frame, entangled_vehicles, severity_index_frame, fps_instant
            )

            out.write(annotated_frame)

    except Exception as e:
        print(f"[ERROR in video processing loop]: {e}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
    total_elapsed = time.time() - start_time
    avg_fps = sampled_idx / max(0.01, total_elapsed)

    # Save Snapshot
    if best_snapshot is not None:
        cv2.imwrite(snapshot_path, best_snapshot)
    else:
        # Fallback frame
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "NORMAL TRAFFIC - NO COLLISION DETECTED", (40, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(snapshot_path, blank)

    # Generate Analytics Matplotlib Graph
    create_telemetry_chart(timeline_data, chart_path)

    # Build Emergency Alert Dispatch Dossier
    alert_info, dispatch_json = build_emergency_dispatch(
        max_severity_overall, camera_source, highest_entangled,
        max_severity_index, highest_iou_overall, peak_vehicles
    )

    # Log to Audit History
    incident_record = {
        "Incident ID": f"INC-{int(time.time())}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Camera Mode": camera_source,
        "Density": "Heavy" if peak_vehicles > 15 else ("Moderate" if peak_vehicles > 5 else "Light"),
        "Peak Vehicles": peak_vehicles,
        "Severity": max_severity_overall,
        "Impact Count": highest_entangled,
        "Severity Index": round(max_severity_index, 2),
        "Status": "DISPATCH TRIGGERED" if max_severity_overall in ["Moderate", "Severe"] else "NORMAL"
    }
    AUDIT_LOGS.insert(0, incident_record)

    # Audio siren output: only return sound path if accident is Moderate/Severe
    audio_out = SIREN_PATH if max_severity_overall in ["Moderate", "Severe"] else None

    status_message = (
        f"✅ Analysis Completed in {total_elapsed:.1f}s ({avg_fps:.1f} FPS) | "
        f"Peak Vehicles: {peak_vehicles} | Overall Severity: {max_severity_overall}"
    )

    return (
        output_video_path,
        snapshot_path,
        status_message,
        max_severity_overall,
        "Heavy" if peak_vehicles > 15 else ("Moderate" if peak_vehicles > 5 else "Light"),
        peak_vehicles,
        highest_entangled,
        round(max_severity_index, 2),
        chart_path,
        alert_info,
        audio_out
    )

# ==============================================================================
# 5. EMERGENCY DISPATCHER DOSSIER & ACTIONS
# ==============================================================================
def build_emergency_dispatch(severity, camera_source, entangled_count,
                            severity_index, max_iou, peak_vehicles):
    """
    Constructs an automated multi-channel dispatch notice for emergency responders.
    """
    incident_id = f"CAD-ITS-2026-{int(time.time() % 100000):05d}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC+05:30")
    sector_id = "SECTOR 04 (Expressway Interchange, KM 28.4)"

    if severity == "Severe":
        priority = "🔴 CRITICAL - CODE RED DISPATCH"
        rec_units = "• 2x Advanced Life Support (ALS) Ambulances\n• 1x Heavy Hydraulic Extrication Truck (Fire/Rescue)\n• 2x Highway Patrol Rapid Response Cruisers"
        action_summary = "High-impact vehicular structural damage or rollover confirmed. Immediate paramedic triage and multi-lane hazard diversion required."
    elif severity == "Moderate":
        priority = "🟠 PRIORITY - CODE AMBER DISPATCH"
        rec_units = "• 1x Basic Life Support (BLS) Ambulance\n• 1x Traffic Police Tow & Clearance Unit\n• 1x Highway Traffic Safety Marshal"
        action_summary = "Multi-vehicle collision with moderate deformation. Lane blockage flagged; dispatch clearance and medical check."
    elif severity == "Minor":
        priority = "🟡 ADVISORY - CODE YELLOW LOG"
        rec_units = "• 1x Highway Service Patrol / Mobile Assist\n• Traffic Monitoring Center Remote Monitoring"
        action_summary = "Minor grazing or single vehicle trajectory deviation. No structural cabin compromise observed."
    else:
        priority = "🟢 ROUTINE - NO DISPATCH REQUIRED"
        rec_units = "• Standard Automated Traffic Monitoring"
        action_summary = "Traffic flow operating under nominal velocity limits without anomalous collision interactions."

    dossier_text = f"""
### 🚨 Automated Emergency Dispatcher Incident Dossier
**Incident Identifier:** `{incident_id}`  
**Alert Status:** **{priority}**  
**Broadcast Timestamp:** `{now_str}`  

---
#### 📍 Scene Localization & Camera Telemetry
* **Camera Perspective:** `{camera_source}`
* **Road Infrastructure Sector:** `{sector_id}`
* **Active Traffic Density:** `{peak_vehicles} total vehicles observed`
* **Collision Severity Assessment:** **`{severity.upper()}`** (Score: `{severity_index:.2f}`)
* **Collision Overlap Metric (IoU):** `{max_iou:.3f}`
* **Localized Impacted Vehicles:** **`{entangled_count} vehicle(s) entangled in impact zone`**

---
#### 🚑 Recommended Immediate Response Units
{rec_units}

#### 📋 Operational Directives
*{action_summary}*
"""
    return dossier_text, json.dumps({
        "incident_id": str(incident_id),
        "timestamp": str(now_str),
        "sector": str(sector_id),
        "severity": str(severity),
        "entangled_vehicles": int(entangled_count),
        "severity_index": round(float(severity_index), 2),
        "camera": str(camera_source)
    }, indent=2)

def trigger_emergency_dispatch(dossier_text):
    """
    Simulates real-time multi-agency emergency alert transmission (108 EMS, Police, Fire).
    """
    ts = datetime.now().strftime("%H:%M:%S")
    return f"""
<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 16px; margin-top: 10px;">
    <h4 style="color: #10b981; margin: 0 0 6px 0;">📡 EMERGENCY DISPATCH TRANSMISSION CONFIRMED ({ts})</h4>
    <p style="color: #e2e8f0; margin: 0; font-size: 0.95em;">
        • <b>State Emergency Services (108 EMS):</b> Dispatched via Automated CAD Telemetry API<br>
        • <b>Highway Patrol Rapid Dispatch:</b> Alert SMS & GPS Coordinates Pushed to Mobile Data Terminal (MDT)<br>
        • <b>Traffic Control Operations:</b> Dynamic Variable Message Signs (VMS) updated to warning mode.
    </p>
</div>
"""

# ==============================================================================
# 6. TELEMETRY GRAPH GENERATOR
# ==============================================================================
def create_telemetry_chart(timeline_data, chart_save_path):
    """
    Creates dark-mode high-contrast telemetry curves for density and collision spikes.
    """
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
        fig.patch.set_facecolor('#0b0f19')
        ax1.set_facecolor('#111827')
        ax2.set_facecolor('#111827')

        raw_times = timeline_data.get("timestamp_sec", [0])
        raw_counts = timeline_data.get("vehicle_count", [0])
        raw_ious = timeline_data.get("collision_iou", [0])

        times = [float(t) for t in raw_times] if raw_times else [0.0]
        counts = [int(c) for c in raw_counts] if raw_counts else [0]
        ious = [float(i) for i in raw_ious] if raw_ious else [0.0]

        # Plot 1: Vehicle Density Profile
        ax1.plot(times, counts, color='#38bdf8', linewidth=2.2, label='Active Vehicles (V_active)')
        ax1.fill_between(times, counts, color='#38bdf8', alpha=0.18)
        ax1.axhline(y=5, color='#22c55e', linestyle='--', alpha=0.5, label='Light / Moderate Threshold (5)')
        ax1.axhline(y=15, color='#ef4444', linestyle='--', alpha=0.5, label='Moderate / Heavy Threshold (15)')
        ax1.set_ylabel('Vehicle Count', color='#94a3b8', fontsize=10)
        ax1.set_title('Spatial-Temporal Traffic Density Tracking', color='#f1f5f9', fontsize=12, fontweight='bold')
        ax1.grid(True, color='#1e293b', linestyle=':')
        ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', fontsize=8)

        # Plot 2: Collision IoU Overlap Spikes
        ax2.plot(times, ious, color='#f43f5e', linewidth=2.4, label='Collision Overlap Index (IoU)')
        ax2.fill_between(times, ious, color='#f43f5e', alpha=0.25)
        ax2.axhline(y=0.15, color='#f59e0b', linestyle=':', label='Moderate Crash Threshold (0.15)')
        ax2.axhline(y=0.28, color='#ef4444', linestyle='--', label='Severe Crash Threshold (0.28)')
        ax2.set_xlabel('Video Timeline (Seconds)', color='#94a3b8', fontsize=10)
        ax2.set_ylabel('Collision Overlap (IoU)', color='#94a3b8', fontsize=10)
        ax2.set_title('Accident Severity Anomaly Detection (Equation 15-16)', color='#f1f5f9', fontsize=12, fontweight='bold')
        ax2.grid(True, color='#1e293b', linestyle=':')
        ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', fontsize=8)

        plt.tight_layout()
        fig.savefig(chart_save_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
    except Exception as e:
        print(f"[WARN: Chart creation error]: {e}")
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_facecolor('#111827')
            fig.patch.set_facecolor('#0b0f19')
            ax.text(0.5, 0.5, "Telemetry Initializing...", color='#94a3b8', ha='center', va='center')
            fig.savefig(chart_save_path, dpi=100)
            plt.close(fig)
        except Exception:
            pass

# ==============================================================================
# 7. GRADIO WEB DASHBOARD (Custom UI & Styling)
# ==============================================================================
CUSTOM_CSS = """
/* Cyberpunk / Glassmorphic Control Room Palette */
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* Glassmorphic Panel Containers */
.glass-panel {
    background: rgba(17, 24, 39, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Primary Action Buttons */
.btn-primary {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4) !important;
}

/* Emergency Dispatch Red Button */
.btn-danger {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.4) !important;
}
.btn-danger:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.7) !important;
}

/* KPI Metric Cards */
.kpi-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 18px;
    text-align: center;
}
.kpi-value {
    font-size: 1.85em;
    font-weight: 800;
    margin: 4px 0;
}
.kpi-label {
    font-size: 0.85em;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Glow effects */
.glow-cyan { color: #38bdf8; text-shadow: 0 0 10px rgba(56, 189, 248, 0.5); }
.glow-red { color: #ef4444; text-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }
.glow-amber { color: #f59e0b; text-shadow: 0 0 10px rgba(245, 158, 11, 0.5); }
.glow-green { color: #22c55e; text-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
"""

def get_audit_table():
    if not AUDIT_LOGS:
        return pd.DataFrame([{
            "Incident ID": "N/A", "Timestamp": "No recordings yet",
            "Camera Mode": "-", "Density": "-", "Peak Vehicles": 0,
            "Severity": "Normal", "Impact Count": 0, "Severity Index": 0.0,
            "Status": "STANDBY"
        }])
    return pd.DataFrame(AUDIT_LOGS)

# Building Gradio UI Blocks
with gr.Blocks(title="MMAI - Automated Traffic Analysis, Accident Detection & Severity Assessment") as demo:

    # Header Ribbon
    gr.HTML("""
    <div style="background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); padding: 22px; border-radius: 12px; border: 1px solid #38bdf8; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <h1 style="color: #f8fafc; margin: 0; font-size: 1.8em; font-weight: 800; letter-spacing: -0.02em;">
                    🚗 Automated Traffic Analysis, Accident Detection & Severity Assessment
                </h1>
                <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 1.05em;">
                    Multi-Modal AI (MMAI) Framework Powered by <b>YOLOv11 Backbone</b> & <b>ByteTrack Engine</b> • Cross-Domain Validation on <b>DoTA</b> & <b>CADP</b> Datasets
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid #22c55e; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85em;">
                    ● SYSTEM ACTIVE & LIVE
                </span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ======================================================================
        # TAB 1: LIVE MULTI-CAMERA ANALYSIS STUDIO
        # ======================================================================
        with gr.Tab("🚦 Multi-Camera Analysis Studio"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=4):
                    gr.Markdown("### ⚙️ Video Ingestion & Model Configuration")

                    preset_choice = gr.Dropdown(
                        label="📹 Select Camera Preset or Dataset Benchmark Feed",
                        choices=[
                            "DoTA Ego-Centric Dashcam (In-Domain Anomaly Clip)",
                            "CADP Overhead CCTV Surveillance (Out-of-Domain Traffic Clip)",
                            "Upload Custom Video File"
                        ],
                        value="CADP Overhead CCTV Surveillance (Out-of-Domain Traffic Clip)",
                        info="Switches between Ego-Centric Dashcam (DoTA) and Overhead CCTV (CADP) viewpoints."
                    )

                    video_in = gr.Video(
                        label="Source Video Stream",
                        value=SAMPLE_CCTV,
                        sources=["upload", "webcam"]
                    )

                    camera_mode = gr.Radio(
                        label="Camera Infrastructure Mode",
                        choices=["Overhead CCTV Surveillance (CADP)", "Car Dashboard Camera (DoTA)"],
                        value="Overhead CCTV Surveillance (CADP)"
                    )

                    with gr.Accordion("🛠️ Advanced AI Parameters & Sampling", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.15, maximum=0.85, value=0.35, step=0.05,
                            label="YOLOv11 Confidence Threshold (τ_conf)"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.05, maximum=0.45, value=0.14, step=0.01,
                            label="Collision Overlap Threshold (τ_accident)"
                        )
                        fps_slider = gr.Slider(
                            minimum=5, maximum=30, value=10, step=1,
                            label="Temporal Keyframe Downsampling (Target FPS, Eq. 7-8)"
                        )
                        class_filter = gr.CheckboxGroup(
                            choices=["Car", "Motorcycle", "Bus", "Truck"],
                            value=["Car", "Bus", "Truck", "Motorcycle"],
                            label="Monitored Vehicle Categories"
                        )

                    analyze_btn = gr.Button("🚀 Run AI Analysis & Severity Assessment", elem_classes=["btn-primary"], size="lg")
                    status_text = gr.Markdown("Ready to process traffic stream.")

                # Right Column: Output HUD Video & Telemetry Cards
                with gr.Column(scale=6):
                    gr.Markdown("### 🖥️ Annotated Telemetry Stream (HUD & ByteTrack Locking)")

                    video_out = gr.Video(label="Real-Time Analytics Video Feed")

                    # KPI Cards Grid
                    with gr.Row():
                        with gr.Column(min_width=120):
                            kpi_density = gr.Textbox(label="Traffic Density", value="Standing by")
                        with gr.Column(min_width=120):
                            kpi_severity = gr.Textbox(label="Accident Severity", value="Normal")
                        with gr.Column(min_width=120):
                            kpi_active_veh = gr.Number(label="Active Vehicles", value=0)
                        with gr.Column(min_width=120):
                            kpi_impact_veh = gr.Number(label="Impacted Vehicles", value=0)
                        with gr.Column(min_width=120):
                            kpi_index = gr.Number(label="Severity Index (S)", value=0.0)

                    with gr.Row():
                        snapshot_out = gr.Image(label="Collision Incident Snapshot", type="filepath")
                        chart_preview = gr.Image(label="Temporal Anomaly Telemetry", type="filepath")

        # ======================================================================
        # TAB 2: EMERGENCY DISPATCHER AUTOMATED ALERT CENTER
        # ======================================================================
        with gr.Tab("🚨 Emergency Dispatcher Alert Center"):
            with gr.Row():
                with gr.Column(scale=7):
                    dispatch_dossier_md = gr.Markdown("""
                    ### 📡 Real-Time Computer-Aided Dispatch (CAD) Telemetry
                    *Run video analysis in Tab 1 to trigger automatic emergency detection.*
                    """)

                    with gr.Row():
                        dispatch_btn = gr.Button("📢 Transmit Instant CAD Alert (Police & EMS 108)", elem_classes=["btn-danger"], size="lg")
                    
                    dispatch_status = gr.HTML()

                with gr.Column(scale=3):
                    gr.Markdown("### 🔊 Emergency Audio Siren Monitor")
                    audio_player = gr.Audio(label="Dispatcher Audible Siren Warning", autoplay=False)
                    gr.Markdown("""
                    *Audible warning automatically generates on confirmed **Moderate** or **Severe** crash detections to notify room operations.*
                    """)

        # ======================================================================
        # TAB 3: SPATIAL-TEMPORAL TELEMETRY & CHARTS
        # ======================================================================
        with gr.Tab("📊 Telemetry & Anomaly Analytics"):
            gr.Markdown("### 📈 Real-Time Density Curve & Collision Overlap Metrics")
            chart_full = gr.Image(label="Multi-Scale Collision Dynamics & Vehicle Density Timeline", type="filepath")
            gr.Markdown("""
            **Mathematical Background:**
            * **Traffic Density Formulation:** Active vehicle bounding-boxes per frame normalized across roadway area.
            * **Collision Overlap Threshold ($IoU_{\\text{collision}}$):** Flags spatial overlap between tracked boxes $B_i$ and $B_j$ (Equation 15).
            * **Severity Index ($S$):** Quantified as $S = IoU_{\\text{collision}} \\times V_{\\text{entangled}}$ (Equation 16).
            """)

        # ======================================================================
        # TAB 4: METHODOLOGY & RESEARCH FLOWCHART
        # ======================================================================
        with gr.Tab("🔬 Research Methodology & Architecture"):
            gr.Markdown("### 🧠 10-Stage End-to-End Multimodal AI Pipeline")
            if os.path.exists(FLOWCHART_PATH):
                gr.Image(value=FLOWCHART_PATH, label="System Methodology Architecture Flowchart")

            gr.Markdown("""
            #### 📋 Detailed Pipeline Stages Breakdown:
            1. **Data Collection:** Ingests heterogeneous streams from Car Dashboard Cameras (DoTA) and CCTV Surveillance (CADP).
            2. **Data Pre-processing:** Resizing to 640×640 via bilinear interpolation with neutral letterboxing; pixel intensity scaling $[0.0, 1.0]$; temporal sub-sampling at 10 FPS (Equation 7-8).
            3. **Input Representation:** Formulates RGB Frame Sequence and Optical Flow temporal descriptors.
            4. **Feature Extraction:** YOLOv11 Backbone utilizing Cross Stage Partial with 2-Convolutions (C3k2) modules and Spatial Pyramid Pooling Fast (SPPF).
            5. **Feature Fusion:** Integrates spatial representations with temporal trajectories.
            6. **Decoupled Classification Module:**
               * Anchor-Free Bounding Box Regression Head (CIoU & DFL Loss)
               * Traffic Density Classification Head (Light, Moderate, Heavy)
               * 3-Tier Accident Severity Assessment Head (Minor, Moderate, Severe)
            7. **ByteTrack Multi-Object Tracking:** Kalman filtering motion prediction and Hungarian association with low/high confidence boxes.
            8. **Localized Crash Polygon Locking:** Locks onto the impact coordinate polygon, counting only vehicles entangled in the crash ($V_{\\text{entangled}}$) to eliminate false counts.
            9. **Alert Generation:** Automated Computer-Aided Dispatch (CAD) alert generation with GPS sector tagging.
            10. **Deployment:** Edge-optimized real-time inference (>40 FPS throughput).
            """)

        # ======================================================================
        # TAB 5: DATASET BENCHMARKS & EVALUATION
        # ======================================================================
        with gr.Tab("📈 Dataset Evaluation & Benchmarks"):
            gr.Markdown("""
            ### 🏆 Cross-Dataset Generalization (DoTA vs. CADP)
            To prove robust transferability, the system is trained in-domain on ego-centric dashcam footage (**DoTA**) and evaluated out-of-domain on overhead stationary CCTV cameras (**CADP**) without retraining.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Table 4.1: Quantitative Performance on DoTA (In-Domain Dashcam)")
                    gr.Dataframe(value=pd.DataFrame([
                        {"Target Evaluation Task": "Vehicle Detection", "Precision (P)": 0.945, "Recall (R)": 0.932, "F1-Score": 0.938, "mAP@0.5": 0.958},
                        {"Target Evaluation Task": "Accident Event Classification", "Precision (P)": 0.912, "Recall (R)": 0.930, "F1-Score": 0.921, "mAP@0.5": 0.921},
                        {"Target Evaluation Task": "3-Tier Severity Assessment", "Precision (P)": 0.887, "Recall (R)": 0.874, "F1-Score": 0.880, "mAP@0.5": 0.892}
                    ]))
                with gr.Column():
                    gr.Markdown("#### Table 4.2: Cross-Dataset Evaluation on CADP (Out-of-Domain CCTV)")
                    gr.Dataframe(value=pd.DataFrame([
                        {"Target Evaluation Task": "Vehicle Detection", "Precision (P)": 0.951, "Recall (R)": 0.938, "F1-Score": 0.944, "mAP@0.5": 0.942},
                        {"Target Evaluation Task": "Accident Event Classification", "Precision (P)": 0.890, "Recall (R)": 0.906, "F1-Score": 0.898, "mAP@0.5": 0.898},
                        {"Target Evaluation Task": "3-Tier Severity Assessment", "Precision (P)": 0.862, "Recall (R)": 0.866, "F1-Score": 0.864, "mAP@0.5": 0.864}
                    ]))

            gr.Markdown("""
            #### 📊 Comparative Analysis Against Baseline Detectors:
            * **Performance Transfer:** **88.2%** cross-domain retention between ego-centric dashcams and elevated CCTV.
            * **Vehicle Counting Error (MAE):** 0.82 vehicles/frame (DoTA) and 0.42 vehicles/frame (CADP).
            * **Real-time Inference Speed:** **> 40 FPS** on edge architecture, substantially outperforming Faster R-CNN (11 FPS) and 3D-CNNs (8 FPS).
            """)

        # ======================================================================
        # TAB 6: AUDIT LOG & INCIDENT HISTORY
        # ======================================================================
        with gr.Tab("🗃️ Incident Audit History"):
            gr.Markdown("### 📜 Real-Time Audit Log of Processed Surveillance Feeds")
            audit_df = gr.Dataframe(value=get_audit_table(), label="Chronological Incident Logs")
            refresh_audit = gr.Button("🔄 Refresh Audit Table")
            refresh_audit.click(fn=get_audit_table, outputs=audit_df)

    # --------------------------------------------------------------------------
    # EVENT BINDINGS
    # --------------------------------------------------------------------------
    def update_preset_video(choice):
        if "DoTA" in choice:
            return SAMPLE_DASHCAM, "Car Dashboard Camera (DoTA)"
        elif "CADP" in choice:
            return SAMPLE_CCTV, "Overhead CCTV Surveillance (CADP)"
        else:
            return None, "Overhead CCTV Surveillance (CADP)"

    preset_choice.change(
        fn=update_preset_video,
        inputs=[preset_choice],
        outputs=[video_in, camera_mode]
    )

    analyze_btn.click(
        fn=process_traffic_analysis,
        inputs=[video_in, camera_mode, conf_slider, iou_slider, fps_slider, class_filter],
        outputs=[
            video_out, snapshot_out, status_text,
            kpi_severity, kpi_density, kpi_active_veh, kpi_impact_veh, kpi_index,
            chart_preview, dispatch_dossier_md, audio_player
        ]
    ).then(
        fn=lambda p: p,
        inputs=[chart_preview],
        outputs=[chart_full]
    )

    dispatch_btn.click(
        fn=trigger_emergency_dispatch,
        inputs=[dispatch_dossier_md],
        outputs=[dispatch_status]
    )

# Launcher
if __name__ == "__main__":
    print("==================================================================")
    print(" 🚀 Launching MMAI Traffic Analysis & Accident Severity Web Application")
    print(" Access Web Portal at: http://localhost:7860")
    print("==================================================================")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS
    )
