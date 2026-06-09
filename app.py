import os
import uuid
import glob
import csv
import subprocess
import threading
import shutil
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tqdm import tqdm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

MODEL_PATH = "new.onnx"

# ── Inference tuning ─────────────────────────────────────────────────────────
# Process every Nth frame.  Skipped frames reuse the last set of detections.
# 1 = every frame (slow), 3 = every 3rd frame (~3× speed-up), etc.
FRAME_SKIP = 3

# ── Class names from your training notebook (8-class vehicle model) ──────────
CLASS_NAMES = {
    0: 'auto',
    1: 'bus',
    2: 'car',
    3: 'lcv',
    4: 'motorcycle',
    5: 'multiaxle',
    6: 'tractor',
    7: 'truck',
}

# Distinct BGR colours per class for bounding boxes
CLASS_COLORS = {
    0: (0,   165, 255),   # auto       → orange
    1: (255,   0,   0),   # bus        → blue
    2: (0,   255,   0),   # car        → green
    3: (128,   0, 128),   # lcv        → purple
    4: (0,   255, 255),   # motorcycle → yellow
    5: (0,   128, 255),   # multiaxle  → light-blue
    6: (255, 255,   0),   # tractor    → cyan
    7: (0,     0, 255),   # truck      → red
}

try:
    model = YOLO(MODEL_PATH, task='detect')
    # Inject class names so r.plot() always has them,
    # even when ONNX metadata is missing
    if not model.names or len(model.names) == 0:
        model.names = CLASS_NAMES
    print(f"✅ Model loaded. Classes: {model.names}")
except Exception as e:
    model = None
    print(f"⚠️  Could not load model ({MODEL_PATH}): {e}")


# ── Progress tracking store (keyed by run_id) ────────────────────────────────
progress_store = {}


# ── Helper: draw boxes manually (fallback / guaranteed path) ─────────────────
def draw_boxes(image: np.ndarray, boxes, names: dict, orig_dims=None, infer_size=416) -> np.ndarray:
    """Draw bounding boxes + labels directly on a copy of *image*.

    Works regardless of whether r.plot() is functioning correctly,
    because it reads raw xyxy tensors and paints them with OpenCV.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for box in boxes:
        cls_id  = int(box.cls[0].item())
        conf    = float(box.conf[0].item())
        
        raw_xyxy = box.xyxy[0].tolist()
        if orig_dims:
            orig_w, orig_h = orig_dims
            # Use raw_xyxy values to reconstruct size.
            # If coordinates are extremely small (normalized), scale relative to original.
            if max(raw_xyxy) <= 2.0:
                scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
            else:
                scaled_xyxy = [raw_xyxy[0]*orig_w/infer_size, raw_xyxy[1]*orig_h/infer_size, raw_xyxy[2]*orig_w/infer_size, raw_xyxy[3]*orig_h/infer_size]
            x1, y1, x2, y2 = [int(v) for v in scaled_xyxy]
        else:
            x1, y1, x2, y2 = [int(v) for v in raw_xyxy]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color     = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label     = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        thickness = max(2, int(min(w, h) / 300))   # scale with image size
        font_scale = max(0.4, min(w, h) / 1000)

        # Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            img,
            (x1, label_y1),
            (x1 + tw + 4, label_y1 + th + baseline + 4),
            color, -1
        )

        # Label text (white)
        cv2.putText(
            img, label,
            (x1 + 2, label_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

    return img


# ── Video writer helpers ─────────────────────────────────────────────────────
def _try_h264_writer(path, fps, size):
    """Try several H.264 fourcc codes that OpenCV may support.
    Returns an opened VideoWriter, or None.
    """
    for codec in ('avc1', 'H264', 'X264'):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, int(fps), size)
            if writer.isOpened():
                print(f"  ✅ Using native H.264 codec '{codec}' — no ffmpeg needed")
                return writer
            writer.release()
        except Exception:
            pass
    return None


def _reencode_with_ffmpeg(src, dst):
    """Re-encode *src* to H.264 at *dst*.  Returns True on success."""
    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        print("  ⚠️  ffmpeg not found in PATH — video may not play in browser")
        return False
    cmd = [
        ffmpeg_bin, '-y', '-i', src,
        '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',          # required for browser compatibility
        '-movflags', '+faststart',      # allows streaming before full download
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"  ⚠️  ffmpeg re-encode failed: {e}")
        return False


# ── Background worker: runs inference + updates progress_store ───────────────
def process_file(run_id, filepath, ext, out_dir, csv_path, is_video, names):
    """Run YOLO inference in a background thread.

    Updates progress_store[run_id] so the /progress endpoint can relay
    frame-level progress to the frontend, and tqdm prints a console bar.
    """
    try:
        progress_store[run_id] = {
            'status': 'processing',
            'progress': 0,
            'total': 1,
            'message': 'Starting analysis…',
        }

        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])

            if is_video:
                cap    = cv2.VideoCapture(filepath)
                fps    = cap.get(cv2.CAP_PROP_FPS)
                fps    = fps if fps and fps == fps else 25.0   # guard NaN/0
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames = max(total_frames, 1)

                # ── Decide codec: try H.264 first, fall back to mp4v ─────
                final_path = os.path.join(out_dir, f"{run_id}.mp4")
                raw_path   = os.path.join(out_dir, f"{run_id}_raw.mp4")

                h264_writer = _try_h264_writer(final_path, fps, (orig_w, orig_h))
                if h264_writer is not None:
                    vid_writer     = h264_writer
                    out_media      = final_path
                    needs_reencode = False
                else:
                    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(raw_path, fourcc, int(fps), (orig_w, orig_h))
                    out_media      = raw_path
                    needs_reencode = True

                infer_count = (total_frames + FRAME_SKIP - 1) // FRAME_SKIP
                progress_store[run_id].update({
                    'total': total_frames,
                    'message': f'Processing {total_frames} frames ({infer_count} inferences, skip={FRAME_SKIP})…',
                })

                # ── tqdm console bar ─────────────────────────────────────
                pbar = tqdm(
                    total=total_frames,
                    desc=f"[{run_id[:8]}] Inferencing (skip={FRAME_SKIP})",
                    unit="frame",
                    dynamic_ncols=True,
                )

                frame_id = 0
                last_boxes = None        # reuse for skipped frames

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run YOLO only on key frames
                    if frame_id % FRAME_SKIP == 0:
                        frame_resized = cv2.resize(frame, (416, 416))
                        r = model.predict(source=frame_resized, imgsz=416, verbose=False)[0]
                        last_boxes = r.boxes

                        # Write CSV for key frames only
                        for box in last_boxes:
                            cls_id   = int(box.cls[0].item())
                            cls_name = names.get(cls_id, str(cls_id))
                            conf     = float(box.conf[0].item())

                            raw_xyxy = box.xyxy[0].tolist()
                            if max(raw_xyxy) <= 2.0:
                                scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
                            else:
                                scaled_xyxy = [raw_xyxy[0]*orig_w/416.0, raw_xyxy[1]*orig_h/416.0, raw_xyxy[2]*orig_w/416.0, raw_xyxy[3]*orig_h/416.0]
                            x1, y1, x2, y2 = [int(max(0, v)) for v in scaled_xyxy]
                            csv_writer.writerow(
                                [frame_id, cls_name, f"{conf:.2f}", x1, y1, x2, y2]
                            )

                    # Draw boxes on every frame (fresh or reused)
                    if last_boxes is not None and len(last_boxes) > 0:
                        annotated = draw_boxes(frame, last_boxes, names,
                                               orig_dims=(orig_w, orig_h), infer_size=416)
                    else:
                        annotated = frame

                    vid_writer.write(annotated)

                    frame_id += 1
                    pbar.update(1)
                    progress_store[run_id]['progress'] = frame_id

                pbar.close()
                cap.release()
                vid_writer.release()

                # ── Re-encode to H.264 if OpenCV couldn't do it natively ─
                if needs_reencode:
                    progress_store[run_id]['message'] = 'Encoding video for browser…'
                    if _reencode_with_ffmpeg(raw_path, final_path):
                        os.remove(raw_path)
                        saved_media = final_path
                    else:
                        # ffmpeg unavailable/failed — rename raw → final
                        # The browser may not play it, but the download will work
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        os.rename(raw_path, final_path)
                        saved_media = final_path
                else:
                    saved_media = out_media

            else:
                # ── Single image inference ────────────────────────────────
                out_media  = os.path.join(out_dir, f"{run_id}.jpg")
                frame = cv2.imread(filepath)
                orig_h, orig_w = frame.shape[:2]

                progress_store[run_id].update({
                    'total': 1,
                    'message': 'Analyzing image…',
                })

                frame_resized = cv2.resize(frame, (416, 416))
                r = model.predict(source=frame_resized, imgsz=416, verbose=False)[0]

                annotated = draw_boxes(frame, r.boxes, names, orig_dims=(orig_w, orig_h), infer_size=416)
                cv2.imwrite(out_media, annotated)

                for box in r.boxes:
                    cls_id   = int(box.cls[0].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    conf     = float(box.conf[0].item())

                    raw_xyxy = box.xyxy[0].tolist()
                    if max(raw_xyxy) <= 2.0:
                        scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
                    else:
                        scaled_xyxy = [raw_xyxy[0]*orig_w/416.0, raw_xyxy[1]*orig_h/416.0, raw_xyxy[2]*orig_w/416.0, raw_xyxy[3]*orig_h/416.0]
                    x1, y1, x2, y2 = [int(max(0, v)) for v in scaled_xyxy]
                    csv_writer.writerow(
                        [0, cls_name, f"{conf:.2f}", x1, y1, x2, y2]
                    )

                progress_store[run_id]['progress'] = 1
                saved_media = out_media

        if not os.path.exists(saved_media):
            progress_store[run_id] = {
                'status': 'error',
                'error': 'Inference produced no output file.',
            }
            return

        result_file = os.path.basename(saved_media)
        file_type   = "video" if is_video else "image"

        # ── Mark as done — the frontend will pick this up on its next poll
        progress_store[run_id] = {
            'status': 'done',
            'url':     f"/static/results/{run_id}/{result_file}",
            'csv_url': f"/static/results/{run_id}/detections.csv",
            'type':    file_type,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_store[run_id] = {
            'status': 'error',
            'error': f'Inference error: {str(e)}',
        }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Accept the file, kick off background inference, return run_id immediately."""
    if not model:
        return jsonify({'error': f'Model not found. Place {MODEL_PATH} in the root directory.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext      = file.filename.rsplit('.', 1)[-1].lower()
    run_id   = uuid.uuid4().hex
    filename = f"{run_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    project_path = os.path.abspath(app.config['RESULT_FOLDER'])
    out_dir      = os.path.join(project_path, run_id)
    os.makedirs(out_dir, exist_ok=True)

    csv_path  = os.path.join(out_dir, "detections.csv")
    is_video  = ext in ['mp4', 'avi', 'mov', 'mkv']

    # Use the model's name dict (patched at startup if needed)
    names = model.names or CLASS_NAMES

    # Seed the progress store so the first poll never 404s
    progress_store[run_id] = {
        'status': 'processing',
        'progress': 0,
        'total': 1,
        'message': 'Uploading complete. Starting inference…',
    }

    # Launch processing in a daemon thread so /upload returns instantly
    thread = threading.Thread(
        target=process_file,
        args=(run_id, filepath, ext, out_dir, csv_path, is_video, names),
        daemon=True,
    )
    thread.start()

    return jsonify({'run_id': run_id})


@app.route('/progress/<run_id>')
def progress(run_id):
    """Lightweight polling endpoint — returns the current inference state."""
    data = progress_store.get(run_id)
    if data is None:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)