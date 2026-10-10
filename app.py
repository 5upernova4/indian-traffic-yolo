"""Vehicle Insights - Flask web app for vehicle detection, tracking and counting.

Upload an image or a video; inference runs in a background thread while the page
polls /progress/<run_id>. Videos are tracked with ByteTrack and vehicles are counted
as they cross a virtual line. Results: annotated media, per-class counts, a density
heatmap (videos) and CSV / JSON downloads.

    python app.py                      # http://127.0.0.1:5000
    TRAFFIC_YOLO_WEIGHTS=weights/yolo11s_uvh26.onnx python app.py
"""

import os
import threading
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

from traffic_yolo.classes import CLASS_COLORS_BGR, CLASS_NAMES, DISPLAY_NAMES
from traffic_yolo.pipeline import IMAGE_EXTS, VIDEO_EXTS, Detector, default_lines, process_image, process_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

MODEL_PATH = os.environ.get('TRAFFIC_YOLO_WEIGHTS', 'weights/yolo11s_uvh26.pt')

# ── Inference tuning ─────────────────────────────────────────────────────────
# Run the model on every Nth video frame (1 = every frame). The tracker keeps
# identities across skipped frames; raise this for long videos on slow machines.
FRAME_STRIDE = int(os.environ.get('FRAME_STRIDE', '1'))

try:
    detector = Detector(MODEL_PATH)
    detector.warmup()
    print(f"Model loaded: {MODEL_PATH} (imgsz={detector.imgsz}, device={detector.device})")
except Exception as e:
    detector = None
    print(f"Could not load model ({MODEL_PATH}): {e}")

# One job at a time: the tracker is stateful and the GPU is shared.
inference_lock = threading.Lock()

# ── Progress tracking store (keyed by run_id) ────────────────────────────────
progress_store = {}


def _hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


CLASS_META = [{'id': n, 'label': DISPLAY_NAMES[n], 'color': _hex(CLASS_COLORS_BGR[n])} for n in CLASS_NAMES]


# ── Background worker: runs inference + updates progress_store ───────────────
def process_file(run_id, filepath, out_dir, is_video, line_cfg):
    base = f"/static/results/{run_id}"
    try:
        if inference_lock.locked():
            progress_store[run_id].update({'message': 'Waiting for the previous job to finish…'})
        with inference_lock:
            if is_video:
                import cv2
                cap = cv2.VideoCapture(filepath)
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                lines = default_lines(w, h, line_cfg['position'], line_cfg['orientation'])
                stride_txt = 'every frame' if FRAME_STRIDE == 1 else f'every {FRAME_STRIDE} frames'

                def on_progress(i, n):
                    progress_store[run_id].update({
                        'progress': i, 'total': n,
                        'message': f'Tracking and counting vehicles (model runs on {stride_txt})…',
                    })

                summary = process_video(filepath, out_dir, detector, lines=lines, stride=FRAME_STRIDE,
                                        progress=on_progress)
                result = {
                    'type': 'video',
                    'url': f"{base}/annotated.mp4",
                    'heatmap_url': f"{base}/heatmap.jpg",
                    'crossings_csv_url': f"{base}/crossings.csv",
                }
            else:
                progress_store[run_id].update({'total': 1, 'message': 'Analyzing image…'})
                summary = process_image(filepath, out_dir, detector)
                result = {'type': 'image', 'url': f"{base}/annotated.jpg"}

        result.update({
            'status': 'done',
            'csv_url': f"{base}/detections.csv",
            'json_url': f"{base}/summary.json",
            'summary': summary,
            'classes': CLASS_META,
        })
        progress_store[run_id] = result
    except Exception as e:
        traceback.print_exc()
        progress_store[run_id] = {'status': 'error', 'error': f'Inference error: {e}'}
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'model': MODEL_PATH, 'loaded': detector is not None,
                    'device': getattr(detector, 'device', None), 'classes': CLASS_NAMES})


@app.route('/upload', methods=['POST'])
def upload():
    """Accept the file, kick off background inference, return run_id immediately."""
    if not detector:
        return jsonify({'error': f'Model not found. Expected weights at {MODEL_PATH}.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        return jsonify({'error': f'Unsupported file type "{ext}". Upload an image or a video.'}), 400

    try:
        position = min(max(float(request.form.get('line_position', 0.6)), 0.05), 0.95)
    except ValueError:
        position = 0.6
    orientation = request.form.get('line_orientation', 'horizontal')
    if orientation not in ('horizontal', 'vertical'):
        orientation = 'horizontal'

    run_id = uuid.uuid4().hex
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{run_id}{ext}")
    file.save(filepath)

    out_dir = os.path.join(os.path.abspath(app.config['RESULT_FOLDER']), run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Seed the progress store so the first poll never 404s
    progress_store[run_id] = {
        'status': 'processing',
        'progress': 0,
        'total': 1,
        'message': 'Upload complete. Starting inference…',
    }

    # Launch processing in a daemon thread so /upload returns instantly
    thread = threading.Thread(
        target=process_file,
        args=(run_id, filepath, out_dir, ext in VIDEO_EXTS, {'position': position, 'orientation': orientation}),
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
    app.run(debug=False, host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 5000)),
            threaded=True)
