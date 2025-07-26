import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance
from datetime import datetime, timedelta
import mediapipe as mp
from flask import Flask, Response, render_template, send_file, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import os
import threading
import queue
import time
from math import atan2, degrees
import sqlite3
import bcrypt
import uuid
from fpdf import FPDF
import shutil
from ratelimit import limits, sleep_and_retry
from typing import List, Dict, Any
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
socketio = SocketIO(app)

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.3,
    model_selection=0
)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Load Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Directories
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)

# SQLite Database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# Dlib shape to numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

# Eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Thresholds (made accessible globally with lock)
EAR_THRESHOLD = [0.22]
POSTURE_THRESHOLD = [30]
SLEEP_THRESHOLD = 5
threshold_lock = threading.Lock()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Tracking storage
tracked_faces = {}
frame_queue = queue.Queue(maxsize=1)

def calculate_posture_angle(landmarks):
    if not landmarks:
        return None
    try:
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_x, hip_y = left_hip.x, left_hip.y
        angle = degrees(atan2(hip_y - shoulder_mid[1], hip_x - shoulder_mid[0]))
        return angle
    except:
        return None

def non_max_suppression(boxes: List[Dict[str, Any]], overlapThresh: float = 0.3) -> List[Dict[str, Any]]:
    if len(boxes) == 0:
        return []
    pick = []
    x1 = np.array([b['x'] for b in boxes])
    y1 = np.array([b['y'] for b in boxes])
    x2 = np.array([b['x'] + b['w'] for b in boxes])
    y2 = np.array([b['y'] + b['h'] for b in boxes])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort([b['ear'] for b in boxes])

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return [boxes[i] for i in pick]

def generate_heatmap(frame, faces):
    if not faces or len(faces) % 10 == 0:
        heatmap = np.zeros_like(frame)
        h, w = frame.shape[:2]
        for face in faces:
            x, y, fw, fh = face['x'], face['y'], face['w'], face['h']
            center = (x + fw // 2, y + fh // 2)
            for i in range(h):
                for j in range(w):
                    dist = distance.euclidean((j, i), center)
                    if dist < 50:
                        heatmap[i, j] += np.array([0, 50, 0], dtype=np.uint8)
        return cv2.addWeighted(frame, 0.8, heatmap, 0.2, 0)
    return frame

def cleanup_screenshots():
    now = datetime.now()
    for f in os.listdir("static/images"):
        path = os.path.join("static/images", f)
        if os.path.isfile(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if now - mtime > timedelta(days=7):
                os.remove(path)

def process_frame():
    global cap, tracked_faces, EAR_THRESHOLD, POSTURE_THRESHOLD
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 2 == 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Primary detection with Mediapipe
        results = face_detection.process(rgb_frame)
        faces_dlib = []
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                faces_dlib.append(dlib.rectangle(x, y, x + w_box, y + h_box))
        else:
            # Fallback to Dlib if Mediapipe fails
            faces_dlib = detector(gray, 1)

        current_faces = []
        pose_results = pose.process(rgb_frame)
        posture_angle = calculate_posture_angle(pose_results.pose_landmarks) if pose_results.pose_landmarks else None

        for face in faces_dlib:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            center = (x + w // 2, y + h // 2)
            try:
                landmarks = predictor(gray, face)
                landmarks_np = shape_to_np(landmarks)
                left_eye = [landmarks_np[i] for i in LEFT_EYE]
                right_eye = [landmarks_np[i] for i in RIGHT_EYE]
                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                face_id = None
                for fid, data in tracked_faces.items():
                    if distance.euclidean(center, data['center']) < 100:
                        face_id = fid
                        break
                if face_id is None:
                    face_id = str(uuid.uuid4())[:8]
                    tracked_faces[face_id] = {
                        'center': center,
                        'sleep_frames': 0,
                        'posture': 'Unknown',
                        'last_seen': datetime.now(),
                        'sleep_start': None,
                        'sleep_duration': 0
                    }

                with threshold_lock:
                    if avg_EAR < EAR_THRESHOLD[0]:
                        tracked_faces[face_id]['sleep_frames'] += 1
                        if tracked_faces[face_id]['sleep_frames'] >= SLEEP_THRESHOLD and not tracked_faces[face_id]['sleep_start']:
                            tracked_faces[face_id]['sleep_start'] = datetime.now()
                    else:
                        if tracked_faces[face_id]['sleep_start']:
                            duration = (datetime.now() - tracked_faces[face_id]['sleep_start']).total_seconds()
                            tracked_faces[face_id]['sleep_duration'] += duration
                            tracked_faces[face_id]['sleep_start'] = None
                        tracked_faces[face_id]['sleep_frames'] = max(0, tracked_faces[face_id]['sleep_frames'] - 1)

                sleep_frames = tracked_faces[face_id]['sleep_frames']
                tracked_faces[face_id]['center'] = center
                tracked_faces[face_id]['last_seen'] = datetime.now()
                with threshold_lock:
                    tracked_faces[face_id]['posture'] = "Slouching" if posture_angle and abs(posture_angle) > POSTURE_THRESHOLD[0] else "Upright"

                color = (0, 0, 255) if sleep_frames >= SLEEP_THRESHOLD else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {face_id} EAR: {avg_EAR:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Posture: {tracked_faces[face_id]['posture']}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                status = "Sleeping" if sleep_frames >= SLEEP_THRESHOLD else "Awake"
                current_faces.append({
                    'face_id': face_id,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'ear': avg_EAR,
                    'posture': tracked_faces[face_id]['posture'],
                    'status': status,
                    'sleep_duration': tracked_faces[face_id]['sleep_duration']
                })

                if sleep_frames >= SLEEP_THRESHOLD:
                    filename = f"static/images/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_face{face_id}.jpg"
                    cv2.imwrite(filename, frame)
                    data = {
                        "Timestamp": datetime.now(),
                        "Face_ID": face_id,
                        "Status": "Sleeping",
                        "Posture": tracked_faces[face_id]['posture'],
                        "Sleep_Duration": tracked_faces[face_id]['sleep_duration']
                    }
                    df = pd.DataFrame([data])
                    report_path = "static/reports/sleep_report.csv"
                    df.to_csv(report_path, mode="a", header=not os.path.exists(report_path), index=False)

            except Exception as e:
                logger.error(f"Error processing face: {e}")
                continue

        current_faces = non_max_suppression(current_faces)

        tracked_faces = {
            fid: data for fid, data in tracked_faces.items()
            if (datetime.now() - data['last_seen']).total_seconds() < 5
        }
        frame = generate_heatmap(frame, current_faces)
        if not frame_queue.full():
            frame_queue.put(frame)
        with threshold_lock:
            socketio.emit('status_update', {
                'faces': current_faces,
                'ear_threshold': EAR_THRESHOLD[0],
                'posture_threshold': POSTURE_THRESHOLD[0]
            })

@socketio.on('connect')
def handle_connect():
    if session.get('logged_in'):
        with threshold_lock:
            socketio.emit('status_update', {
                'faces': [],
                'ear_threshold': EAR_THRESHOLD[0],
                'posture_threshold': POSTURE_THRESHOLD[0]
            })

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', page='home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username or email already exists")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', page='home')

@app.route('/images')
def images():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', page='images')

@app.route('/live_feed')
def live_feed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', page='live_feed')

def generate_frames():
    frame_skip = 0
    while True:
        try:
            if frame_queue.qsize() > 1 and frame_skip < 2:
                frame_skip += 1
                continue
            frame_skip = 0
            frame = frame_queue.get(timeout=1)
            quality = 70 if frame_queue.qsize() < 1 else 50
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            continue

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@sleep_and_retry
@limits(calls=10, period=60)
@app.route('/screenshots')
def screenshots():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    cleanup_screenshots()
    screenshots = [f for f in os.listdir("static/images") if f.endswith('.jpg')]
    return jsonify({'screenshots': screenshots})

@app.route('/static/images/<filename>')
def serve_image(filename):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return send_file(os.path.join("static/images", filename))

@app.route('/delete_screenshot/<filename>', methods=['DELETE'])
def delete_screenshot(filename):
    if not session.get('logged_in'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    path = os.path.join("static/images", filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'status': 'success', 'message': 'Screenshot deleted'})
    return jsonify({'status': 'error', 'message': 'File not found'}), 404

@app.route('/download_report/<format>')
def download_report(format):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    report_path = "static/reports/sleep_report.csv"
    if not os.path.exists(report_path):
        return "No report available", 404
    if format == 'csv':
        return send_file(report_path, as_attachment=True)
    elif format == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Sleep Report", ln=True, align='C')
        df = pd.read_csv(report_path)
        for i, row in df.iterrows():
            pdf.cell(200, 10, txt=str(row), ln=True)
        pdf_path = "static/reports/sleep_report.pdf"
        pdf.output(pdf_path)
        return send_file(pdf_path, as_attachment=True)
    return "Invalid format", 400

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    global EAR_THRESHOLD, POSTURE_THRESHOLD
    data = request.get_json()
    try:
        ear = float(data['ear'])
        posture = float(data['posture'])
        if 0.1 <= ear <= 0.5 and 10 <= posture <= 60:
            with threshold_lock:
                EAR_THRESHOLD[0] = ear
                POSTURE_THRESHOLD[0] = posture
                logger.debug(f"Thresholds updated: EAR={ear}, Posture={posture}")
                socketio.emit('threshold_update', {'ear_threshold': ear, 'posture_threshold': posture})
            return jsonify({'status': 'success', 'ear_threshold': ear, 'posture_threshold': posture})
        return jsonify({'status': 'error', 'message': 'Invalid threshold values'}), 400
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

if __name__ == "__main__":
    threading.Thread(target=process_frame, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)