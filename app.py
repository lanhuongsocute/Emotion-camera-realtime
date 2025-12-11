import os
import time
import csv
import datetime
import random
import cv2
import warnings

# Tắt cảnh báo hệ thống và TensorFlow
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request
from deepface import DeepFace

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

app = Flask(__name__)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("Không thể mở webcam. Vui lòng kiểm tra thiết bị!")

# Biến toàn cục
current_emotion = {
    "label": "🤔",
    "description": "Đang phân tích khuôn mặt của bạn..."
}
alert_mode = False
sad_start_time = None
SAD_THRESHOLD = 2
session_log_file = None
is_paused = False
is_stopped = True
log_enabled = False

# Biểu tượng & mô tả
emotion_icons = {
    'happy': ['😄', '😁', '🤩', '😊'],
    'sad': ['😢', '😭', '😞'],
    'angry': ['😠', '😡', '🤬'],
    'surprise': ['😲', '😮', '🤯'],
    'neutral': ['😐', '😶'],
    'fear': ['😨', '😱', '😰'],
    'disgust': ['🤢', '😖', '🤮']
}

emotion_texts = {
    'happy': 'Bạn đang rất vui! Tiếp tục lan toả năng lượng tích cực nhé! 💖',
    'sad': 'Có vẻ bạn đang buồn... Hãy thư giãn và nghỉ ngơi một chút 🌧️',
    'angry': 'Bình tĩnh nào! Cơn giận không giúp bạn nhẹ lòng đâu 💢',
    'surprise': 'Ồ! Một điều gì đó khiến bạn bất ngờ phải không? 😮',
    'neutral': 'Biểu cảm trung tính, có lẽ bạn đang tập trung 🤔',
    'fear': 'Bạn có vẻ lo lắng? Mọi chuyện rồi sẽ ổn thôi 🫣',
    'disgust': 'Bạn cảm thấy khó chịu? Hít một hơi thật sâu nhé 😖'
}

def get_log_writer():
    global session_log_file
    if session_log_file and log_enabled:
        f = open(session_log_file, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        return f, writer
    return None, None

from collections import deque, Counter

EMOTION_UPDATE_INTERVAL = 5  # giây
emotion_buffer = deque()
last_update_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    return jsonify({**current_emotion, "alert": alert_mode})

@app.route('/reset_alert', methods=['POST'])
def reset_alert():
    global alert_mode
    alert_mode = False
    return jsonify({"status": "reset"})

@app.route('/start_session')
def start_session():
    global session_log_file, is_paused, is_stopped, alert_mode, sad_start_time, log_enabled
    index = request.args.get('index', '1')
    session_log_file = f"session_{index}.csv"
    if not os.path.exists(session_log_file):
        with open(session_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Emotion"])
    is_paused = False
    is_stopped = False
    alert_mode = False
    sad_start_time = None
    log_enabled = True
    return jsonify({"status": "created", "file": session_log_file})

@app.route('/pause', methods=['POST'])
def pause():
    global is_paused, log_enabled
    is_paused = True
    log_enabled = False
    return jsonify({"status": "paused"})

@app.route('/resume', methods=['POST'])
def resume():
    global is_paused, log_enabled
    is_paused = False
    log_enabled = True
    return jsonify({"status": "resumed"})

@app.route('/stop', methods=['POST'])
def stop():
    global is_stopped, is_paused, log_enabled
    is_stopped = True
    is_paused = False
    log_enabled = False
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
