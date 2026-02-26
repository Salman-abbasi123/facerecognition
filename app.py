from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
from dataset import capture_images, capture_images_from_frame
from face_recong import start_face_recognition, generate_frames, get_training_progress
import threading
import base64
import numpy as np

app = Flask(__name__)
detection_active = False
capture_active = False
model_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training_progress')
def training_progress():
    progress = get_training_progress()
    return jsonify(progress)

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_active
    person_name = request.json.get('name', '')
    if not person_name:
        return jsonify({'error': 'Name is required'}), 400
    
    capture_active = True
    try:
        capture_images(person_name)
        return jsonify({'success': True, 'message': f'Dataset captured for {person_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        capture_active = False

@app.route('/start_detection', methods=['POST'])
def start_detect():
    global detection_active, model_thread
    if not detection_active:
        detection_active = True
        # Start model training in a separate thread
        model_thread = threading.Thread(target=start_face_recognition, args=('dataset',))
        model_thread.start()
        return jsonify({'success': True, 'message': 'Face detection initialized'})
    return jsonify({'success': False, 'message': 'Detection already active'})

@app.route('/stop_detection', methods=['POST'])
def stop_detect():
    global detection_active
    detection_active = False
    return jsonify({'success': True, 'message': 'Face detection stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recent_attendance')
def recent_attendance():
    from face_recong import attendance_list
    return jsonify(list(attendance_list))

@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.json
    person_name = data.get('name')
    img_data = data.get('image')
    if not person_name or not img_data:
        return jsonify({'error': 'Name and image are required'}), 400
    img_bytes = base64.b64decode(img_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    msg = capture_images_from_frame(person_name, frame)
    return jsonify({'success': True, 'message': msg})

@app.route('/check_name_exists', methods=['POST'])
def check_name_exists():
    name = request.json.get('name', '')
    if not name:
        return jsonify({'error': 'Name is required'}), 400
        
    dataset_path = os.path.join('dataset', name)
    exists = os.path.exists(dataset_path)
    return jsonify({'exists': exists})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
