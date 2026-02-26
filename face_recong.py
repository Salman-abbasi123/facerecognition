import os
import cv2
import dlib
import numpy as np
import csv
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import deque

# Initialize dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Global variables
attendance_log = {}
attendance_list = deque(maxlen=10)
clf = None  # SVM classifier
le = None   # Label encoder

# Training progress tracking
model_training_progress = {
    'stage': 'Not started',
    'progress': 0,
    'message': 'Waiting to start',
    'model_ready': False
}

def update_training_progress(stage, progress, message):
    """Update the training progress state"""
    model_training_progress['stage'] = stage
    model_training_progress['progress'] = progress
    model_training_progress['message'] = message

def get_training_progress():
    """Get the current training progress"""
    return model_training_progress.copy()

def mark_attendance(name):
    current_time = datetime.now()
    if name in attendance_log:
        last_time = attendance_log[name]
        # Mark attendance every 5 seconds for testing
        if current_time - last_time >= timedelta(seconds=5):
            attendance_log[name] = current_time
            save_attendance(name, current_time)
            return True
    else:
        attendance_log[name] = current_time
        save_attendance(name, current_time)
        return True
    return False

def save_attendance(name, time):
    try:
        # Get absolute path for attendance.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        attendance_file = os.path.join(current_dir, 'attendance.csv')
        # Create file with headers if it doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Date Time'])
                file.flush()
        # Append the attendance
        with open(attendance_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, time.strftime('%Y-%m-%d %H:%M:%S')])
            file.flush()
        attendance_list.appendleft({'name': name, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    except PermissionError:
        pass
    except Exception as e:
        pass

def start_face_recognition(dataset_path):
    global model_training_progress, clf, le

    update_training_progress('init', 0, 'Initializing face recognition...')
    
    # Load face encodings
    face_encodings = []
    labels = []
    total_files = 0
    
    # Count total files for progress tracking
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            total_files += len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
    
    files_processed = 0
    update_training_progress('loading', 10, 'Loading dataset...')
    
    # Process each image
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        # Load and process image
                        frame = cv2.imread(img_path)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray)
                        
                        if len(faces) > 0:
                            shape = sp(gray, faces[0])
                            face_descriptor = facerec.compute_face_descriptor(frame, shape)
                            face_encodings.append(face_descriptor)
                            labels.append(person_name)
                        
                        files_processed += 1
                        progress = 10 + int((files_processed / total_files) * 40)  # Progress from 10% to 50%
                        update_training_progress('loading', progress, f'Processing images: {files_processed}/{total_files}')
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        continue

    # Convert lists to numpy arrays
    X = np.array(face_encodings)
    y = np.array(labels)
    
    update_training_progress('preprocessing', 50, 'Preprocessing data...')
    
    # Label encoding
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    update_training_progress('training', 60, 'Training model...')
    
    # Train SVM classifier
    clf = SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    update_training_progress('evaluating', 80, 'Evaluating model...')
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Completed. Accuracy: {accuracy * 100:.2f}%")
    
    update_training_progress('complete', 100, f'Model ready (Accuracy: {accuracy*100:.1f}%)')
    model_training_progress['model_ready'] = True
    print("\nStarting face detection... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # Draw box for each face and recognize
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            shape = sp(gray, face)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)
            prediction = clf.predict(face_descriptor)
            name = le.inverse_transform(prediction)[0]
            # Try to mark attendance and get feedback
            if mark_attendance(name):
                color = (0, 255, 0)  # Green for successful marking
            else:
                color = (0, 255, 255)  # Yellow for already marked
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if len(faces) == 0:
            print("No face detected in frame.")
        # Do NOT save any images during detection! Only use dataset images for recognition.
        # cv2.imshow('Face Recognition', frame)  # Removed for web mode
        
        # Check for 'q' key press - Added more robust key checking
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nStopping face detection...")
            break

    # Cleanup
    cap.release()
    # cv2.destroyAllWindows()  # Removed to disable windowed display
    print("Face detection stopped.")

def generate_frames():
    global model_training_progress
    if not model_training_progress['model_ready']:
        print("Warning: Model not ready for detection")
        return

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try using DirectShow backend
            if not cap.isOpened():
                raise ValueError("Could not open camera")
            
            # Set camera properties for better stability
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"Frame read failed. Attempt {consecutive_failures} of {max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive frame read failures. Restarting camera...")
                        break
                        
                    # Small delay before next attempt
                    import time
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                
                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                # Draw box for each face and recognize
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    shape = sp(gray, face)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    face_descriptor = np.array(face_descriptor).reshape(1, -1)
                    prediction = clf.predict(face_descriptor)
                    name = le.inverse_transform(prediction)[0]
                    
                    if mark_attendance(name):
                        attendance_list.appendleft({'name': name, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Camera error occurred: {str(e)}")
            retry_count += 1
            print(f"Retrying camera initialization... Attempt {retry_count} of {max_retries}")
            
            if cap is not None:
                cap.release()
                
            # Wait before retrying
            import time
            time.sleep(1)
            
        finally:
            if cap is not None:
                cap.release()
    
    print("Face detection stopped after maximum retries")
