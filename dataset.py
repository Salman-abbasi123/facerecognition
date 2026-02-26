import cv2
import os
import time
import dlib

# Initialize face detector for validation
detector = dlib.get_frontal_face_detector()

# Capture images for dataset
def capture_images(person_name, save_path='dataset', num_images=10):
    person_name = person_name.strip()
    if not person_name:
        raise ValueError("Name cannot be empty")

    # Create directory for the person
    current_folder = os.path.dirname(os.path.abspath(__file__))
    person_path = os.path.join(current_folder, save_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    count = 0
    failed_attempts = 0
    max_failed_attempts = 20  # Maximum number of attempts to capture a valid face

    while count < num_images and failed_attempts < max_failed_attempts:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 1:  # Ensure exactly one face is detected
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save the original color frame
            img_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            count += 1
            failed_attempts = 0  # Reset failed attempts counter
        else:
            failed_attempts += 1

    cap.release()

    if count < num_images:
        if failed_attempts >= max_failed_attempts:
            raise RuntimeError("Failed to capture enough valid face images")
        else:
            raise RuntimeError("Image capture was interrupted")

    return f"Successfully captured {count} images for {person_name}"

def capture_images_from_frame(person_name, frame, save_path='dataset'):
    # This function will be called from the web frontend, frame is already captured
    person_name = person_name.strip()
    if not person_name:
        raise ValueError("Name cannot be empty")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    person_path = os.path.join(current_folder, save_path, person_name)
    os.makedirs(person_path, exist_ok=True)
    count = len(os.listdir(person_path))
    img_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
    cv2.imwrite(img_name, frame)
    return f"Image saved for {person_name}"
