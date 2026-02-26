# Face Recognition and Attendance System

## Python Version

This project requires **Python 3.8.6**. It is recommended to use the provided virtual environment (`venv_py386`) or create a new one with this version for best compatibility.

This project is a Face Recognition and Attendance System built with Python, OpenCV, dlib, and Flask. It allows you to:

- Run as a web application using `app.py` (Flask web server)
- Capture face images for new users and build a dataset
- Recognize faces in real-time using your webcam
- Mark attendance automatically when a face is recognized
- View and manage attendance records

## Features

- **Face Detection & Recognition:** Uses dlib and OpenCV for accurate face recognition.
- **Dataset Management:** Easily add new users and collect their face images.
- **Attendance Logging:** Automatically logs attendance with timestamps in a CSV file.
- **Web Interface:** Flask-based web app for easy interaction (customizable).

## Project Structure

```text
├── app.py                  # Flask web app (if used)
├── main.py                 # Main CLI for dataset and recognition
├── dataset.py              # Dataset creation logic
├── face_recong.py          # Face recognition logic
├── attendance.csv          # Attendance log file
├── models/                 # Pre-trained dlib models
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── shape_predictor_68_face_landmarks.dat
├── dataset/                # Collected face images per user
├── static/                 # Static files for web app
├── templates/              # HTML templates for web app
├── requirements.txt        # Python dependencies
└── venv_py386/             # Python virtual environment
```

## Setup Instructions

1. **Clone the repository**
2. **Create and activate a virtual environment** (such as `venv_py386` for Python 3.8.6)
3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download dlib models:**
   - Place `dlib_face_recognition_resnet_model_v1.dat` and `shape_predictor_68_face_landmarks.dat` in the `models/` folder. (already included)

## Usage

### 1. Run the Web Application

Start the Flask web server:

```sh
python app.py
```

- Open your browser and go to `http://localhost:5000` to use the web interface.

### 2. Add a new user to the dataset

### 3. Start face recognition and attendance

### 4. View attendance

- Open `attendance.csv` to see the attendance records.

## Requirements

See `requirements.txt` for all dependencies. Key libraries:

- dlib
- opencv-python
- numpy
- scikit-learn
- Flask

## Notes

- The project is designed for Windows (uses `pywin32` and related packages).
- Make sure your webcam is connected and accessible.
- For best results, collect at least 10 images per user in different lighting and angles.

## Credits

- Built using dlib, OpenCV, and Flask.
- Pre-trained models from the dlib library.

---

Feel free to customize the web interface or extend the system for your needs!
