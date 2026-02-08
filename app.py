from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

app = Flask(__name__)

# --------------------------
# Load trained model
# --------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.p')

try:
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
    31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

# --------------------------
# Initialize MediaPipe
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# --------------------------
# Global variables for text
# --------------------------
predicted_letter = ""
word = ""
sentence = ""
camera = None


# --------------------------
# Initialize camera
# --------------------------
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera


# --------------------------
# Generate video frames
# --------------------------
def generate_frames():
    global predicted_letter, word, sentence

    cap = get_camera()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Detect and draw landmarks
        if results.multi_hand_landmarks and model is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Prepare data for prediction
            data_aux, x_, y_ = [], [], []
            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Predict letter
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_letter = labels_dict[int(prediction[0])]

                # Draw bounding box and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_letter, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# --------------------------
# Routes
# --------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_text')
def get_text():
    global predicted_letter, word, sentence
    return jsonify({
        'letter': predicted_letter,
        'word': word,
        'sentence': sentence
    })


@app.route('/add_letter', methods=['POST'])
def add_letter():
    global predicted_letter, word
    if predicted_letter != "":
        word += predicted_letter
    return jsonify({'success': True, 'word': word})


@app.route('/add_space', methods=['POST'])
def add_space():
    global word, sentence
    if word.strip() != "":
        sentence = (sentence + word + " ").strip() + " "
        word = ""
    return jsonify({'success': True, 'sentence': sentence})


@app.route('/backspace', methods=['POST'])
def backspace():
    global word
    word = word[:-1]
    return jsonify({'success': True, 'word': word})


@app.route('/clear_all', methods=['POST'])
def clear_all():
    global word, sentence
    word, sentence = "", ""
    return jsonify({'success': True})


if __name__ == '__main__':
    # Run on all interfaces so it can be accessed from other devices
    app.run(host='0.0.0.0', port=5000, debug=True)
