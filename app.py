from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import pickle
import warnings
import threading

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize MediaPipe Hand Recognition
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

output = ""

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

def speak_text(text):
    """Function to handle speaking in a separate thread."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    global output
    output = ""  # Clear the output after speaking

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    timer_started = False
    timer_duration = 3
    current_timer = timer_duration
    global output
    current_character = ""
    box_color = (0, 0, 255)
    
    while True:
        data_aux = []
        x_ = []
        y_ = []
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if not timer_started:
                timer_started = True
                start_time = time.time()

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                data_aux.extend([0.0] * 42)

            if len(data_aux) == 84 and model is not None:
                prediction = model.predict([np.asarray(data_aux)])
                current_character = prediction[0]

                elapsed_time = time.time() - start_time
                current_timer = timer_duration - elapsed_time

                cv2.putText(frame, f"Timer: {int(current_timer)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                box_color = (0, 0, 255)

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                cv2.putText(frame, current_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                if current_timer <= 0:
                    box_color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                    if current_character == 'space':
                        output += '  '
                    elif current_character == 'del':
                        output = output[:-1]
                    else:
                        output += current_character

                    timer_started = False
                    current_timer = timer_duration
                    current_character = ""

        else:
            timer_started = False
            current_timer = timer_duration
            box_color = (0, 0, 255)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/speak', methods=['GET'])
def speak_output():
    global output
    if output:
        # Start the speak function in a new thread
        threading.Thread(target=speak_text, args=(output,)).start()
    return "", 204  # Return an empty response with status code 204 (No Content)

@app.route('/get_output', methods=['GET'])
def get_output():
    global output
    return jsonify({"output": output})

@app.route('/clear_output', methods=['POST'])
def clear_output():
    global output
    output = ""
    return "", 204  # Return an empty response with status code 204 (No Content)

if __name__ == "__main__":
    app.run(debug=True)
