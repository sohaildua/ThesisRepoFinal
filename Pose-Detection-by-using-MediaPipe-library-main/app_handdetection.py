

import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp




app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    """Video streaming generator function."""
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
        cap = cv2.VideoCapture(0)

        while True:
            success, image = cap.read()
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # converting image to RGB from BGR cuz mediapipe only work on RGB
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # checking video frame rate
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            # Writing FrameRate on video
            cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            #cv2.imshow("Pose detection", img)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)







