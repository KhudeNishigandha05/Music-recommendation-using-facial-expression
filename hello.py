from __future__ import division, print_function
import cv2
from keras.models import model_from_json
#import sys
import os
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st

app = Flask(__name__)


emotion =['Angry', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised', 'Disgust']

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights('model/emotion_model.h5')
print("Loaded model from disk")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/camera", methods=['POST', 'GET'])
def camera():
    output=[]
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier(
            'C:\\Users\\NISHIGANDHA\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Take each face available on the camera and preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            predicted_emotion = emotion[maxindex]
            output.append(predicted_emotion)
            cv2.putText(frame, predicted_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("index.html", final_output=final_output1)



@app.route('/songs/Sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("Sad.html")

@app.route('/songs/Happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("Happy.html")

@app.route('/songs/Angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("Angry.html")

@app.route('/songs/Neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("Neutral.html")

@app.route('/songs/Fearful', methods = ['GET', 'POST'])
def songsFearful():
    return render_template("Fearful.html")

@app.route('/songs/Surprised', methods = ['GET', 'POST'])
def songsSurprised():
    return render_template("Surprised.html")


if __name__=='__main__':
    app.run(debug=True,host='localhost', port=8080)