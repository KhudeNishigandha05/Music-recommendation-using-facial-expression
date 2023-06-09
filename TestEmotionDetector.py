import cv2
import numpy as np
from keras.models import model_from_json

emotion =['Angry', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised', 'Disgust']

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights('model/emotion_model.h5')
print("Loaded model from disk")

def emotion_testing():
    # Start the webcam feed
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
            cv2.rectangle(frame, (x,y-50), (x+w,y+h+10),(0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

         # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            predicted_emotion = emotion[maxindex]
            cv2.putText(frame, predicted_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted_emotion

emotion_testing()