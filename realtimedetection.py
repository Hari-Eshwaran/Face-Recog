import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion detection model
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Emotion labels corresponding to model output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess the face for prediction
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))   # <-- Correct size (64x64)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)    # (1, 64, 64)
        face = np.expand_dims(face, axis=-1)   # (1, 64, 64, 1)

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Put emotion text above rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Face Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
