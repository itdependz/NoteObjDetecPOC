from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import json

rf = Roboflow(api_key="v6L7b4dceS02WW7pcCUx")
project = rf.workspace().project("2024-frc")
model = project.version(8).model

video_path = "2024-frc-robot-pov.mp4"
cap = cv2.VideoCapture(video_path)

#Output video setup
output_path = "NoteDetections.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame, confidence=40, overlap=30).json()

    for prediction in results['predictions']:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        if class_name=="note":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Detection', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()