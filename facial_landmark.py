import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector =  dlib.get_frontal_face_detector()
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image_copy = np.copy(cap)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        x_pts = []
        y_pts = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            x_pts.append(x)
            y_pts.append(y)

            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    x1 = min(x_pts)
    x2 = max(x_pts)
    y1 = min(y_pts)
    y2 = max(y_pts)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything is done, release the capture and save m
a = cv2.waitKey(0)
if a == 27:
    cv2.destroyAllWindows()
elif a == ord("s"):
    cv2.imwrite("kaydettiginresim.png", frame)
    # s tusuna bas ve kaydet

