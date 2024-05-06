import cv2
import pickle
import cvzone
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

parkingVideo = 'VideoSource/carPark2.mp4'

cap = cv2.VideoCapture(parkingVideo)

width, height = 107, 48
parkingSpacePicker = "CarParkPos"
with open(parkingSpacePicker, 'rb') as f:
    posList = pickle.load(f)

def empty(a):
    pass

cv2.namedWindow("Values")
cv2.resizeWindow("Values", 640, 240)
cv2.createTrackbar("Threshold", "Values", 25, 50, empty)
cv2.createTrackbar("Pixels", "Values", 16, 50, empty)
cv2.createTrackbar("ImageBlur", "Values", 5, 50, empty)
cv2.createTrackbar("Slow", "Values", 50, 150, empty)

def checkSpaces(img, imgThres):
    spaces = 0
    for pos in posList:
        x, y = pos
        w, h = width, height
        imgCrop = imgThres[y:y + h, x:x + w]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 200, 0)
            thickness = 5
            spaces += 1
        else:
            color = (0, 0, 200)
            thickness = 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img, str(cv2.countNonZero(imgCrop)), (x, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (50, 60), thickness=3, offset=20, colorR=(0, 200, 0))
    cv2.imshow("Image", img)

def parking_space_counter():
    while True:
        success, img = cap.read()
        if not success:
            break
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        val1 = cv2.getTrackbarPos("Threshold", "Values")
        val2 = cv2.getTrackbarPos("Pixels", "Values")
        val3 = cv2.getTrackbarPos("ImageBlur", "Values")
        val4 = cv2.getTrackbarPos("Slow", "Values")
        if val1 % 2 == 0:
            val1 += 1
        if val3 % 2 == 0:
            val3 += 1
        imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, val1, val2)
        imgThres = cv2.medianBlur(imgThres, val3)
        kernel = np.ones((3, 3), np.uint8)
        imgThres = cv2.dilate(imgThres, kernel, iterations=1)
        checkSpaces(img, imgThres)
        cv2.waitKey(val4)

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parking_space_counter()
    app.run(debug=True)
