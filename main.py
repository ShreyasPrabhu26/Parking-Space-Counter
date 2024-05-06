import cv2
import pickle
import cvzone
import numpy as np

parkingVidio = f'VideoSource/carPark2.mp4'

cap = cv2.VideoCapture(parkingVidio)

# Width and Height of individual Car
width, height = 107, 48

# parkingSpacePicker is a binary file for storing the parking spots
parkingSpacePicker = "CarParkPos"
with open(parkingSpacePicker, 'rb') as f:
    posList = pickle.load(f)


def empty(a):
    pass


# Play with the values for better Results
cv2.namedWindow("Values")
cv2.resizeWindow("Values", 640, 240)
cv2.createTrackbar("Threshold", "Values", 25, 50, empty)
cv2.createTrackbar("Pixels", "Values", 16, 50, empty)
cv2.createTrackbar("ImageBlur", "Values", 5, 50, empty)
cv2.createTrackbar("Slow", "Values", 50, 150, empty)


# Checks if the object (car){with the help of - number of pixels} within the region then decides if its empty or not
def checkSpaces():
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

        cv2.putText(img, str(cv2.countNonZero(imgCrop)), (x, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1,
                    color, 2)

    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (50, 60), thickness=3, offset=20,
                       colorR=(0, 200, 0))


while True:
    # Get image frame from the source
    success, img = cap.read()

    # Looping the vidio (if currentFrame == lastFrame)-->currentFrame=0
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Converting the image frame to greyscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Custom Values for better accuracy
    val1 = cv2.getTrackbarPos("Threshold", "Values")
    val2 = cv2.getTrackbarPos("Pixels", "Values")
    val3 = cv2.getTrackbarPos("ImageBlur", "Values")
    val4 = cv2.getTrackbarPos("Slow", "Values")

    if val1 % 2 == 0:
        val1 += 1

    if val3 % 2 == 0:
        val3 += 1

    # Thresold will captures only pure black and white pixels
    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, val1, val2)
    imgThres = cv2.medianBlur(imgThres, val3)
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    checkSpaces()

    cv2.imshow("Image", img)
    cv2.imshow("ImageGray", imgThres)
    cv2.imshow("ImageBlur", imgBlur)
    key = cv2.waitKey(1)
    if key == ord('r'):
        pass

    # Vidio Speed
    cv2.waitKey(val4)
