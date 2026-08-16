import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = "Data/I love you"

os.makedirs(folder, exist_ok=True)

print("Press 'S' to save image")
print("Press 'Q' to quit")

while True:

    success, img = cap.read()

    if not success:
        print("Camera not detected!")
        break

    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)

        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            aspectRatio = h / w

            if aspectRatio > 1:

                k = imgSize / h
                wCal = math.ceil(k * w)

                imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap:wGap + wCal] = imgResize

            else:

                k = imgSize / w
                hCal = math.ceil(k * h)

                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        if hands:

            counter += 1

            filename = os.path.join(
                folder,
                f"Image_{counter}_{int(time.time()*1000)}.jpg"
            )

            cv2.imwrite(filename, imgWhite)

            print(f"Saved: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()