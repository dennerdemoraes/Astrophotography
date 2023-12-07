import os
import cv2
import numpy as np

images = os.listdir("/Users/denner/Developer/Astrophotography/src/minguante")

frameNum = 0

for image in images:
    img = cv2.imread(f'/Users/denner/Developer/Astrophotography/src/minguante/{image}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]

    max_size = max(h,w)
    min_size = 400
    margin = 100

    circle_img = img.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max_size, param1=50, param2=30, minRadius=min_size, maxRadius=max_size)

    for circle in circles[0]:
        # draw the outer circle
        # cv2.circle(circle_img, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0), 2)
        # draw the center of the circle
        # cv2.circle(circle_img, (int(circle[0]), int(circle[1])), 2, (0,0,255), 3)

        x, y, w, h = int(circle[0]-circle[2]-margin), int(circle[1]-circle[2]-margin), int(circle[0]+circle[2]+margin), int(circle[1]+circle[2]+margin)
        cropped = circle_img[y:h, x:w]

        cv2.imwrite(f'/Users/denner/Developer/Astrophotography/src/frames/frame_{frameNum}.jpeg', cropped)

        frameNum = frameNum + 1