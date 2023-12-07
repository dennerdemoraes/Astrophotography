import os
import cv2
import numpy as np

img = cv2.imread('/Users/denner/Developer/Astrophotography/src/moon/moon.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = img.shape[:2]

max_size = max(h,w)
min_size = 200
margin = 100

circle_img = img.copy()
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max_size, param1=50, param2=30, minRadius=min_size, maxRadius=max_size)

circles = np.uint16(np.around(circles))
for circle in circles[0]:
    x, y, w, h = circle[0]-circle[2]-margin, circle[1]-circle[2]-margin, circle[0]+circle[2]+margin, circle[1]+circle[2]+margin
    cropped = circle_img[y:h, x:w]

# sr = cv2.dnn_superres.DnnSuperResImpl.create()
# sr.readModel("/Users/denner/Developer/Astrophotography/src/EDSR_x4.pb")
# sr.setModel("edsr", 4)

# sr_result = sr.upsample(cropped)

cv2.imshow('cropped', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()