import os
import cv2
import numpy as np

images = os.listdir("/Users/denner/Developer/Astrophotography/src/images")

radius = 45
frameNum = 0


for img in images:
    frame = cv2.imread(f'/Users/denner/Developer/Astrophotography/src/images/{img}')

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.GaussianBlur(grayscale_image, (radius, radius), 0)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(grayscale_image)

    x, y, w, h = max_loc[0]-300, max_loc[1]-300, max_loc[0]+300, max_loc[1]+300

    cropped = frame[y:h, x:w]

    sr = cv2.dnn_superres.DnnSuperResImpl.create()
    sr.readModel("/Users/denner/Developer/Astrophotography/src/EDSR_x4.pb")
    sr.setModel("edsr", 4)

    sr_result = sr.upsample(cropped)

    cv2.imwrite(f'/Users/denner/Developer/Astrophotography/src/frames/frame_{frameNum}.jpeg', sr_result)

    frameNum = frameNum + 1
