import cv2

capture = cv2.VideoCapture("/Users/denner/Developer/Astrophotography/src/videos/2023-11-03-JUPITER-3.mov", cv2.CAP_FFMPEG)

frameNum = 0
radius = 45

while(True):
    success, frame = capture.read()

    if success:
        original_image = frame.copy()
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grayscale_image = cv2.GaussianBlur(grayscale_image, (radius, radius), 0)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(grayscale_image)

        image = original_image.copy()

        x, y, w, h = max_loc[0]-radius, max_loc[1]-radius, max_loc[0]+radius, max_loc[1]+radius

        cropped = image[y:h, x:w]

        sr = cv2.dnn_superres.DnnSuperResImpl.create()
        sr.readModel("/Users/denner/Developer/Astrophotography/src/EDSR_x4.pb")
        sr.setModel("edsr", 4)
        
        sr_result = sr.upsample(cropped)
        
        cv2.imwrite(f'/Users/denner/Developer/Astrophotography/src/frames/frame_{frameNum}.jpeg', sr_result)

    else:
        break

    frameNum = frameNum + 1

capture.release()
