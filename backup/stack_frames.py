import os
import cv2
import numpy as np

def find_homography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0,len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt


    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography


def align_images(images):
    outimages = []

    detector = cv2.ORB.create(1000)

    outimages.append(images[0])
    image1gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    image_1_kp, image_1_desc = detector.detectAndCompute(image1gray, None)

    for i in range(1, len(images)):
        image_i_kp, image_i_desc = detector.detectAndCompute(images[i], None)

        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # rawMatches = bf.match(image_i_desc, image_1_desc)

        bf = cv2.BFMatcher()
        pairMatches = bf.knnMatch(image_i_desc, image_1_desc, k=2)
        rawMatches = []
        for m, n in pairMatches:
            if m.distance < 0.7*n.distance:
                rawMatches.append(m)

        sortMatches = sorted(rawMatches, key=lambda x: x.distance)
        matches = sortMatches[0:128]

        hom = find_homography(image_i_kp, image_1_kp, matches)
        newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

        outimages.append(newimage)

    return outimages



def do_lap(image):
    kernel_size = 5
    blur_size = 5

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


def focus_stack(images):
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    for i in range(0, len(images)):
        output = cv2.bitwise_not(images[i], output)
        
    return 255-output


images_stack = []

images = os.listdir("/Users/denner/Developer/Astrophotography/src/frames")

for img in images:
    images_stack.append(cv2.imread(f'/Users/denner/Developer/Astrophotography/src/frames/{img}'))

merged = focus_stack(images_stack)

cv2.imwrite("/Users/denner/Developer/Astrophotography/src/merged/merged.png", merged)