import cv2
import matplotlib.pyplot as plt

def main():
    super_res = cv2.dnn_superres.DnnSuperResImpl.create()

    image = cv2.imread('image/_MG_3568.tif', cv2.IMREAD_UNCHANGED)

    super_res.readModel('EDSR_x4.pb')
    super_res.setModel('edsr', 4)

    image = super_res.upsample(image)

    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
