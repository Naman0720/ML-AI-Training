import cv2
import matplotlib.pyplot as plt
img1= cv2.imread('download (1).jpeg')
img2=cv2.imread('download.jpeg')
img2=img2[150:150+85,150:150+60]
def showimg(im):
    plt.imshow(im[:,:,::-1])
    plt.show()

def imshow(im):
    cv2.imshow('Preview',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

added=cv2.add(img1,img2)
showimg(added)

subtracted=cv2.subtract(img1,img2)
showimg(subtracted)