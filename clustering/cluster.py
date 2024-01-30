import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
   # img = cv2.imread("circle.jpeg")
   # Z = img.reshape((-1,3))

   # # convert to np.float32
   # Z = np.float32(Z)

   # # define criteria, number of clusters(K) and apply kmeans()
   # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   # K = 6
   # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

   # # Now convert back into uint8, and make original image
   # center = np.uint8(center)
   # res = center[label.flatten()]
   # res2 = res.reshape((img.shape))

   # cv2.imshow('res2',res2)
   # cv2.waitKey(1)

   img = cv2.imread("circle.jpeg", cv2.IMREAD_GRAYSCALE)
   cv2.imshow('grey',img)
   assert img is not None, "file could not be read, check with os.path.exists()"
   kernel = np.ones((11,11),np.uint8)
   erosion = cv2.erode(img,kernel,iterations = 1)
   # dilation = cv2.dilate(img,kernel,iterations = 1)
   cv2.imshow('res2',img)

   cv2.waitKey(0) 
   cv2.destroyAllWindows()