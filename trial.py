import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder

# cap = cv2.VideoCapture('video5.mp4')
cap = cv2.VideoCapture(2)
# frameCounter = 0 

colorFinder = ColorFinder(True) # To Decide HSV values of objects "True"
# cornerPoints = [[92,27],[11,545],[457,87],[376,600]]  #video2.py
# cornerPoints = [[19,667],[39,136],[385,678],[407,157]]  #video3.py


cornerPoints = [[58,129],[579,121],[62,448],[590,430]]  # board

def getBoard(img):
    scale = 1
    bw, bh = 1110, 660
    width, height = int(bw*scale),int(bh*scale)  # A4 paper size will be changed for the board size (height = 600 mm) width 1110mm
 
    pts1 =  np.float32(cornerPoints)
    pts2 =  np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width, height))

    return imgOutput

while True:
    # frameCounter += 1
    # if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     frameCounter = 0 
    #     cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()
    # img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    imgBoard = getBoard(img)

    

    imgColor, mask = colorFinder.update(imgBoard)

    cv2.imshow("ImageColor",imgColor)
    # cv2.imshow("Image",img)
    # cv2.imshow("imgBoard",imgBoard)

    cv2.imwrite("img.png",img)
    mm=cv2.imread('img.png')
    # cv2.imshow("hee",mm)
    
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()