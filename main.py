import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle

cap = cv2.VideoCapture('video1.mp4')
frameCounter = 0 

cornerPoints = [[418,142],[418,683],[21,76],[24,752]]  # Calibrate acc board

colorFinder = ColorFinder(False) # To Decide HSV values of objects "True"

hsvTargetVals = {'hmin': 156, 'smin': 67, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}  # Red Target
hsvPuck1Vals = {'hmin': 26, 'smin': 63, 'vmin': 0, 'hmax': 58, 'smax': 255, 'vmax': 255}     # Yellow Puck
hsvPuck2Vals = {'hmin': 56, 'smin': 97, 'vmin': 0, 'hmax': 111, 'smax': 255, 'vmax': 255}    # Blue Puck
hsvObstacleVals = {'hmin': 0, 'smin': 25, 'vmin': 66, 'hmax': 179, 'smax': 50, 'vmax': 160}  # Black Obstacles

def getBoard(img):
    width, height = int(297*2.5),int(210*2.5)  # A4 paper size will be changed for the board size (height = 600 mm)

    pts1 =  np.float32(cornerPoints)
    pts2 =  np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width, height))

    return imgOutput

def createHsvMask(img,hsvVals):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor, mask = colorFinder.update(imgBlur,hsvVals)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Difference btw dilation and erosion
    mask = cv2.dilate(mask, kernel, iterations=1)          # Increase white region
    # cv2.imshow("ImageColor",imgColor)                     

    return mask

def detectContour(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE -> 4 points | cv2.CHAIN_APPROX_NONE -> all boundary points
    numberofObjects = len(contours)

    return contours, numberofObjects 

def findCenterContour(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(imgBoard,center,radius,(255,255,255),2)
    cv2.circle(imgBoard, center, 3, (255, 255, 255), -1)

    return center[0], center[1], radius

def findEdgePointsObst(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(imgBoard,[box],0,(0,0,255),2)

    return box
myTurn = 1

while True:
    
    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0 
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()
    imgBoard = getBoard(img)

    maskPuck1 = createHsvMask(imgBoard, hsvPuck1Vals)
    maskPuck2 = createHsvMask(imgBoard, hsvPuck2Vals)
    maskTarget = createHsvMask(imgBoard, hsvTargetVals)
    maskObst = createHsvMask(imgBoard, hsvObstacleVals)

    contTarget, numberofTarget =  detectContour(maskTarget)
    contObstac, numberofObstac =  detectContour(maskObst)
    contPuck1, numberofPunks1 =  detectContour(maskPuck1)
    contPuck2, numberofPunks2 =  detectContour(maskPuck2)
    
    if myTurn:
        obstacleEdgePoints = []
        Puck1CenterRadius = []
        Puck2CenterRadius = []
        for cont in contPuck1:
            cX, cY, r = findCenterContour(cont)
            Puck1CenterRadius.append([cX, cY, r])
        for cont in contPuck2:
            cX, cY, r = findCenterContour(cont)
            Puck2CenterRadius.append([cX, cY, r])
        for cont in contTarget:  # Constraint for 1 target 1 contour
            cX, cY, r = findCenterContour(cont)
            targetCenterRadius = [cX, cY, r]
        for cont in contObstac:
            boxPoints = findEdgePointsObst(cont)  # 4 edge points of one rectangele obstacle
            obstacleEdgePoints.append(boxPoints)
    
    # print("puck1",str(Puck1CenterRadius))
    # print("puck2",str(Puck2CenterRadius))
    # print("target", str(targetCenterRadius))
    # print("obstac", str(obstacleEdgePoints))

    # cv2.drawContours(imgBoard, contPuck1, -1, (0, 255, 0), 3)

    # print("Number of Contours Puck1 = " + str(numberofPunks1))
    # print("Number of Contours Puck2 = " + str(numberofPunks2))
    # print("Number of Contours Target = " + str(numberofTarget)) 
    # print("Number of Contours Obstacles = " + str(numberofObstac))


    cv2.imshow("ImageBoard",imgBoard)
    # cv2.imshow("Image",img)
    # cv2.imshow("ImageMask",maskPuck1)
    # cv2.imshow("ImageContours",imgContours)

    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

