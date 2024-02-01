import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder

# cap = cv2.VideoCapture('video5.mp4')
cap = cv2.VideoCapture(2)
# frameCounter = 0 

colorFinder = ColorFinder(False) # To Decide HSV values of objects "True"

## BOARD HSV ###

hsvTargetVals = {'hmin': 0, 'smin': 28, 'vmin': 0, 'hmax': 19, 'smax': 255, 'vmax': 255}
hsvPuck1Vals = {'hmin': 63, 'smin': 4, 'vmin': 163, 'hmax': 120, 'smax': 75, 'vmax': 255} # WHITE
hsvPuck2Vals = {'hmin': 0, 'smin': 73, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 98} #PURPLE
hsvObstacleVals = {'hmin': 0, 'smin': 73, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 98} # CAMERA NORMAL
# hsvObstacleVals = {'hmin': 0, 'smin': 73, 'vmin': 0, 'hmax': 156, 'smax': 255, 'vmax': 124} # CAMERA aNORMAL

# cornerPoints = [[58,129],[579,121],[62,448],[590,430]]  # board

def getBoard(img):
    global scale
    scale = 1
    bw, bh = 1110, 660
    width, height = int(bw*scale),int(bh*scale)  # A4 paper size will be changed for the board size (height = 600 mm) width 1110mm
    
    maskObst = createHsvMask(img, hsvObstacleVals)
    contObstac, numberofObstac =  detectContour(maskObst)
    obstacleEdgePoints = []

    getPointArrays(contObstac,obstacleEdgePoints,"edge")
    obstacleEdgePoints = np.array(obstacleEdgePoints)

    left_upper = [np.min(obstacleEdgePoints[:, :, 0]), np.max(obstacleEdgePoints[:, :, 1])]
    right_upper = [np.max(obstacleEdgePoints[:, 2, 0]), np.max(obstacleEdgePoints[:, 2, 1])]
    left_lower = [np.min(obstacleEdgePoints[:, 0, 0]), np.min(obstacleEdgePoints[:, 0, 1])]
    right_lower = [np.max(obstacleEdgePoints[:, 1, 0]), np.min(obstacleEdgePoints[:, 1, 1])]
    cornerPoints=[left_upper, right_upper, left_lower, right_lower ]

    pts1 =  np.float32(cornerPoints)
    pts2 =  np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width, height))

    return imgOutput

def getDistance(x1, y1, x2, y2):
    global scale
    distance = (((x1-x2)**2 + (y1-y2)**2)**0.5)/scale

    return distance

def detectCircle(img, minRad,maxRad, hsvVals):
    global imgBoarder
    circleArray = []
    if hsvVals == 0 :
        maskCircle = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        maskCircle = createHsvMask(img, hsvVals)
    img_gray = cv2.medianBlur(maskCircle,5)
    circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1.5,20, param1=50,param2=30,minRadius = minRad,maxRadius = maxRad)
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(imgBoarder,(i[0],i[1]),i[2],(0,255,0),2)  # draw the outer circle
            cv2.circle(imgBoarder,(i[0],i[1]),2,(0,255,0),3) # draw the center of the circle
            circleArray.append([i[0],i[1],i[2]])
        print("circle detected:",str(len(circles[0,:])))
        cv2.imshow('imgBoarder',imgBoarder)
    except:
        pass
    
    return circleArray

def createHsvMask(img,hsvVals):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor, mask = colorFinder.update(imgBlur,hsvVals)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Difference btw dilation and erosion
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=2)          # Increase white region
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask

def detectContour(mask):
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE -> 4 points | cv2.CHAIN_APPROX_NONE -> all boundary points
    numberofObjects = len(contours)

    return contours, numberofObjects 

def findEdgePointsObst(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(imgBoarder, [box], 0, (0,0,255), 2)

    return [[int(box[0][0]),int(box[0][1])], [int(box[1][0]),int(box[1][1])],[int(box[2][0]),int(box[2][1])],[int(box[3][0]),int(box[3][1])]]

def getPointArrays(contlist,pointlist, shape):
    if shape == "circle":
        for cont in contlist:
            cX, cY, r = detectCircle(imgBoard, 10, 40, hsvPuck1Vals)
            pointlist.append([cX, cY, r])

    elif shape == "edge":
        for cont in contlist:
            pts= findEdgePointsObst(cont)
            # print("Distance")
            # print(getDistance(pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
            if getDistance(pts[0][0], pts[0][1], pts[1][0], pts[1][1]) > 50:
                # print(getDistance(pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
                
                pointlist.append(pts)
            # cv2.drawContours(imgBoarder, [pointlist], 0, (0,0,255), 2)

START = 1

while True:
    success, img = cap.read()
    imgBoarder = img.copy()
    imgBoard = getBoard(img)
    # imgBoardInner = getBoard(img,"inner")
    # cv2.imshow("imgBoardInner", imgBoardInner)
    imgBoarder = imgBoard.copy()
    
    maskTarget = createHsvMask(imgBoard, hsvTargetVals)
    maskObst = createHsvMask(imgBoard, hsvObstacleVals)
    maskPuck1 = createHsvMask(imgBoard, hsvPuck1Vals)
    maskPuck2 = createHsvMask(imgBoard, hsvPuck2Vals)

    contTarget, numberofTarget =  detectContour(maskTarget)
    contObstac, numberofObstac =  detectContour(maskObst)
    contPuck1, numberofPunks1 =  detectContour(maskPuck1)
    contPuck2, numberofPunks2 =  detectContour(maskPuck2)
    

    obstacleEdgePoints = []
  
    getPointArrays(contObstac,obstacleEdgePoints,"edge")
    
    Puck1CenterRadius = detectCircle(imgBoard, 10, 40, hsvPuck1Vals)
    Puck2CenterRadius = detectCircle(imgBoard, 5, 30, hsvObstacleVals)
    targetCenterRadius = detectCircle(imgBoard, 50, 70, 0)

    print("obstac", str(obstacleEdgePoints))
    # cv2.imshow("ImgBoard", imgBoard)
    cv2.imshow("imgBoarder", imgBoarder)


    # cv2.imshow("maskPuck1",maskPuck1)
    # cv2.imshow("maskPuck2",maskPuck2)
    # cv2.imshow("maskTarget",maskTarget)
    # cv2.imshow("maskObst", maskObst)
    print("Number of Contours Puck1 = " + str(len(Puck1CenterRadius)))
    print("Number of Contours Puck2 = " + str(len(Puck2CenterRadius)))
    print("Number of Contours Target = " + str(len(targetCenterRadius))) 
    print("Number of Contours Obstacles = " + str(len(obstacleEdgePoints)))

    # print("Number of Contours Puck1 = " + str(numberofPunks1))
    # print("Number of Contours Puck2 = " + str(numberofPunks2))
    # print("Number of Contours Target = " + str(numberofTarget)) 
    # print("Number of Contours Obstacles = " + str(numberofObstac))

    # cv2.imshow("Image",img)
    # cv2.imshow("imgBoard",imgBoard)
    # cv2.imwrite("img.png",imgColor)
    # mm=cv2.imread('img.png')
    # cv2.imshow("hee",mm)
    
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()