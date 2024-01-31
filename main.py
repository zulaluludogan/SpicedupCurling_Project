import cv2
import time
import numpy as np
from cvzone.ColorModule import ColorFinder


cap = cv2.VideoCapture('video5.mp4')
frameCounter = 0 

# cornerPoints = [[418,142],[418,683],[21,76],[24,752]]  # Calibrate acc board
# cornerPoints = [[92,27],[11,545],[457,87],[376,600]] # video2.py coordinates
cornerPoints = [[19,667],[39,136],[385,678],[407,157]] # video3.py

colorFinder = ColorFinder(False) # To Decide HSV values of objects "True"

### VIDEO1&2 HSV ###
# hsvTargetVals = {'hmin': 156, 'smin': 67, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}  # Red Target
# hsvPuck1Vals = {'hmin': 26, 'smin': 63, 'vmin': 0, 'hmax': 58, 'smax': 255, 'vmax': 255}     # Yellow Puck
# hsvPuck2Vals = {'hmin': 56, 'smin': 97, 'vmin': 0, 'hmax': 111, 'smax': 255, 'vmax': 255}    # Blue Puck
# # hsvObstacleVals = {'hmin': 0, 'smin': 25, 'vmin': 66, 'hmax': 179, 'smax': 50, 'vmax': 160}  # Black Obstacles #video1.py
# hsvObstacleVals = {'hmin': 0, 'smin': 0, 'vmin': 0, 'hmax': 179, 'smax': 18, 'vmax': 124}    # Black Obstacles #video2.py

### VIDEO 3&5 HSV ###
hsvTargetVals = {'hmin': 0, 'smin': 182, 'vmin': 0, 'hmax': 25, 'smax': 255, 'vmax': 255}
hsvPuck1Vals = {'hmin': 28, 'smin': 0, 'vmin': 0, 'hmax': 50, 'smax': 255, 'vmax': 255}  # YELLOW
hsvPuck2Vals = {'hmin': 56, 'smin': 0, 'vmin': 0, 'hmax': 112, 'smax': 255, 'vmax': 255} # BLUE
hsvObstacleVals = {'hmin': 0, 'smin': 25, 'vmin': 0, 'hmax': 179, 'smax': 117, 'vmax': 94}

###### DIP FUNCTIONS

def getBoard(img):
    global scale 
    scale = 2.5
    width, height = int(297*scale),int(210*scale)  # A4 paper size will be changed for the board size (height = 600 mm)

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

    return mask

def detectContour(mask):
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE -> 4 points | cv2.CHAIN_APPROX_NONE -> all boundary points
    numberofObjects = len(contours)

    return contours, numberofObjects 

def findCenterContour(cnt):
    global imgBoard
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

def getPointArrays(contlist,pointlist):
    for cont in contlist:
        cX, cY, r = findCenterContour(cont)
        pointlist.append([cX, cY, r])

def getDistance(x1, y1, x2, y2):
    distance = (((x1-x2)**2 + (y1-y2)**2)**0.5)/scale

    return distance

def getFirstFrame(imgBoard):
    maskFirstFrame = createHsvMask(imgBoard, hsvPuck2Vals)
    
    return maskFirstFrame

def pausePuckisOnBoard():
    global myTurn,  maskPausePuck
    if maskPausePuck[PausePuck[1]][PausePuck[0]] == 0:
        PauseFlag = False
    else:
        PauseFlag = True
    # print(PausePuck[0],PausePuck[1])
    # print(maskPausePuck[PausePuck[1]][PausePuck[0]])

    return PauseFlag

def defineMyTurn(imgBoard):          # EDIT THIS PART FOR THE PUCK NEXT TO BOARD
    global myTurn, PausePuck, maskPausePuck
    maskPuck1 = createHsvMask(imgBoard, hsvPuck1Vals)
    maskPuck2 = createHsvMask(imgBoard, hsvPuck2Vals)
    contPuck1, numberofPunks1 =  detectContour(maskPuck1)
    contPuck2, numberofPunks2 =  detectContour(maskPuck2)
    
    if numberofPunks1 > numberofPunks2:
        myTurn = 1
        px, py, _ = findCenterContour(contPuck1[0])
        maskPausePuck  = maskPuck1
        print("my punk:",str(numberofPunks1),"opponent punk:",str(numberofPunks2))
        print(" We are starting!")
    else:
        myTurn = 0
        px, py, _ = findCenterContour(contPuck2[0])
        maskPausePuck  = maskPuck2
        print(" Opponent is starting!")
    PausePuck = [px, py]  # If pausepuck moves, PAUSE the game !!!!
     
motionDetected = 0
START = 1
targetCenterRadius = []
     
while True:
    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0 
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5) # For video3.py
    imgBoard = getBoard(img)

    maskTarget = createHsvMask(imgBoard, hsvTargetVals)
    maskObst = createHsvMask(imgBoard, hsvObstacleVals)
    maskPuck1 = createHsvMask(imgBoard, hsvPuck1Vals)
    maskPuck2 = createHsvMask(imgBoard, hsvPuck2Vals)

    contTarget, numberofTarget =  detectContour(maskTarget)
    contObstac, numberofObstac =  detectContour(maskObst)
    contPuck1, numberofPunks1 =  detectContour(maskPuck1)
    contPuck2, numberofPunks2 =  detectContour(maskPuck2)

    if START :  # Define beginning conditions
        maskFirstFrame = getFirstFrame(imgBoard)
        defineMyTurn(imgBoard)
        getPointArrays(contTarget,targetCenterRadius)
        START = 0

    if not myTurn:  
        diff = cv2.absdiff(maskPuck2, maskFirstFrame)
        threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        maskFirstFrame = getFirstFrame(imgBoard)
        if threshold.sum() > 150000:
            print("---motion is detected")
            # "wait until motion stops"
            motionDetected = 1
 
        elif motionDetected:
            myTurn = 1
            motionDetected = 0
            print(">>>motion stopped")

        cv2.imshow("threshold",threshold)
        # print(threshold.sum())

    elif myTurn:
        obstacleEdgePoints = []
        Puck1CenterRadius  = []
        Puck2CenterRadius  = []
        
        getPointArrays(contPuck1,Puck1CenterRadius)
        getPointArrays(contPuck2,Puck2CenterRadius)
        getPointArrays(contObstac,obstacleEdgePoints)
        
        maskFirstFrame = getFirstFrame(imgBoard)
        myTurn = 0
        print( " ACTIVE PLAYER 2")

        print("MY TURN!!!! RUN ALGORITHM")
        print( "ROBOT TAKES AN ACTION")
        
        print("puck1",str(Puck1CenterRadius))
        print("puck2",str(Puck2CenterRadius))
        print("target", str(targetCenterRadius))
        print("obstac", str(obstacleEdgePoints))

        # cv2.imshow("maskPuck1",maskPuck1)
        # cv2.imshow("maskPuck2",maskPuck2)
        # cv2.imshow("maskTarget",maskTarget)
        # cv2.imshow("maskObst",maskObst)

        # time.sleep(1)
        # WAIT UNTIL ROBOT SAYS "MISSION COMPLETED"
        #...
        #...
        #...
    print(pausePuckisOnBoard())
    # cv2.imshow("maskPausePuck",maskPausePuck)

    # print("Number of Contours Puck1 = " + str(numberofPunks1))
    # print("Number of Contours Puck2 = " + str(numberofPunks2))
    # print("Number of Contours Target = " + str(numberofTarget)) 
    # print("Number of Contours Obstacles = " + str(numberofObstac))

    # print("Distance",getDistance(175, 393, 199, 358))
    # print("Distance",getDistance(175, 393, 284, 468))
   
    cv2.imshow("ImageBoard",imgBoard)
    # cv2.imshow("imgContour",imgContour)

    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

