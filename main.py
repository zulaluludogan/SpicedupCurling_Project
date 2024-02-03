from raycast import RayCast
import threading 
import numpy as np 
import cv2
import ImagePro
import utility

# import communicate


cap = cv2.VideoCapture(2)

# State Machine Variable 
initialize = 0 
object_detection = 1
path_planning = 2
motion_detection1 = 3
motion_detection2 = 4
point_calculation = 5
target_shoot = 6
enemy_shoot = 7


# Set initial state
state_machine = initialize

object_detection_success = False
myTurn = None
pause = 0

our_pucks = 0
enemy_pucks = 0
path_available = 0

distance = -1
START = 0
puck1_pos = []
puck2_pos = []
# 
if __name__== "__main__":
    raycaster = RayCast([], [], [], [])
    while True:
        success, img = cap.read()
        imgBoard = ImagePro.getBoard(img,START)
        imgDraw = imgBoard.copy()
        # cv2.imshow("imgDraw", imgDraw)
        

        if START < 5: 
            START += 1 
        else: 
            pass
        
        if state_machine == initialize:
            print("INITIALIZE")
            target_pos = ImagePro.getTargetPosition(imgBoard)
            obstacle_pos = ImagePro.getObstaclePosition(imgBoard)
            myTurn = ImagePro.turnDetermination(imgBoard)

            ImagePro.drawcircles(imgDraw, target_pos)
            ImagePro.drawRect(imgDraw, obstacle_pos)
            cv2.imshow("imgDraw", imgDraw)
            
            print(target_pos)
            print(obstacle_pos)
            print(myTurn)

            raycaster.target=  target_pos[0:2]
            raycaster.obstacles = utility.make_compatible(obstacle_pos)
            print(utility.make_compatible(obstacle_pos))
            
            state_machine = object_detection
            # if myTurn:
                # state_machine =  object_detection
            # else: 
                # state_machine = motion_detection2
                # maskFirstFrame = ImagePro.getFirstFrame(imgBoard,"puck2")
                

        elif state_machine == object_detection:
            print("OBJECT DETECTION")
            # PUCK = puckFind*()
            puck1_pos, puck2_pos = ImagePro.getPuckPositions(imgBoard)
            ImagePro.drawcircles(imgDraw, target_pos)
            ImagePro.drawRect(imgDraw, obstacle_pos)
            ImagePro.drawcircles(imgDraw, puck1_pos)
            ImagePro.drawcircles(imgDraw, puck2_pos)
            cv2.imshow("imgDraw", imgDraw)

            #print(puck1_pos)
            #print(puck2_pos)

            raycaster.our_pucks =  [[row[0], row[1]] for row in puck1_pos]
            raycaster.enemy_pucks =  [[row[0], row[1]] for row in puck2_pos]
            
            our_pucks, enemy_pucks = ImagePro.findNumberPucks(puck1_pos, puck2_pos)

            # if our_pucks == 5 and enemy_pucks == 5:
                # state_machine = point_calculation
            
            state_machine = path_planning


        elif state_machine == path_planning:
            print("hello to the last stage")
            points, distance, _ = raycaster.ray_cast(np.array([0,340]))
            print(points)

            for i  in range(0,len(points)-1):
                cv2.line(imgDraw, points[i], points[i+1], (255,0,0),5)
            
            # if path_available:
                # state_machine = target_shoot
            # else: state_machine = enemy_shoot 
        
        elif  state_machine == motion_detection1:
            print("MOTION DETECTION 1")
            puck1Detected = ImagePro.motionDetection(imgBoard, maskFirstFrame, "puck1")
            ImagePro.drawcircles(imgDraw, target_pos)
            ImagePro.drawRect(imgDraw, obstacle_pos)
            ImagePro.drawcircles(imgDraw, puck1_pos)
            ImagePro.drawcircles(imgDraw, puck2_pos)
            
            if puck1Detected:
                state_machine = motion_detection2
                maskFirstFrame = ImagePro.getFirstFrame(imgBoard,"puck2")

        elif state_machine == motion_detection2:
            print("MOTION DETECTION 2")
            puck2Detected = ImagePro.motionDetection(imgBoard, maskFirstFrame,"puck2")
            ImagePro.drawcircles(imgDraw, target_pos)
            ImagePro.drawRect(imgDraw, obstacle_pos)
            ImagePro.drawcircles(imgDraw, puck1_pos)
            ImagePro.drawcircles(imgDraw, puck2_pos)
            
            if puck2Detected:
                state_machine = object_detection
        
        elif state_machine == point_calculation:
            pass
        
        elif state_machine == target_shoot:
            print("TARGET SHOOT")
            path_planning = 0

            state_machine = motion_detection1
            maskFirstFrame = ImagePro.getFirstFrame(imgBoard,"puck1")
        
        elif state_machine == enemy_shoot:
            print("ENEMY SHOOT")
            path_planning = 0

            state_machine = motion_detection1
            maskFirstFrame = ImagePro.getFirstFrame(imgBoard,"puck1")

        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Dont forget to restate necesary variables