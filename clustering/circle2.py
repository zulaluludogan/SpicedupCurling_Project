import numpy as np
import cv2 as cv


cap = cv.VideoCapture(2)

while True:
    read_ok, img = cap.read()

    img = cv.medianBlur(img,5)
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, w = grey.shape[:]


    circles = cv.HoughCircles(grey,cv.HOUGH_GRADIENT,1.5,20,
                                param1=50,param2=30,minRadius=int(w / 40),maxRadius=int(w / 15))
    
    circles = np.uint16(np.around(circles))
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    try:
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
            cv.putText(img, 'big circle', (i[0]-10, i[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(hsv[i[0],i[1]])
        
        cv.imshow('detected circles',img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    except:
        print("e")
        # cv.imshow('No Circle', img)
    
cap.release()
cv.destroyAllWindows()