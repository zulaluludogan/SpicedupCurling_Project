import cv2
from cvzone.ColorModule import ColorFinder

# cap = cv2.VideoCapture('video1.mp4')
cap = cv2.VideoCapture(0)
# frameCounter = 0 

# colorFinder = ColorFinder(True) # To Decide HSV values of objects "True"

while True:
    # frameCounter += 1
    # if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     frameCounter = 0 
    #     cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    # success, img = cap.read()
    # imgColor, mask = colorFinder.update(img)

    # cv2.imshow("ImageColor",imgColor)
    mm=cv2.imread('img.png')
    cv2.imshow("hee",mm)
    
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()