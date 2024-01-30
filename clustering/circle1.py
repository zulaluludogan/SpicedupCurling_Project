import cv2
import numpy as np
#  # define range of red color in HSV
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
     
#     # define range of green color in HSV
#     lower_green = np.array([40, 20, 50])
#     upper_green = np.array([90, 255, 255])
     
#     # define range of puck2 color in HSV
#     lower_puck2 = np.array([100, 50, 50])
#     upper_puck2 = np.array([130, 255, 255])
# Below function will read video imgs
cap = cv2.VideoCapture(0)
 
while True:
    read_ok, img = cap.read()
    img_bcp = img.copy()
  
    img = cv2.resize(img, (640, 480))
    # Make a copy to draw contour outline
    input_image_cpy = img.copy()
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # define range of red color in HSV
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
     
    # define range of orange color in HSV
    lower_puck1 = np.array([5, 50, 20])
    upper_puck1 = np.array([25, 255, 255])
     
    # define range of puck2 color in HSV
    lower_puck2 = np.array([80, 100, 50])
    upper_puck2 = np.array([130, 255, 255])
 
    # create a mask for red color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # create a mask for puck1 color
    mask_puck1 = cv2.inRange(hsv, lower_puck1, upper_puck1)
    # create a mask for puck2 color
    mask_puck2 = cv2.inRange(hsv, lower_puck2, upper_puck2)
 
    # find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the puck1 mask
    contours_puck1, _ = cv2.findContours(mask_puck1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the puck2 mask
    contours_puck2, _ = cv2.findContours(mask_puck2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    # loop through the red contours and draw a rectangle around them
    for cnt in contours_red:
        print(contours_red)
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.circle(img, (x + int(w/2), y + int(h/2)), int(w/2), (0, 0, 255), 2)
            cv2.putText(img, 'Target', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 
    # loop through the puck1 contours and draw a rectangle around them
    for cnt in contours_puck1:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.circle(img, (x + int(w/2), y + int(h/2)), int(w/2), (0, 255, 0), 2)
            cv2.putText(img, 'Puck1', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # print(hsv[x + int(w/2)][y + int(h/2)])
    # loop through the puck2 contours and draw a rectangle around them
    for cnt in contours_puck2:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.circle(img, (x + int(w/2), y + int(h/2)), int(w/2), (255, 0, 0), 2)
            cv2.putText(img, 'Puck2', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
  
    cv2.imshow('Color Recognition Output', img)
     
    # Close video window by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break