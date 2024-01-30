import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


# cap = cv2.VideoCapture(0)
def cluster():
    img = cv2.imread("circle.jpeg")
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 7
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def draw_contour(frame,canny_edge):
    
    contours, hierarchy = cv2.findContours(canny_edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
              
    print("Number of Contours found = " + str(len(contours))) 
    numberofPunks = len(contours)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) 
    
    return numberofPunks

def detect_circle(frame, hsv, frame_gau_blur, lower_range, higher_range):
    # getting the range of blue color in frame
    color_range = cv2.inRange(hsv, lower_range, higher_range)
    res_blue = cv2.bitwise_and(frame_gau_blur,frame_gau_blur, mask = color_range)

    color_s_gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    canny_edge = cv2.Canny(color_s_gray, 50, 240)

    # applying HoughCircles
    h, w = canny_edge.shape[:]
    circles = cv2.HoughCircles(canny_edge,cv2.HOUGH_GRADIENT,1.5, 20, param1=50, param2=30, minRadius=int(w / 40), maxRadius=int(w / 15))
    cv2.imshow('gray', color_s_gray)
    cv2.imshow('canny', canny_edge)
    draw_contour(frame,canny_edge)

    return circles

if __name__ == "__main__":

    if True: #cap.isOpened():
        while(True):
            # ret, frame = cap.read()
            frame = cv2.imread("circle.jpeg", cv2.IMREAD_COLOR)
            # frame = cluster()

            # blurred captured frame
            frame_gau_blur = cv2.GaussianBlur(frame, (11, 11), 2)

            # converting BGR to HSV
            hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_BGR2HSV)

            # define range of red color in HSV
            lower_red = np.array([170, 50, 50])
            higher_red = np.array([180, 255, 255])

            # lower_red = np.array([0, 50, 50])
            # higher_red = np.array([180, 255, 255])

            # the range of blue color in HSV
            lower_puck1 = np.array([10, 100, 50])
            higher_puck1 = np.array([20, 255, 255])

            # define range of puck2 color in HSV
            lower_puck2 = np.array([80, 100, 50])
            higher_puck2 = np.array([130, 255, 255])

            lower_object = np.array([0, 0, 0])
            higher_object = np.array([180, 180, 100])

            target_circle = detect_circle(frame, hsv, frame_gau_blur, lower_red, higher_red)
            puck1_circles = detect_circle(frame, hsv, frame_gau_blur, lower_puck1, higher_puck1)
            puck2_circles = detect_circle(frame, hsv, frame_gau_blur, lower_puck2, higher_puck2)
            object = detect_circle(frame, hsv, frame_gau_blur, lower_object, higher_object)
            
            all_circles = [target_circle, puck1_circles, puck2_circles, object ]

            all_circles = [object ]

            cir_cen = []
            try:
                for circles in all_circles:
                    if circles.any() != None:
                        circles = np.uint16(np.around(circles))
                        largest_circle = max(circles[0, :], key=lambda x: x[2])
                        print(largest_circle)

                        for i in circles[0,:]:
                            # drawing on detected circle and its center
                            # cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                            # cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                            cir_cen.append((i[0],i[1]))

                        
                    print("hello")
                    print(cir_cen)
                    cv2.imshow('circles', frame)

            except:
                cv2.imshow('circles', frame)
                # cv2.imshow('gray', blue_s_gray)
                # cv2.imshow('canny', canny_edge)
        
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    else:
        print('Cam is not open!') 