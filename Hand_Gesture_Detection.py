import numpy as np
import cv2
import math
import time
import collections

roi_count = 0
last_roi = None
state = True
state_count = 10
frame_no = 0
gesture_list = []


def fastest_calc_dist(x,y):
    return math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(x,y)]))

def detectGesture(src,roi_rect,thresh):
    global gesture_list
    p,q,r,s = roi_rect
    image, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    line_count = 0;

    if len(contours):
        # find contour with max area
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(src, (p+x, q+y), (p+x+w, q+y+h), (0, 0, 255), 0)
        
        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(src.shape,np.uint8)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0

        # applying Cosine Rule to find angle for all defects (between fingers)

        if(defects is not None and len(defects) >0):
            if(defects.shape[0] != None):
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]

                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    ls = [[start[0], start[1]],[end[0] ,end[1]],[far[0] ,far[0]]]
                    ctr = np.array(ls).reshape((-1,1,2)).astype(np.int32)

                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                    # ignore angles > 90 and highlight rest with red dots
                    if angle <= 90 and cv2.contourArea(ctr) > 2000:
                        count_defects += 1
                        far = (far[0] +p ,far[1] + q)
                        cv2.circle(src, far, 1, [0,0,255], -1)


                    start = (start[0] +p ,start[1] + q)
                    end = (end[0] +p ,end[1] + q)
                    
                    cv2.line(src,start, end, [0,255,0], 2)
                    
                gesture_list.append(count_defects+1)

        cv2.imshow('frame',src)

def gestureRecognition(src,roi_rect,hand_roi):
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    converted = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
     
    #cv2.imshow('converted',converted)   
    #cv2.imshow('skinMask',skinMask)

    kernel = np.ones((5,5), np.uint8)

    #img_erosion = cv2.erode(skinMask, kernel, iterations=1)
    img_dilated = cv2.dilate(skinMask, kernel, iterations=3)
    #cv2.imshow('img_dilation',img_dilation)

    detectGesture(src,roi_rect,img_dilated)

def getROI(frame,cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rois = cascade.detectMultiScale(gray, 1.3, 5)
    if(len(rois) > 0):
        return rois[0]
    
    return None


hand_cascade = cv2.CascadeClassifier('aGest.xml')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # get the RoI of the plam, None will be returned if there is no palm detected 
    roi = getROI(frame,hand_cascade)

    
    if not(roi is None):
        # check consecutive 10 palm detections to detect event detection 
        if(roi_count != 10):
            roi_count = roi_count + 1

        if(roi_count == 10):
            cv2.putText(frame,"Event Detected", (50, 50),cv2.FONT_HERSHEY_SIMPLEX,2, 1, 2) # will show if event detected
            

        last_roi = roi
        x,y,w,h = roi
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        state = True
    else:

        if(not state and state_count > 0):
            # update the boundry to detect hand
            x,y,w,h = last_roi
            x = x - w
            y = y - h
            w = w * 3
            h = h * 2
            if(x < 0): 
                x = 0
            if(y < 0): 
                y = 0
                
            state_count = state_count - 1

            roi_rect = x,y,w,h
            # call hand gesture recognition
            gestureRecognition(frame,roi_rect,frame[y:y+h, x:x+w])

        if(not state and state_count == 0):
            #print gesture_list
            if(len(gesture_list) > 0):
                gesture = collections.Counter(gesture_list).most_common(1)[0][0]
                print gesture
                cv2.putText(frame,str(gesture), (70, 70),cv2.FONT_HERSHEY_SIMPLEX,3, 2, 2)
                gesture_list = []
                state = True
                    
        if(roi_count > 0):
            roi_count = roi_count - 1

        if(roi_count == 5):
            state = False
            state_count = 10
    
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
