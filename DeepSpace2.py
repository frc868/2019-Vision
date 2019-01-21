import libjevois as jevois
import cv2
import numpy as np

## Detect retroreflective field markings
#
# Add some description of your module here.
#
# @author Armaan Goel
# 
# @videomapping YUYV 320 240 30 YUYV 320 240 30 TechHOUNDS DeepSpace2
# @email armaangoel78@gmail.com
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Armaan Goel
# @mainurl techhounds.org
# @supporturl techhounds.org
# @otherurl techhounds.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

class DeepSpace2:
    
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        
    def process(self, inframe, outframe):
        text = ""
        raw = inframe.getCvBGR()
                
        self.timer.start()
        
        hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
        
        min = np.array([81,  0, 220])
        max = np.array([110, 19, 255])
        filtered = cv2.inRange(hsv, min, max)
        
            # kernel = np.ones((5,5),np.uint8)
            # eroded = cv2.erode(filtered,kernel,iterations = 2)
            # dilated = cv2.dilate(eroded,kernel,iterations = 4)

        edged = cv2.Canny(filtered, 30, 200)

             
        cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if (cnts is not None):
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

            box_one = cv2.boundingRect(cnts[0])
            box_two = None

            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if (np.absolute(box_one[0] - x) > box_one[2]):
                    box_two = (x,y,w,h)
                    break

            x0,y0,w0,h0 = box_one
            x1,y1,w1,h1 = box_two

            cv2.rectangle(raw,(x0,y0),(x0+w0,y0+h0),(0,255,0),2)
            cv2.rectangle(raw,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

            dist = np.absolute(x0-x1)
            mid = np.minimum(x0,x1) + dist/2

            text = "dist: " + str(dist) + " mid: " + str(mid)
            
            # if (len(cnts) > 0):
            #     # draw bounding box around object
            #     x,y,w,h = cv2.boundingRect(cnts[0])
            #     cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
            #     text = "position: (" + str(x) + "," + str(y) + "), height: " + str(h) + ", width: " + str(w)

            #     x,y,w,h = cv2.boundingRect(cnts[1])
            #     cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
            #     text = "position: (" + str(x) + "," + str(y) + "), height: " + str(h) + ", width: " + str(w)
        
        outimg = raw
        
        cv2.putText(outimg, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        outframe.sendCv(outimg)
        
