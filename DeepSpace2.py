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
        raw = inframe.getCvBGR()
                
        self.timer.start()
        
        # convert to hsv colorspace
        hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
        
        # filter based on hsv vales
        min = np.array([45, 50, 180])
        max = np.array([125, 255,255])
        filtered = cv2.inRange(hsv, min, max)
        
            # kernel = np.ones((5,5),np.uint8)
            # eroded = cv2.erode(filtered,kernel,iterations = 2)
            # dilated = cv2.dilate(eroded,kernel,iterations = 4)

        # detect edges        
        edged = cv2.Canny(filtered, 30, 200)
        
        # detect contours
        cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if (cnts is not None):
            # find 10 contours of biggest area 
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        
            text = "No object detected"
            
            if (len(cnts) > 0):
                # draw bounding box around object
                x,y,w,h = cv2.boundingRect(cnts[0])
                cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
                text = "position: (" + str(x) + "," + str(y) + "), height: " + str(h) + ", width: " + str(w)
        
        outimg = raw
        
        cv2.putText(outimg, "JeVois DeepSpace2", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        outframe.sendCv(outimg)
        
