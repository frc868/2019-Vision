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

class BoundingBox:
    def __init__(self, box):
        self.x = box[0]
        self.y = box[1]
        self.w = box[2]
        self.h = box[3]

    def getBox(self):
        return (self.x,self.y,self.w,self.h)

    def draw(self, img):
        cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),2)
        
class FitLine:
    def __init__(self, line):
        self.vx = line[0]
        self.vy = line[1]
        self.lx = line[2]
        self.ly = line[3]
    
    def slope(self):
        return float(self.vy/self.vx)

    def getLine(self):
        return (self.vx,self.vy,self.lx,self.ly)

    def draw(self, img):
        point0 = (self.lx - self.vx*100, self.ly - self.vy*100)
        point1 = (self.lx + self.vx*100, self.ly + self.vy*100)
        cv2.line(img, point0, point1, (0,255,0), 2)


class DetectedObject:
    def __init__(self, contour):
        self.contour = contour
        self.box = BoundingBox(cv2.boundingRect(contour))
        self.line = FitLine(cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01))

    def draw(self, img):
        self.box.draw(img)
        self.line.draw(img)

class DeepSpace2:
    
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        
    def process(self, inframe, outframe):
        text = ""
        raw = inframe.getCvBGR()
                
        self.timer.start()
        
        # --- HSV FILTER ---
        hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
        
        #               H    S    V
        min = np.array([50,  210, 180])
        max = np.array([180, 255, 255])
        filtered = cv2.inRange(hsv, min, max)
        
        # --- ERODE/DILATE ---
        kernel = np.ones((2,2),np.uint8)
        eroded = cv2.erode(filtered,kernel,iterations = 1)
        dilated = cv2.dilate(eroded,kernel,iterations = 4)

        edged = cv2.Canny(dilated, 30, 200)

        # --- CONTOURS ---
        cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        rows, cols = dilated.shape[:2]
        
        editimg = raw

        if (cnts is not None) and (len(cnts) > 0):
            objs = [DetectedObject(c) for c in cnts]
            def xPos(obj):
                return obj.box.x

            objs = sorted(objs, key=xPos)
            pairs = []

            slopes = []

            for i in range(len(objs)-1):
                slope0 = int(objs[i].line.slope())
                slope1 = int(objs[i+1].line.slope())
                slopes.append((slope0,slope1))
                if (slope0 < 0 and slope1 > 0):
                    pairs.append((objs[i], objs[i+1]))

            def pairArea(pair):
                return cv2.contourArea(pair[0].contour) + cv2.contourArea(pair[1].contour)

            if (len(pairs) > 0):
                pairs = sorted(pairs, key=pairArea, reverse=True)
                text = str([pairArea(pair) for pair in pairs])
                topPair = pairs[0]


                topPair[0].draw(editimg)
                topPair[1].draw(editimg)
            
        
        outimg = editimg
        
        cv2.putText(outimg, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        outframe.sendCv(outimg)
        