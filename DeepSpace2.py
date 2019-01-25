import libjevois as jevois
import cv2
import numpy as np

## Detect retroreflective field markings
#
# Add some description of your module here.
#
# @author Armaan Goel, Skyler Cleland, Hazel Levine
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

    def distance(box0, box1):
        return box0.x - box1.x

    def position(box0, box1):
        return box0.x + distance(box0, box1)/2

    def height_difference(box0, box1):
        return box0.h - box1.h

    def calculate(box0, box1):
        return self.distance(box0, box1), self.position(box0, box1), self.height_difference(box0, box1)
        
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
                    
        # filter by hsv values
        hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
        min = np.array([50,  210, 180])
        max = np.array([180, 255, 255])
        filtered = cv2.inRange(hsv, min, max)
        
        # erode and dialate to remove noise
        kernel = np.ones((2,2),np.uint8)
        eroded = cv2.erode(filtered,kernel,iterations = 1)
        dilated = cv2.dilate(eroded,kernel,iterations = 4)

        # detect edges
        edged = cv2.Canny(dilated, 30, 200)

        # get contours of image
        cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        editimg = raw.copy()

        if (cnts is not None) and (len(cnts) > 0):

            # create a detected object for each contour
            objs = [DetectedObject(c) for c in cnts]

            # sort objects by x value
            def xPos(obj):
                return obj.box.x

            objs = sorted(objs, key=xPos)

            # create list of pairs of objects based on fitline's slope
            pairs = []
            for i in range(len(objs)-1):
                slope0 = int(objs[i].line.slope())
                slope1 = int(objs[i+1].line.slope())

                if (slope0 < 0 and slope1 > 0):
                    pairs.append((objs[i], objs[i+1]))

            if (len(pairs) > 0):
                # sort pairs by total area
                def pairArea(pair):
                    return cv2.contourArea(pair[0].contour) + cv2.contourArea(pair[1].contour)

                pairs = sorted(pairs, key=pairArea, reverse=True)

                # get objects of top pair
                top0, top1 = pairs[0]

                # draw boxes and lines of these objects
                top0.draw(editimg)
                top1.draw(editimg)

                # get calculations 
                dist, pos, h_diff = BoundingBox.calculate(top0.box, top1.box)
                text = "Dist: " + str(dist) + " Pos: " + str(pos) + " H_Diff: " + str(h_diff)
            
        
        outimg = editimg # could be set to: raw, filtered, eroded, dialated, edged, editimg
        
        # put text on the image
        cv2.putText(outimg, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        outframe.sendCv(outimg)
        