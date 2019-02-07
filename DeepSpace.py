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
        cv2.rectangle(img,(self.x,self.y), \
                     (self.x+self.w,self.y+self.h),(0,255,0),2)

    def distance(boxL, boxR):
        return boxR.x - boxL.x

    def position(boxL, boxR):
        return (boxL.x + BoundingBox.distance(boxL, boxR)/2)

    def height_ratio(boxL, boxR):
        return boxL.h/boxR.h

    def calculate(boxL, boxR):
        return BoundingBox.distance(boxL, boxR), \
               BoundingBox.position(boxL, boxR), \
               BoundingBox.height_ratio(boxL, boxR)
        
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
        cv2.line(img, point0, point1, (255,0,0), 2)


class DetectedObject:
    def __init__(self, contour):
        self.contour = contour
        self.box = BoundingBox(cv2.boundingRect(contour))
        self.line = FitLine(cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01))

    def draw(self, img):
        self.box.draw(img)
        self.line.draw(img)

class DeepSpace:
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)

    def run(self, inframe):
        text = ""
        data = ",,"
        raw = inframe.getCvBGR()
        
        blurred = cv2.blur(raw,(2, 2))

                    
        # filter by hsv values
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        hmin = np.array([53, 144, 41])
        hmax = np.array([96, 255, 255])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        hfiltered = cv2.inRange(hsv, hmin, hmax)
        
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        rmin = np.array([0, 0, 100])
        rmax = np.array([20, 255, 200])
        rfiltered = cv2.inRange(rgb, rmin, rmax)
        
        hrmask = cv2.bitwise_or(hfiltered, rfiltered)
        filtered = cv2.bitwise_and(raw, raw, mask=hrmask)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # erode and dialate to remove noise
        kernel = np.ones((2,2),np.uint8)
        eroded = cv2.erode(filtered.copy(),kernel,iterations = 2)
        blurred = cv2.blur(eroded.copy(), (2, 2))
        dilated = cv2.dilate(blurred.copy(),kernel,iterations = 2)

        # detect edges
        edged = cv2.Canny(dilated.copy(), 30, 200)

        # get contours of image
        cnts, hierarchy = cv2.findContours(edged.copy(), \
                                           cv2.RETR_LIST, \
                                           cv2.CHAIN_APPROX_SIMPLE)
        
        editimg = raw.copy()

        if (cnts is not None) and (len(cnts) > 0):
            # sorts contours by area (largest to smallest) and gets top 4
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

            # create a detected object for each contour
            objs = [DetectedObject(c) for c in cnts]

            #[obj.draw(editimg) for obj in objs]

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
                # sort pairs by area of both boxes
                def pairArea(pair):
                    return cv2.contourArea(pair[0].contour) \
                           + cv2.contourArea(pair[1].contour)

                pairs = sorted(pairs, key=pairArea, reverse=True)

                # get objects of top pair
                top0, top1 = pairs[0]

                if (top0.box.x < top1.box.x):
                    topL = top0
                    topR = top1
                else:
                    topL = top1
                    topR = top0

                # draw boxes and lines of these objects
                topL.draw(editimg)
                topR.draw(editimg)

                # retrieve and store data 
                dist, pos, h_ratio = BoundingBox.calculate(topL.box, topR.box)
                text = "Dist: " + str(dist) + " Pos: " + str(pos) \
                       + " H_Ratio: " + str(h_ratio)
                data = str(dist) + "," + str(pos) + "," + str(h_ratio)

        # could be set to: raw, filtered, eroded, dilated, edged, editimg
        outframe = editimg

        return outframe, text, data

    def process(self, inframe, outframe):
        outimg, text, data = self.run(inframe)
        cv2.putText(outimg, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.5, (255,255,255))
        outframe.sendCv(outimg)
        jevois.sendSerial(data)


    def processNoUSB(self, inframe):
        _, _, data = self.run(inframe)
        jevois.sendSerial(data)
        
