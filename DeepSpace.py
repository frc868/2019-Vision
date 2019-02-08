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
        
    def remove_dups(objs):
        def area(obj):
            return obj.box.w * obj.box.h
        objs = sorted(objs, key=area)
        
        new_objs = []
        
        for i in range(len(objs)):
            obj = objs[i]
            
            dup = False
            for x in range(len(new_objs)):
                box0 = obj.box
                box1 = new_objs[x].box
                if (BoundingBox.distance(box0, box1) < (box0.w + box1.w)/4):
                    dup = True
                    
            if (not dup):
                new_objs.append(obj)
                
        return new_objs

class ValueBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def addValue(self, value):
        if (len(self.buffer) < self.buffer_size):
            self.buffer.append(value)
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(value)

    def average(self):
        return np.average(self.buffer)

    def median(self):
        return np.median(self.buffer)

class DeepSpace:
    def __init__(self):
        buffer_size = 5

        # create buffer objects for each value type
        self.distBuffer = ValueBuffer(buffer_size)
        self.posBuffer = ValueBuffer(buffer_size)
        self.hRatioBuffer = ValueBuffer(buffer_size)

    def run(self, inframe):
        text = ""
        data = ",,"
        raw = inframe.getCvBGR()
        
        # filter by hsv values
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        hsvmin = np.array([53,  144, 41])
        hsvmax = np.array([96, 255, 255])
        hsvfiltered = cv2.inRange(hsv, hsvmin, hsvmax)

        # filter by rgb values
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        rgbmin = np.array([0, 150, 75])
        rgbmax = np.array([100, 255, 255])
        rgbfiltered = cv2.inRange(rgb, rgbmin, rgbmax)

        # combine both filtered version into one image
        filtered = cv2.bitwise_and(hsvfiltered, rgbfiltered)
        
        # erode, blur and dialate to remove noise
        kernel = np.ones((2,2),np.uint8)
        eroded = cv2.erode(filtered.copy(), kernel, iterations = 1)
        #blurred = cv2.blur(eroded.copy(), (2, 2))
        dilated = cv2.dilate(eroded.copy(), kernel, iterations = 2)


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
            objs = DetectedObject.remove_dups([DetectedObject(c) for c in cnts])

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

                if (slope1 - slope0 >= 0):
                    pairs.append((objs[i], objs[i+1]))

            if (len(pairs) > 0):
                # sort pairs by area of both boxes
                def pairArea(pair):
                    return cv2.contourArea(pair[0].contour) \
                           + cv2.contourArea(pair[1].contour)

                pairs = sorted(pairs, key=pairArea, reverse=True)

                # get objects of top pair
                topL, topR = pairs[0]

                # draw boxes and lines of these objects
                topL.draw(editimg)
                topR.draw(editimg)

                # retrieve and store data 
                dist, pos, h_ratio = BoundingBox.calculate(topL.box, topR.box)

                # add newest calculations to respective buffers
                self.distBuffer.addValue(dist)
                self.posBuffer.addValue(pos)
                self.hRatioBuffer.addValue(h_ratio)

                # set the data to send to the median of the buffer
                dist = self.distBuffer.median()
                pos = self.posBuffer.median()
                h_ratio = self.hRatioBuffer.median()

                text = "Dist: " + str(dist) + " Pos: " + str(pos) \
                       + " H_Ratio: " + str(h_ratio)
                data = str(dist) + "," + str(pos) + "," + str(h_ratio)

        # could be set to: raw, {hsv,rgb}filtered, eroded, dilated, edged, editimg
        outframe = editimg

        return outframe, text, data

    def process(self, inframe, outframe):
        # process the image and get the output image and the serial data 
        outimg, text, data = self.run(inframe)

        # write vision calculations on camera
        cv2.putText(outimg, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.5, (255,255,255))

        # send the image
        outframe.sendCv(outimg)

        # send the serial data
        jevois.sendSerial(data)


    def processNoUSB(self, inframe):
        # process the image and get just the serial data
        _, _, data = self.run(inframe)

        # send the serial data
        jevois.sendSerial(data)
        
