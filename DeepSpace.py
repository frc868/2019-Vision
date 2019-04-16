import libjevois as jevois
import cv2
import numpy as np
import statistics as stat

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

# note that this is how the coordinate plane is layed out for the image
# 0 ---------------- +x
# |
# | 
# | 
# | 
# | 
# |
# +y

class BoundingBox:
    def __init__(self, box):
        self.x = box[0]
        self.y = box[1]
        self.w = box[2]
        self.h = box[3]

    # returns box information (x,y coordinates, width and height)
    def getBox(self):
        return (self.x,self.y,self.w,self.h)

    # draws the bounding box on the image
    def draw(self, img):
        cv2.rectangle(img,(self.x,self.y), \
                     (self.x+self.w,self.y+self.h),(0,255,0),2)

    # returns distance between the two boxes
    @staticmethod
    def distance(boxL, boxR):
        return boxR.x - boxL.x

    # returns the position of the center of the two boxes
    @staticmethod
    def position(boxL, boxR):
        return (boxL.x + BoundingBox.distance(boxL, boxR)/2)

    # returns the ratio of the heights of the two boxes
    @staticmethod
    def angle(boxL, boxR):
        return (boxL.y-boxR.y)/BoundingBox.distance(boxL, boxR)

    # returns distance, position, and h ratio
    @staticmethod
    def calculate(boxL, boxR):
        return BoundingBox.distance(boxL, boxR), \
               BoundingBox.position(boxL, boxR), \
               BoundingBox.angle(boxL, boxR)
        
class FitLine:
    def __init__(self, line):
        self.vx = line[0]
        self.vy = line[1]
        self.lx = line[2]
        self.ly = line[3]
    
    # returns the slope of the line
    def slope(self):
        return float(self.vy/self.vx)

    # returns the line information (x/y slope and x/y intersection)
    def getLine(self):
        return (self.vx,self.vy,self.lx,self.ly)

    # returns the intersection point between two lines
    @staticmethod
    def getIntersection(line0, line1):
        slope0 = line0.slope()
        slope1 = line1.slope()
        x0 = line0.lx
        x1 = line1.lx
        y0 = line0.ly
        y1 = line1.ly
        a = np.array([[-slope0, 1], [-slope1, 1]])
        b = np.array([slope0 + y0, slope1 + y1])

        return np.linalg.solve(a,b)
       
    # draws lines on the image
    def draw(self, img):
        point0 = (self.lx - self.vx*100, self.ly - self.vy*100)
        point1 = (self.lx + self.vx*100, self.ly + self.vy*100)
        cv2.line(img, point0, point1, (255,0,0), 2)


class DetectedObject:
    def __init__(self, contour):
        self.contour = contour
        self.box = BoundingBox(cv2.boundingRect(contour))
        self.line = FitLine(cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01))

    # draws bounding boxes and fitlines
    def draw(self, img):
        self.box.draw(img)
        self.line.draw(img)

    # removes duplicate objects
    @staticmethod
    def remove_dups(objs):
        def xPos(obj):
            return obj.box.x
        objs = sorted(objs, key=xPos)
        
        i = 0
        while (i < len(objs) - 1):
            box0 = objs[i].box
            box1 = objs[i+1].box

            # remove an object if the object is within the average width of two boxes in the x direction
            if (BoundingBox.distance(box0,box1) < (box0.w + box1.w)/2):
                objs.remove(objs[i])
            i += 1
                
        return objs
    
    # returns true if the two objects given are a valid pair of vision targets (based on y-similarity and orientation)
    @staticmethod
    def isValidPair(obj0, obj1):
        line0 = obj0.line
        line1 = obj1.line
        box0 = obj0.box
        box1 = obj1.box
        
        # is the y cordinate of the intersection of the two lines above the 
        # TODO get this working again, as this is a better evaluation method
        # cond1 = FitLine.getIntersection(line0, line1)[1] < (box0.y + box0.h + box1.y + box0.h)/2

        # temporary fix to ensure that we don't select between two actual targets (ship)
        # TODO remove when best solution (above) is implemented)
        # ensures that the slope of line 0 is negative and the slope of line 1 is positive
        # (please note that this is counterintuitive due to image coordinate plane being flipped, see coordinate plane above)
        cond1 = (line0.slope() < 0) and (line1.slope() > 0)

        # evaluate if the two targets are in the same general y-plane
        # the further away we get, the less pixels bounding boxes take up
        # so we need to scale it by distance between the two boxes
        box_dist = BoundingBox.distance(box0, box1)
        #y_deadband = box_dist * 0.25 # tune as needed
        #cond2 = abs(box0.y - box1.y) < y_deadband
        cond2 = True

        return (cond1 and cond2)

# buffer of calculated values
class ValueBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    # adds a value to the buffer
    def addValue(self, value):
        # if the buffer isn't full, simply add the value
        if (len(self.buffer) < self.buffer_size):
            self.buffer.append(value)
        # if the buffer is full, remove the first value, then add
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(value)

    # returns average of the buffer
    def average(self):
        return np.average(self.buffer)

    # returns the median of the buffer
    def median(self):
        return np.median(self.buffer)
    
    # returns the standard deviation of the buffer
    def stdev(self):
        return np.std(self.buffer)

    # returns the "derivative of the buffer" (really just the dy part from dy/dx)
    def deriv(self):
        rarr = []
        for i in range(0, self.buffer_size-1):
            rarr.append((self.buffer[i+1] - self.buffer[i]) / 2)

        return rarr

class VisionSetup:
    def __init__(self, hsvmin, hsvmax, rgbmin, rgbmax, kernel, erode_iter, dilate_iter):
        self.hsvmin = hsvmin
        self.hsvmax = hsvmax
        self.rgbmin = rgbmin
        self.rgbmax = rgbmax
        self.kernel = kernel
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter

    # def hsvmin(self):
    #     return self.hsvmin 

    # def hsvmax(self):
    #     return self.hsvmax

    # def rgbmin(self):
    #     return self.rgbmin

    # def rgbmax(self):
    #     return self.rgbmax

    # def kernel(self):
    #     return self.kernel

    # def erode_iter(self):
    #     return self.hsvmin

    # def dilate_iter(self):
    #     return self.dilate_iter

    @staticmethod
    def center_grove():
        hsvmin = np.array([39,  211, 106])
        hsvmax = np.array([100, 255, 255])
        rgbmin = np.array([0, 0, 0])
        rgbmax = np.array([255, 255, 255])
        kernel = np.ones((2,2),np.uint8)
        erode_iter = 2
        dilate_iter = 5

        return VisionSetup(hsvmin, hsvmax, rgbmin, rgbmax, kernel, erode_iter, dilate_iter)

    @staticmethod
    def devlins():
        hsvmin = np.array([39,  211, 106])
        hsvmax = np.array([100, 255, 255])
        rgbmin = np.array([0, 0, 0])
        rgbmax = np.array([255, 255, 255])
        kernel = np.ones((2,2),np.uint8)
        erode_iter = 2
        dilate_iter = 5

        return VisionSetup(hsvmin, hsvmax, rgbmin, rgbmax, kernel, erode_iter, dilate_iter)

    @staticmethod
    def gyansys():
        hsvmin = np.array([50,  200, 75])
        hsvmax = np.array([100, 255, 255])
        rgbmin = np.array([0, 0, 0])
        rgbmax = np.array([255, 255, 255])
        kernel = np.ones((2,2),np.uint8)
        erode_iter = 2
        dilate_iter = 5

        return VisionSetup(hsvmin, hsvmax, rgbmin, rgbmax, kernel, erode_iter, dilate_iter)

    @staticmethod
    def state():
        #hsvmin = np.array([39,  211, 106])
        hsvmin = np.array([40,  140, 95])
        hsvmax = np.array([90, 255, 255])
        rgbmin = np.array([0, 0, 0])
        rgbmax = np.array([255, 255, 255])
        kernel = np.ones((2,2),np.uint8)
        erode_iter = 1
        dilate_iter = 5

        return VisionSetup(hsvmin, hsvmax, rgbmin, rgbmax, kernel, erode_iter, dilate_iter)

class DeepSpace:
    def __init__(self):
        buffer_size = 5

        # create buffer objects for each value type
        self.distBuffer = ValueBuffer(buffer_size)
        self.posBuffer = ValueBuffer(buffer_size)
        self.hRatioBuffer = ValueBuffer(buffer_size)


    def run(self, inframe):
        text = "868V "
        data = ",,"
        raw = inframe.getCvBGR() # raw, unaltered image
        setup = VisionSetup.state()
        
        # filter by hsv values
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        hsvmin = setup.hsvmin
        hsvmax = setup.hsvmax
        hsvfiltered = cv2.inRange(hsv, hsvmin, hsvmax)

        # filter by rgb values
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        rgbmin = setup.rgbmin
        rgbmax = setup.rgbmax
        rgbfiltered = cv2.inRange(rgb, rgbmin, rgbmax)

        # combine both filtered version into one image
        filtered = cv2.bitwise_and(hsvfiltered, rgbfiltered)
        
        # erode, blur and dialate to remove noise
        kernel = setup.kernel
        eroded = cv2.erode(filtered.copy(), kernel, iterations = setup.erode_iter)
        blurred = cv2.blur(eroded.copy(), (2, 2))
        dilated = cv2.dilate(blurred.copy(), kernel, iterations = setup.dilate_iter)


        # detect edges
        edged = cv2.Canny(dilated.copy(), 30, 200)

        # get contours of image
        cnts, hierarchy = cv2.findContours(edged.copy(), \
                                           cv2.RETR_LIST, \
                                           cv2.CHAIN_APPROX_SIMPLE)
        
        editimg = raw.copy()

        if (cnts is not None) and (len(cnts) > 0):
            # sorts contours by area (largest to smallest) and gets top 6
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]

            # create a detected object for each contour
            objs = [DetectedObject(c) for c in cnts]
            #[obj.draw(editimg) for obj in objs]
            objs = DetectedObject.remove_dups(objs)

            #[obj.draw(editimg) for obj in objs]

            # sort objects by x value
            def xPos(obj):
                return obj.box.x

            objs = sorted(objs, key=xPos)

            # create list of pairs of objects based on fitline's slope
            pairs = []
            for i in range(len(objs)-1):
                obj0 = objs[i]
                obj1 = objs[i+1]
                
                try:
                    if (DetectedObject.isValidPair(obj0, obj1)):
                        pairs.append((objs[i], objs[i+1]))
                except np.linalg.LinAlgError: # "singular matrix"
                    pass

            if (len(pairs) > 0):
                # sort pairs by area of both boxes
                def pairArea(pair):
                    return cv2.contourArea(pair[0].contour) \
                           + cv2.contourArea(pair[1].contour)

                pairs = sorted(pairs, key=pairArea, reverse=True)

                # get objects of top pair
                topL, topR = pairs[0]

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
                
                d = stat.stdev(self.distBuffer.deriv())   < 1.5
                p = stat.stdev(self.posBuffer.deriv())    < 1.5
                h = stat.stdev(self.hRatioBuffer.deriv()) < 1.5
                # 2/3 conditions must be met
                if (h and (d ^ p)) or (d and p): 
                  text = text + "Dist: " + str(dist) + " Pos: " + str(pos) \
                         + " H_Ratio: " + str(h_ratio)
                  data = str(dist) + "," + str(pos) + "," + str(h_ratio)
                  # draw boxes and lines of these objects
                  topL.draw(editimg)
                  topR.draw(editimg)

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
        
