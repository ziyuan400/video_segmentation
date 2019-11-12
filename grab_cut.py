from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cv2
import time
import scipy.misc
import skvideo.io
import numpy as np
import scipy as scp

STABLE = False       #Assume the frames are stable
REMOTE = False       #Run on batch machine with initial masks
DEBUG = True         #Print a lot of things
QUICK = True         #get a mask with lower resolution
ACC = 5              #How many times lower are the resolutions
class VideoLabel:
    def __init__(self, classes = 5):
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)
        self.vfm = VideoFileManager()
        self.classes = classes
        self.drawing = False
        self.waiting = False
        self.interrupt = False
        self.SCALE = 0.5
        self.acc = ACC
        self.progress = 0
        self.startPoint = 0

        self.setColor()

    def initCapturedVideo(self, cap, name = 'a'):
        self.cap = cap
        self.name = name
        self.shape = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3]
        self.masks = cv2.GC_PR_BGD * np.ones((self.classes, self.shape[0], self.shape[1], self.shape[2]), np.uint8)
        #Ex: 5-Classes * 900-Frames * 480 * 640
        self.m_mask = 200*np.ones((self.shape[0], self.shape[1], self.shape[2], 3), np.uint8)
        #Merged mask Ex: 900-Frames * 480 * 640 * 3
        self.maskeds = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), np.uint8)

        if DEBUG and False:
            self.cap.set(cv2.CAP_PROP_FRAME_COUNT, 5)
            self.shape[0] = 2
       
        r, self.frame = cap.read() 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.startPoint = 0 
        self.progress = 0    

    def videoGrabCut(self, cap = None, name = 'a'):
        self.initCapturedVideo(cap, name)
        def nothing(x):
            pass

        cv2.imshow('frame', self.frame)
        cv2.createTrackbar("P", 'frame', 0, self.shape[0], nothing)
        cv2.createTrackbar("stay", 'frame', 0, 2, nothing)
        cv2.createTrackbar("acc", 'frame', 1, 20, nothing)
        cv2.createTrackbar("Class", 'frame', 0, self.classes - 1, nothing)
        self.refresh(True)
        cv2.setMouseCallback('frame', self.drawMask)

        for self.progress in range(self.startPoint, self.shape[0]): 
            print ("Start handeling video from position",self.progress, ' in ', self.startPoint, "to", self.shape[0])            
            ret, self.frame = cap.read()
            if self.progress == 0:
                if REMOTE:
                    for c in range(0, classes):
                        self.masks[c][0] = cv2.imread("./masks/" + self.name + str(c) +"_mask.png",
                                                       cv2.IMREAD_GRAYSCALE).reshape(self.shape[1], self.shape[2])            
                else:
                    self.waiting = True
            else:
                for i in range(0, self.classes):
                    relation = cv2.getTrackbarPos('stay', 'frame')
                    if relation == 0:
                        #Result is the next mask
                        self.masks[i][self.progress] = self.masks[i][self.progress - 1]
                    elif relation == 1:     
                        #Result mask Probility of the next mask                        
                        fmask = self.masks[i][self.progress - 1]
                        self.masks[i][self.progress] = np.where((fmask==cv2.GC_BGD)|(fmask==cv2.GC_PR_BGD),
                                                       cv2.GC_PR_BGD, cv2.GC_PR_FGD)
                    else:
                        #Next mask are PR backgound
                        self.masks[i][self.progress] = cv2.GC_PR_BGD * np.ones((self.shape[1], self.shape[2]), np.uint8)

            #Draw and press "p" at the first frame 
            if self.waiting:
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('p'):
                        break
                self.waiting = False

            for i in range(0, self.classes):
                if not QUICK:
                    try:
                        self.masks[i][self.progress],self.bgdModel,self.fgdModel = cv2.grabCut(self.frame, 
                                                     self.masks[i][self.progress].reshape(self.shape[1], self.shape[2]),
                                                     None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            #When there are no infomations in a given mask, grab_cut will throw a error
                    except cv2.error:
                        pass
                else:
                    ACC = cv2.getTrackbarPos('acc', 'frame')
                    self.acc = 1 if ACC not in range(0,20) else ACC
                    try:
                        d_frame = cv2.resize(self.frame,
                                         (int(self.shape[2] / self.acc), int(self.shape[1] / self.acc)))
                        d_mask = cv2.resize(self.masks[i][self.progress],
                                         (int(self.shape[2] / self.acc), int(self.shape[1] / self.acc)),
                                          interpolation = cv2.INTER_NEAREST)


                        d_newMask,self.bgdModel,self.fgdModel = cv2.grabCut(d_frame, 
                                                     d_mask,
                                                     None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                        self.masks[i][self.progress] =  cv2.resize(d_newMask, 
                                                                   (self.shape[2], self.shape[1]),
                                                                   interpolation = cv2.INTER_NEAREST)


                    except cv2.error:
                        self.m_mask[self.progress]  = np.ones((self.shape[1], self.shape[2], 3), np.uint8)

                    #print(self.maskeds.shape, self.masks.shape, self.m_mask.shape, d_newMask.shape, self.shape)

            if not REMOTE and cv2.waitKey(1) & 0xFF == ord('q'):
                self.startPoint = -1  
                self.interrupt = True
                break
            if self.progress != cv2.getTrackbarPos('P', 'frame'):
                print("Interupted!  ", self.progress, 'to', cv2.getTrackbarPos('P', 'frame'))
                self.startPoint = cv2.getTrackbarPos('P', 'frame')  
                self.interrupt = True    
                break
            cv2.setTrackbarPos('P', 'frame', self.progress + 1)
            self.refresh(True)
        if self.interrupt:
            return

        self.vfm.get_gt(self.cap, self.m_mask, self.name)         
        if DEBUG:        
            if not os.path.exists('maskedVideo/' ):
                os.makedirs('maskedVideo/' )
            if not os.path.exists('masks/' ):
                os.makedirs('masks/' )

            skvideo.io.vwrite("./maskedVideo/" + self.name + '_masked.avi', self.maskeds)
            skvideo.io.vwrite("./maskedVideo/" + self.name + '_mask.avi', self.m_mask)
            for c in range(0, self.classes):
                cv2.imwrite("./masks/" + self.name + str(c) +"_mask.png",self.masks[c][-1])

        self.startPoint = -1
        cv2.destroyAllWindows()
    def setColor(self):
    #Set colors for each classes
    #Output mask is due to RGB to BGR issue reversed 
        self.colorSet=[[127,0,0]]         #blue-road
        self.colorSet.append([127,127,0]) #h-blue-greenzone
        self.colorSet.append([0,127,127]) #yellow-sky
        self.colorSet.append([127,0,127]) #purple-building
        self.colorSet.append([0,0,127])   #red-others

    def drawMask(self, event, x, y, flags, img):
    #When dragging the mouse, draw a circle of cv2.GC_FGD on the selected mask and cv2.GC_BGD on all other masks
        if event ==cv2.EVENT_LBUTTONUP:
            self.drawing= False
        if event ==cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.waiting = True
        if event ==cv2.EVENT_MOUSEMOVE and self.drawing == True:   
            if x < int(self.shape[2] * self.SCALE) and int (y < self.shape[1] * self.SCALE):
                for c in range(0, self.classes):
                    if c == cv2.getTrackbarPos('Class', 'frame'):
                        cv2.circle(self.masks[c][self.progress], (x*2, y*2), 5,  cv2.GC_FGD, -1)
                    else:
                        cv2.circle(self.masks[c][self.progress], (x*2, y*2), 5,  cv2.GC_BGD, -1)
    
                cv2.circle(self.output, (x, y), 3,  self.colorSet[cv2.getTrackbarPos('Class', 'frame')], -1)
            self.refresh()

    def refresh(self, reComposation = False):
    #Visualization of the masks
        c = cv2.getTrackbarPos('Class', 'frame')
        try:
            if reComposation:  
                for c in range(0, self.classes):
                    gmask = self.masks[c][self.progress]        
                    gmask = np.concatenate((gmask[:, :, np.newaxis], gmask[:, :, np.newaxis], gmask[:, :, np.newaxis]), axis = 2)
                    self.m_mask[self.progress]  = np.where((gmask==cv2.GC_BGD)|(gmask==cv2.GC_PR_BGD),
                                                       self.m_mask[self.progress],self.colorSet[c]).astype('uint8')
                v1 = np.concatenate((self.frame, self.m_mask[self.progress]), axis=1)        
                if DEBUG:
                    self.maskeds[self.progress] = self.m_mask[self.progress]*0.5 + self.frame
                    v2 = np.concatenate((self.frame, self.maskeds[self.progress]), axis=1)
                    h = np.concatenate((v1, v2), axis=0)
                    h = cv2.resize(h,(0,0),h,self.SCALE,self.SCALE)
                    self.output = h 
                else:
                    self.output = cv2.resize(v1,(0,0),v1,self.SCALE,self.SCALE)
        except ValueError:
            pass
        cv2.imshow('frame', self.output)

    def video(self, path = "./VideoData"):
        videos = self.vfm.loadVideos(path)#Load all file names end with .avi, and "_mask" not included
        for v in videos:
            print("Reading " + v)
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                continue
            while True:
                self.videoGrabCut(cap, os.path.basename(v)[:-4])
                if self.startPoint == -1:
                    break
                else:                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.startPoint) 
            cap.release()



class VideoFileManager:
    def loadVideos(self, path = ".", extensions = ['.avi', '3gp'], notin = '_mask', mustin = ''):
    #Load all file names end with .avi, and "_mask" not included
        videoList = []
        for endWith in extensions:
            for f in os.listdir(path):
                if f[-len(endWith):] in endWith and not notin in f and mustin in f:
                    videoList.append(os.path.join(path, f))    
        return videoList   
    def extrackFiles(self, path = ".", extensions = ['.avi', '.3gp'], notin = '_mask', mustin = ''):
        if not os.path.exists('VideoData/' ):
            os.makedirs('VideoData/' )
        videoList = []
        antiReplace = 1
        for endWith in extensions:
            for root, dirs, files in os.walk(path):    
                print(root)
                if root != './VideoData':
                    for f in files:
                        if f[-len(endWith):] == endWith and not notin in f and mustin in f:
                            if f in videoList:
                                newName = os.path.splitext(f)[0] + str(antiReplace) + os.path.splitext(f)[1]
                                antiReplace += 1   
                            else:
                                newName = f
                            videoList.append(newName)
                            os.rename(os.path.join(root, f), os.path.join("VideoData", newName))
        print(videoList,)
        return videoList   

    def get_gt(self, cap, mask, name):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not os.path.exists('training/' ):
            os.makedirs('training/' )
        f = open('train.txt','w') 
        print("Creating training files: ",name)
        for n in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            f.write('training/' + name + str(n) + '.jpg' + ' training/' + name + str(n) + '_mask_gt.png\n')
            r,frame = cap.read()
            cv2.imwrite('training/' + name + str(n) + '.jpg', frame)
            cv2.imwrite('training/' + name + str(n) + '_mask_gt.png', mask[n])
        print("Finish Creating training files: ",name)

    def i_write(self):        
        f = open('train.txt','w')
        names = ["Video12", "20110512_123754", "20110929_201029", "Video(28)", "20110512_123917", "20120203_121636"]
        frames = [850, 195, 299, 896, 299, 299]
        for i in range (0,6):
            for n in range (0,frames[i]):
                f.write('training/' + names[i] + str(n) + '.jpg' + ' training/' + names[i] + str(n) + '_mask_gt.png\n')
        
l = VideoLabel()
l.video()


