#!coding=utf-8

from __future__ import print_function
import numpy as np
import scipy.misc
import os
import cv2
import time
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import detector

def main():
    fps = 0.0
    minsize = 20
    threshold = [0.06, 0.6, 0.6]
    factor = 0.709
    
    Pnet ,Rnet ,Onet = detector.initFaceDetector()
    video_capture = cv2.VideoCapture(0)
    
    while True:
        t1 = time.time()
        
        _, img = video_capture.read()
        cv2.imshow("origin",img)
        cv2.waitKey(10)
        
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]     ####  BGR    CHW
        img_matlab[:,:,0] = tmp 
        
        boundingboxes = detector.detect_face(img_matlab, minsize, Pnet, Rnet, Onet, threshold, False, factor)
        detector.drawBoxes(img,boundingboxes)
        cv2.imshow("Pnet",img)
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
