#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:10:17 2018

@author: aakanksha
"""

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

import tkinter as tk
from tkinter.filedialog import askopenfilename

#Open the video file which needs to be processed     
root = tk.Tk()
movieName =  askopenfilename(filetypes=[("Video files","*")])
cap = cv2.VideoCapture(movieName)

nframe =cap.get(cv2.CAP_PROP_FRAME_COUNT)
nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

steps=30
i=0-steps
ncount=0
while(cap.isOpened() & (i<(nframe-steps-1))):
    
  i = i + steps
  ncount=ncount+1
  cap.set(cv2.CAP_PROP_POS_FRAMES,i)
  ret, frame = cap.read()
  
  cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/deepTracker/data/still_images/'+movieName[len(movieName)-19:len(movieName)-4]+'_'+str(i)+'.png',frame)
  
  
print("Done")