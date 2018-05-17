import os

import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_detector import yoloDetector
from deep_sort.tracker import Tracker

sys.path.append("..")
from models.yolo_models import get_yolo_sort



input_file = 'test.avi'
output_file = 'out.avi'

width = 1920
height = 1080
max_cosine_distance = 0.2
display = True


metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance)
tracker = Tracker(metric)
yolo = yoloDetector(width,height)
results = []

cap = cv2.VideoCapture(0)

frame_idx=0
while True:
    ret, frame = cap.read() 
    if ret != True:
        break;

    detections = yolo.create_detections(frame)
    # Update tracker.
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() and track.time_since_update >1 :
            continue 
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx+=1

    if display:
        cv2.imshow('', frame)


