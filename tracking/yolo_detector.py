import numpy as np
import os, cv2, sys
import time
from deep_sort.detection import Detection
sys.path.append("..")
from models.yolo_models import get_yolo_sort


class yoloDetector(object):
    """
    This class creates a yolo object detector

    """

    anchors = np.array([[53.57159857, 42.28639429], [29.47927551, 51.27168234], [37.15496912, 26.17125211]])
    obj_thresh=0.5
    nms_thresh=0.4 #0.25
    nb_box=3
    base = 32

    weight_file = '../weights/yolo-v3-coco.h5'

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3          

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))


    def bbox_iou(box1, box2):
        
        intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
        intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
        
        intersect = intersect_w * intersect_h

        w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
        w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
        
        union = w1*h1 + w2*h2 - intersect
        
        return float(intersect) / union


    def __init__(self, width, height):
        self.width = (width // base * base)
        self.height = (height // base * base)
        self.model = get_yolo_sort(width, height)
        self.model.load_weights(weight_file)

    def create_detections(self, image):
            image = cv2.resize(image, (self.width, self.height))
            image = np.expand_dims(image, 0)
            netout = model.predict(image)[0]

            grid_h, grid_w = netout.shape[:2]
            netout = netout.reshape(grid_h,grid_w,nb_box,-1)

            # convert from raw output
            netout[..., :2]  = _sigmoid(netout[..., :2])
            netout[..., 4:]  = _sigmoid(netout[..., 4:])
            netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]

            # process the coordinates
            x = np.linspace(0, grid_w-1, grid_w)
            y = np.linspace(0, grid_h-1, grid_h)

            xv,yv = np.meshgrid(x, y)
            xv = np.expand_dims(xv, -1)
            yv = np.expand_dims(yv, -1)
            xpos =(np.tile(xv, (1,1,3))+netout[...,0]) * IMAGE_W / grid_w 
            ypos =(np.tile(yv, (1,1,3))+netout[...,1]) * IMAGE_H / grid_h
            wpos = np.exp(netout[...,2])
            hpos = np.exp(netout[...,3])


            for b in range(nb_box):
                wpos[...,b] *= anchors[b,0]
                hpos[...,b] *= anchors[b,1]

            objectness = netout[...,5]

            # select only objects above threshold
            indexes = objectness > obj_thresh


            new_boxes = np.column_stack((xpos[indexes]-wpos[indexes]/2, \
                                         ypos[indexes]-hpos[indexes]/2, \
                                         xpos[indexes]+wpos[indexes]/2, \
                                         ypos[indexes]+hpos[indexes]/2, \
                                         objectness[indexes]))

            # do nms 
            sorted_indices = np.argsort(-new_boxes[:,4])
            boxes=new_boxes.tolist()

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if new_boxes[index_i,4] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= nms_thresh:
                        new_boxes[index_j,4] = 0

            new_boxes = new_boxes[new_boxes[:,4]>0]
        detection_list = []
        for row in detection_mat[mask]:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        return detection_list


