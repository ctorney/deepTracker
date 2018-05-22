import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import pickle
sys.path.append("../..") 
from models.yolo_models import get_yolo_coco
from decoder import decode
ROOTDIR = os.path.expanduser('~/workspace/deepTracker/')
ROOTDIR = os.path.expanduser('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/deepTracker/')
image_dir = ROOTDIR + 'data/still_images/'
train_dir = ROOTDIR + 'train/train_images/'


train_images =  glob.glob( image_dir + "*.png" )

width=1920
height=1080

im_size=864 #size of training imageas for yolo

nx = width//im_size
ny = height//im_size

wb_size=64 #size of bounding boxes we're going to create
sz_2=wb_size//2
yolov3 = get_yolo_coco(im_size,im_size)

yolov3.load_weights('../../weights/yolo-v3-coco.h5')
im_num=1
all_imgs = []
for imagename in train_images: 
    im = cv2.imread(imagename)
    print('processing image ' + imagename + ', ' + str(im_num) + ' of 500 ...')
    im_num+=1

    n_count=0
    for x in np.arange(0,width-im_size,im_size):
        for y in np.arange(0,height-im_size,im_size):
            img_data = {'object':[]}
            save_name = train_dir + '/' + os.path.basename(imagename) + '-' + str(n_count) + '.png'
            box_name = train_dir + '/bbox/' + os.path.basename(imagename) + '-' + str(n_count) + '.png'
            img = im[y:y+im_size,x:x+im_size,:]
            cv2.imwrite(save_name, img)
            img_data['filename'] = save_name
            img_data['width'] = im_size
            img_data['height'] = im_size
            n_count+=1
            # use the yolov3 model to predict 80 classes on COCO

            # preprocess the image
            image_h, image_w, _ = img.shape
            new_image = img[:,:,::-1]/255.
            new_image = np.expand_dims(new_image, 0)

            # run the prediction
            yolos = yolov3.predict(new_image)

            boxes = decode(yolos, im_size, box_name, img)
            for b in boxes:
                xmin=b[0]
                xmax=b[2]
                ymin=b[1]
                ymax=b[3]
                obj = {}

                obj['name'] = 'wildebeest'

 #               xmin = point['xcoord'] - x - sz_2
 #               xmax = point['xcoord'] - x + sz_2
 #               ymin = point['ycoord'] - y - sz_2
 #               ymax = point['ycoord'] - y + sz_2

                if xmin<0: continue
                if ymin<0: continue
                if xmax>im_size: continue
                if ymax>im_size: continue
                obj['xmin'] = int(xmin)
                obj['ymin'] = int(ymin)
                obj['xmax'] = int(xmax)
                obj['ymax'] = int(ymax)
                img_data['object'] += [obj]

            all_imgs += [img_data]


#print(all_imgs)
with open(train_dir + '/annotations.pickle', 'wb') as handle:
   pickle.dump(all_imgs, handle)
                        

