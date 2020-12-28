import numpy as np
import os
import json
import cv2


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


centroid_map = json.load(open('centroid_to_img.json', 'r'))
data = json.load(open('labelled_data.json', 'r'))
vid_id = 'rl6i8bTPk3Q'

real_imgs = [cv2.imread(file) for file in os.listdir(os.path.join('saved_imgs', vid_id))]
quantized_imgs = [cv2.imread(centroid_map[str(centroid)]) for centroid in data[vid_id]]

real_tile = concat_tile([real_imgs[20:25],
                         real_imgs[25:30],
                         real_imgs[30:35],
                         real_imgs[35:40]])
quantized_tile = concat_tile([quantized_imgs[20:25],
                              quantized_imgs[25:30],
                              quantized_imgs[30:35],
                              quantized_imgs[35:40]])
cv2.imwrite('~/satyajit_drive/real_imgs.jpg', real_tile)
cv2.imwrite('~/satyajit_drive/quantized_imgs.jpg', quantized_tile)
