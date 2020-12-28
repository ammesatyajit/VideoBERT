import os
import json
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vid-id', type=str, required=True,
                    help='video id to show images for')
args = parser.parse_args()

vid_id = args.vid_id


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


centroid_map = json.load(open('centroid_to_img.json', 'r'))
data = json.load(open('labelled_data.json', 'r'))
vid_id = 'oI8CMo-pX7I'

real_imgs = [cv2.imread(os.path.join('saved_imgs', vid_id, file)) for file in sorted(os.listdir(os.path.join('saved_imgs', vid_id)))]
quantized_imgs = [cv2.imread(centroid_map[str(centroid)]) for centroid in data[vid_id]]

real_tile = concat_tile([real_imgs[10*i:10*(i+1)] for i in range(5, 15)])
quantized_tile = concat_tile([quantized_imgs[10*i:10*(i+1)] for i in range(5, 15)])

cv2.imwrite('real_imgs.jpg', real_tile)
cv2.imwrite('quantized_imgs.jpg', quantized_tile)
