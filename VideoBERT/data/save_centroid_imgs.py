import os
import json
from tqdm import tqdm
import shutil

centroid_img_file = 'centroid_to_img.json'
save_dir = 'centroid_imgs'

os.mkdir(save_dir)
centroid_map_json = json.load(open(centroid_img_file, 'r'))

for key in tqdm(centroid_map_json):
    shutil.copy(centroid_map_json[key], os.path.join(save_dir, "centroid-{:05}".format(int(key))))
