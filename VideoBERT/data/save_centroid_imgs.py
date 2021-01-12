import os
import json
import tqdm
import shutil

centroid_img_file = 'centroid_to_img.json'
save_dir = 'centroid_imgs'

os.mkdir(save_dir)
centroid_map_json = json.load(open(centroid_img_file, 'r'))

for key in centroid_map_json:
    print(key, centroid_map_json[key])
    shutil.copy(centroid_map_json[key], os.path.join(save_dir, "centroid-{:05}".format(int(key))))
