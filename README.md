# VideoBERT
This repo reproduces the results of VideoBERT (https://arxiv.org/pdf/1904.01766.pdf). Inspiration was taken from https://github.com/MDSKUL/MasterProject, but this repo tackles video prediction rather than captioning and masked language modeling. Here are all the steps taken:

# Step 1: Download 47k videos from the HowTo100M dataset
Using the HowTo100M dataset https://www.di.ens.fr/willow/research/howto100m/, filter out the cooking videos and download them for feature extraction. The dataset is also used for extracting images for each feature vector. The ids for the videos are contained in the ids.txt file. 

# Step 2: Do feature extraction with the I3D model
The I3D model is used to extract the features for every 1.5 seconds of video while saving the median image of the 1.5 seconds of video as well. I3D model used: https://tfhub.dev/deepmind/i3d-kinetics-600/1. Note that CUDA should be used to decrease the runtime. Here is the usage for the code to run:

```
$ python3 VideoBERT/VideoBERT/I3D/batch_extract.py -h
usage: batch_extract.py [-h] -f FILE_LIST_PATH -r ROOT_VIDEO_PATH -s FEATURES_SAVE_PATH -i IMGS_SAVE_PATH

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_LIST_PATH, --file-list-path FILE_LIST_PATH
                        path to file containing video file names
  -r ROOT_VIDEO_PATH, --root-video-path ROOT_VIDEO_PATH
                        root directory containing video files
  -s FEATURES_SAVE_PATH, --features-save-path FEATURES_SAVE_PATH
                        directory in which to save features
  -i IMGS_SAVE_PATH, --imgs-save-path IMGS_SAVE_PATH
                        directory in which to save images
```

# Step 3: Hierarchical Minibatch K-means
To find the centroids for the feature vectors, minibatch k-means is used hierarchically to save time and memory. After this, the nearest feature vector for each centroid is found, and the corresponding image is chosen to represent tht centroid. To use the hierarchical minibatch k-means independently for another project, consider using the python package hkmeans-minibatch, which is also used in this VideoBERT project (https://github.com/ammesatyajit/hierarchical-minibatch-kmeans).

Here is the usage for the kmeans code:
```
$ python3 VideoBERT/VideoBERT/I3D/minibatch_hkmeans.py -h 
usage: minibatch_hkmeans.py [-h] -r ROOT_FEATURE_PATH -p FEATURES_PREFIX [-b BATCH_SIZE] -s SAVE_DIR -c CENTROID_DIR

optional arguments:
  -h, --help            show this help message and exit
  -r ROOT_FEATURE_PATH, --root-feature_path ROOT_FEATURE_PATH
                        path to folder containing all the video folders with the features
  -p FEATURES_PREFIX, --features-prefix FEATURES_PREFIX
                        prefix that is common between the desired files to read
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch_size to use for the minibatch kmeans
  -s SAVE_DIR, --save-dir SAVE_DIR
                        save directory for hierarchical kmeans vectors
  -c CENTROID_DIR, --centroid-dir CENTROID_DIR
                        directory to save the centroids in
```
Note that after this step the centroids will need to be concatenated for ease of use.

After doing kmeans, the image representing each centroid needs to be found to display the video during inference.
```
$ python3 VideoBERT/VideoBERT/data/centroid_to_img.py
```

# Step 4: Label and group data
Using the centroids, videos are labelled and text captions are punctuated. Using the timestamps for each caption, video ids are extracted and paired with the text captions in the training data file. Captions can be found here: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/

# Step 5: Training
The training data from before is used to train a next token prediction transformer. saved model is used for inference in the next step.

# Step 6: Inference
Model is used for predicting video sequences and results can be seen visually. Note that since the model does uses vector quantized images as tokens, it only understands the actions and approximate background of the scene, not the exact person or dish. Here are some samples:

<p float="center" padding=10px>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40071.jpg" alt="out1" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40171.jpg" alt="out2" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40371.jpg" alt="out3" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-42671.jpg" alt="out4" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-44471.png" alt="out5" width="150"/>
</p>
