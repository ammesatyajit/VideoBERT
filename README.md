# VideoBERT
This repo reproduces the results of VideoBERT. Here are all the steps taken so far below:

# Step 1: Download 47k videos from the HowTo100M dataset
Using the HowTo100M dataset https://www.di.ens.fr/willow/research/howto100m/, filter out the cooking videos and download them for feature extraction. The dataset is also used for finding images for each feature vector.

# Step 2: Do feature extraction with the I3D model
The I3D model is used to extract the features for every 1.5 seconds of video while saving the median image of the 1.5 seconds of video as well. I3D model used: https://tfhub.dev/deepmind/i3d-kinetics-600/1

# Step 3: Hierarchical Minibatch K-means
To find the centroids for the feature vectors, minibatch k-means is used in a hierarchy to save time and memory. After this, the nearest feature vector for each centroid is found, and the corresponding image is chosen to represent tht centroid.

# Step 4: Label and group data
Using the centroids, videos are labelled and text captions are punctuated. Using the timestamps for each caption, video ids are extracted and paired with the text captions in the training data file. Captions can be found here: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/

# Step 5: Training
The training data from before is used to train a next token prediction transformer. saved model is used for inference in the next step.

# Step 6: Inference
Model is used for predicting video sequences and results can be seen visually. Note that since the model does uses vector quantized images as tokens, it only understands the actions and approximate background of the scene, not the exact person or dish. Here are some samples:

.column {
  flex: 33.33%;
  padding: 5px;
}
<img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40071.jpg" alt="out1" width="100"/>
<img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40171.jpg" alt="out2" width="100"/>
<img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40371.jpg" alt="out3" width="100"/>
<img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-42671.jpg" alt="out4" width="100"/>
<img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-44471.png" alt="out5" width="100"/>
