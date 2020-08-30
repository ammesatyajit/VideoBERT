# VideoBERT
This repository contains the code for the project of my master thesis about VideoBERT. The code in this repository is based on code from https://github.com/huggingface/transformers .

# Step 1: Collection of the training data
In this step the videos and text annotations are collected from the HowTo100M dataset. The file step1 / ids.txt contains all ids of the 47470 videos that were included in the workout data. The annotations can be viewed at https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/ .

# Step 2: Transformation of the data
In this step, the videos are transformed by adjusting the frame rate to 10 fps and adding punctuation to the text. For the text, the trained punctuation models can be consulted at https://drive.google.com/drive/folders/0B7BsN5f2F1fZQnFsbzJ3TWxxMms .

# Step 3: Extraction of the I3D characteristics
The I3D characteristics of the videos are constructed in this step using the I3D network. The folder step3 / checkpoint contains the original Tensorflow checkpoint for the I3D model.

# Step 4: Clustering the I3D features
In this step the I3D features are grouped by hierarchical k-means. The best results were obtained when k = 12 and h = 4 . The file containing the cluster centroids can be found at https://drive.google.com/file/d/1i1mDYTnY-3SIkehEDGT5ip_xj0wXIZOr/view?usp=sharing .

# Step 5: Convert BERT to VideoBERT
The starting point of VideoBERT is the BERT model. The state_dict of the trained BERT model can be adjusted in Pytorch to take into account the new vocabulary. In addition, a new class VideoBertForPreTraining was also constructed to realize the training regimes and input modalities .

# Step 6: Training the model
In the last step, the model was trained. They experimented with a model that does not take into account the new proposed alignment task, as well as a model that does take this into account. The processed training data can be viewed at https://drive.google.com/file/d/1nlXQuRdzpsF9V95D8zPOnZz5miOw3FpV/view?usp=sharing .

# Evaluation
The YouCookII validation dataset was used for the evaluation of the model. The trained model achieves similar results to the original model on a zero-shot classification task. The lists for the verbs and nouns can be found in evaluation / verbs.txt and evaluation / nouns.txt . The ground-truth YouCookII linguistic and visual sentences file along with the verbs and nouns can be found at https://drive.google.com/file/d/1hxbiS3mrQdJLkXsPo23dwl4m-dnCMcfV/view?usp=sharing .


# Bronnen
De belangrijkste bronnen zijn:
  - VideoBERT paper: https://arxiv.org/pdf/1904.01766.pdf
  - Hierarchical k-means paper: https://www.andrew.cmu.edu/user/hgifford/projects/k_means.pdf
  - HuggingFace: https://github.com/huggingface/transformers
  - I3D model: https://github.com/deepmind/kinetics-i3d
  - RevoScaleR: https://docs.microsoft.com/en-us/machine-learning-server/r-reference/revoscaler/revoscaler
  - FFmpeg: https://ffmpeg.org/
  - Youtube-dl: https://youtube-dl.org/
  - HowTo100M data: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/
  - YouCookII: http://youcook2.eecs.umich.edu/
  - Punctuator2 repository: https://github.com/ottokart/punctuator2
  - punctuator module: https://pypi.org/project/punctuator/
