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
$ python3 VideoBERT/VideoBERT/data/centroid_to_img.py -h 
usage: centroid_to_img.py [-h] -f ROOT_FEATURES -i ROOT_IMGS -c CENTROID_FILE -s SAVE_FILE

optional arguments:
  -h, --help            show this help message and exit
  -f ROOT_FEATURES, --root-features ROOT_FEATURES
                        path to folder containing all the video folders with the features
  -i ROOT_IMGS, --root-imgs ROOT_IMGS
                        path to folder containing all the video folders with the images corresponding to the features
  -c CENTROID_FILE, --centroid-file CENTROID_FILE
                        the .npy file containing all the centroids
  -s SAVE_FILE, --save-file SAVE_FILE
                        json file to save the centroid to image dictionary in
```

# Step 4: Label and group data
Using the centroids, videos are tokenized and text captions are punctuated. Using the timestamps for each caption, video ids are extracted and paired with the text captions in the training data file. Captions can be found here: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/. 

The python file below tokenizes the videos:
```
$ python3 VideoBERT/VideoBERT/data/label_data.py -h     
usage: label_data.py [-h] -f ROOT_FEATURES -c CENTROID_FILE -s SAVE_FILE

optional arguments:
  -h, --help            show this help message and exit
  -f ROOT_FEATURES, --root-features ROOT_FEATURES
                        path to folder containing all the video folders with the features
  -c CENTROID_FILE, --centroid-file CENTROID_FILE
                        the .npy file containing all the centroids
  -s SAVE_FILE, --save-file SAVE_FILE
                        json file to save the labelled data to
```

After that the following file can be run to both punctuate text and group the text with the corresponding video. This uses the Punctuator module, which requires a .pcl model file to punctuate the data. 
```
$ python3 VideoBERT/VideoBERT/data/punctuate_text.py -h 
usage: punctuate_text.py [-h] -c CAPTIONS_PATH -p PUNCTUATOR_MODEL -l LABELLED_DATA -f ROOT_FEATURES -s SAVE_PATH

optional arguments:
  -h, --help            show this help message and exit
  -c CAPTIONS_PATH, --captions-path CAPTIONS_PATH
                        path to filtered captions
  -p PUNCTUATOR_MODEL, --punctuator-model PUNCTUATOR_MODEL
                        path to punctuator .pcl model
  -l LABELLED_DATA, --labelled-data LABELLED_DATA
                        path to labelled data json file
  -f ROOT_FEATURES, --root-features ROOT_FEATURES
                        directory with all the video features
  -s SAVE_PATH, --save-path SAVE_PATH
                        json file to save training data to
```
If desired, an evaluation data file can be created by splitting the training data file.

# Step 5: Training
The training data from before is used to train a next token prediction transformer. The saved model and tokenizer is used for inference in the next step. here is the usage of the train.py file.

```
$ python3 VideoBERT/VideoBERT/train/train.py -h
usage: train.py [-h] --output_dir OUTPUT_DIR [--should_continue] [--model_name_or_path MODEL_NAME_OR_PATH] [--train_data_path TRAIN_DATA_PATH] [--eval_data_path EVAL_DATA_PATH] [--config_name CONFIG_NAME] [--block_size BLOCK_SIZE]
                [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
                [--max_grad_norm MAX_GRAD_NORM] [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS] [--log_dir LOG_DIR] [--warmup_steps WARMUP_STEPS] [--local_rank LOCAL_RANK] [--logging_steps LOGGING_STEPS]
                [--save_steps SAVE_STEPS] [--save_total_limit SAVE_TOTAL_LIMIT] [--overwrite_output_dir] [--overwrite_cache] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --should_continue     Whether to continue from latest checkpoint in output_dir
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
  --train_data_path TRAIN_DATA_PATH
                        The json file for training the model
  --eval_data_path EVAL_DATA_PATH
                        The json file for evaluating the model
  --config_name CONFIG_NAME
                        Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.
  --block_size BLOCK_SIZE
                        Optional input sequence length after tokenization.The training dataset will be truncated in block of this size for training.Default to the model max input length for single sentence inputs (take into account
                        special tokens).
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform. Override num_train_epochs.
  --log_dir LOG_DIR     Directory to store the logs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default
  --overwrite_output_dir
                        Overwrite the content of the output directory
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --seed SEED           random seed for initialization
```

# Step 6: Inference
Model is used for predicting video sequences and results can be seen visually. Note that since the model does uses vector quantized images as tokens, it only understands the actions and approximate background of the scene, not the exact person or dish. Here are some samples:

<p float="center" padding=10px>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40071.jpg" alt="out1" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40171.jpg" alt="out2" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-40371.jpg" alt="out3" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-42671.jpg" alt="out4" width="150"/>
  <img src="https://github.com/ammesatyajit/videobert/blob/master/results/out-vid-44471.png" alt="out5" width="150"/>
</p>

Here is the usage for the inference file. Feel free to modify it to suit any specific needs:

```
$ python3 VideoBERT/VideoBERT/evaluation/inference.py -h 
usage: inference.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] --output_dir OUTPUT_DIR [--example_id EXAMPLE_ID] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
  --output_dir OUTPUT_DIR
                        The output directory where the checkpoint is.
  --example_id EXAMPLE_ID
                        The index of the eval set for evaluating the model
  --seed SEED           random seed for initialization
```
