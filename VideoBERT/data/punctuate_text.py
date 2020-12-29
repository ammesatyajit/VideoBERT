import json
import os
from punctuator import Punctuator

punc = Punctuator('model.pcl')
captions = json.load(open('raw_caption_superclean.json', 'r'))
vid_ids = os.listdir('saved_imgs')

text = captions[vid_ids]['text']
print(text)
punc_text = punc.punctuate(' '.join(text))
print(punc_text)
