import json
import pprint
import string
from nltk.tokenize import sent_tokenize
import nltk
from punctuator import Punctuator


punc = Punctuator('model.pcl')
nltk.download('punkt')

file = 'captions.txt'

def parse_line(line):
  parts = line.split(" ")
  if len(parts) > 1:
    id = parts[0]
    val = " ".join(parts[1:])
    return id,val


with open(file, 'r') as fd:
  while True:
    line = fd.readline()
    if not line:
      break

    id, valstr = parse_line(line)
    val = json.loads(valstr)
    text = val.get("text")
    text = " ".join(text)
    text = text.replace("[Music]", " ").replace("\r", " ").replace("\n", " ")
    text = ' '.join(text.split())
    sentences = sent_tokenize(text)

    punc_text = None
    punc_needed = True

    if len(sentences) > 2:
      punc_needed = False
      punc_text = text
    else:
      punc_text = punc.punctuate(text)

    
