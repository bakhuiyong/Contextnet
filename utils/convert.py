import json
from utils.label_loader import *


char2index = dict()
index2char = dict()

from utils.label_loader import *
label_path = '/home/dkdlenrh/contextnet_ctc/data/kor_syllable.json'

char2index, index2char = load_label_json(label_path)

SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']
UNK_token = None

voca_size = len(char2index)+1

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '_'
UNK_TOKEN = None
