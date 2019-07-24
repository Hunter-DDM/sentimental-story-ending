import json
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import random

with open('story_data/integrated_data.json', 'r', encoding='utf-8') as f:
    integrated_data = json.load(f)
    srcs = integrated_data['srcs']
    tgts = integrated_data['tgts']
    tgt_scores = integrated_data['tgt_scores']

data_size = len(srcs)
train_num = int(data_size * 0.9)
val_num = int(data_size * 0.05)
test_num = data_size - train_num - val_num

print (train_num, val_num, test_num)

with open('story_data/train.src', 'w', encoding='utf-8') as f: 
    for i in range(0, train_num):
        f.write(srcs[i] + '\n')

with open('story_data/train.tgt', 'w', encoding='utf-8') as f: 
    for i in range(0, train_num):
        f.write(tgts[i] + '\n')

with open('story_data/train.tgt.emotion', 'w', encoding='utf-8') as f: 
    for i in range(0, train_num):
        f.write(str(tgt_scores[i]) + '\n')

with open('story_data/valid.src', 'w', encoding='utf-8') as f: 
    for i in range(train_num, train_num + val_num):
        f.write(srcs[i] + '\n')
        # f.write(srcs[i] + '\n')
        # f.write(srcs[i] + '\n')
        # f.write(srcs[i] + '\n')
        # f.write(srcs[i] + '\n')

with open('story_data/valid.tgt', 'w', encoding='utf-8') as f: 
    for i in range(train_num, train_num + val_num):
        f.write(tgts[i] + '\n')
        # f.write(tgts[i] + '\n')
        # f.write(tgts[i] + '\n')
        # f.write(tgts[i] + '\n')
        # f.write(tgts[i] + '\n')

with open('story_data/valid.tgt.emotion', 'w', encoding='utf-8') as f: 
    for i in range(train_num, train_num + val_num):
        f.write(str(tgt_scores[i]) + '\n')
        # f.write(str(0.0) + '\n')
        # f.write(str(0.3) + '\n')
        # f.write(str(0.5) + '\n')
        # f.write(str(0.7) + '\n')
        # f.write(str(1.0) + '\n')

with open('story_data/test.src', 'w', encoding='utf-8') as f: 
    for i in range(train_num + val_num, train_num + val_num + test_num):
        f.write(srcs[i] + '\n')
        f.write(srcs[i] + '\n')
        f.write(srcs[i] + '\n')
        f.write(srcs[i] + '\n')
        f.write(srcs[i] + '\n')

with open('story_data/test.tgt', 'w', encoding='utf-8') as f: 
    for i in range(train_num + val_num, train_num + val_num + test_num):
        f.write(tgts[i] + '\n')
        f.write(tgts[i] + '\n')
        f.write(tgts[i] + '\n')
        f.write(tgts[i] + '\n')
        f.write(tgts[i] + '\n')

with open('story_data/test.tgt.emotion', 'w', encoding='utf-8') as f: 
    for i in range(train_num + val_num, train_num + val_num + test_num):
        # f.write(str(tgt_scores[i]) + '\n')
        f.write(str(0.0) + '\n')
        f.write(str(0.3) + '\n')
        f.write(str(0.5) + '\n')
        f.write(str(0.7) + '\n')
        f.write(str(1.0) + '\n')
    