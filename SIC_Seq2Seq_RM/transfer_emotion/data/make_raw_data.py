import csv
import json
import re
import os
from nltk.tokenize import WordPunctTokenizer 
import random

# ======================== SST ======================

name_score_map = {
    'very pos': '5',
    'pos': '4',
    'neu': '3',
    'neg': '2',
    'very neg': '1'
}

SST_train_input = []
SST_train_emotion = []
SST_train_belong = []
SST_valid_input = []
SST_valid_emotion = []
SST_valid_belong = []
SST_test_input = []
SST_test_emotion = []
SST_test_belong = []

print('SST:')

with open('SST5_data/sst_train_phrases.csv', 'r', encoding='utf-8') as f:
    list_train_csv = list(csv.reader(f))
    list_train_csv = [pair for pair in list_train_csv if len(pair[1].split()) > 3]
    print('train_num:', len(list_train_csv))
    for pair in list_train_csv:
        SST_train_input.append(pair[1])
        SST_train_emotion.append(pair[0])
        SST_train_belong.append('source')

with open('SST5_data/sst_dev.csv', 'r', encoding='utf-8') as f:
    list_valid_csv = list(csv.reader(f))
    print('valid_num:', len(list_valid_csv))
    for pair in list_valid_csv:
        SST_valid_input.append(pair[1])
        SST_valid_emotion.append(pair[0])
        SST_valid_belong.append('source')

with open('SST5_data/sst_test.csv', 'r', encoding='utf-8') as f:
    list_test_csv = list(csv.reader(f))
    print('test_num:', len(list_test_csv))
    for pair in list_test_csv:
        SST_test_input.append(pair[1])
        SST_test_emotion.append(pair[0])
        SST_test_belong.append('source')

with open('raw_data/SST_train.input', 'w', encoding='utf-8') as f:
    for sentence in SST_train_input:
        f.write(sentence + '\n')

with open('raw_data/SST_train.emotion', 'w', encoding='utf-8') as f:
    for score in SST_train_emotion:
        f.write(score + '\n')

with open('raw_data/SST_train.belong', 'w', encoding='utf-8') as f:
    for belong in SST_train_belong:
        f.write(belong + '\n')

with open('raw_data/SST_valid.input', 'w', encoding='utf-8') as f:
    for sentence in SST_valid_input:
        f.write(sentence + '\n')

with open('raw_data/SST_valid.emotion', 'w', encoding='utf-8') as f:
    for score in SST_valid_emotion:
        f.write(score + '\n')

with open('raw_data/SST_valid.belong', 'w', encoding='utf-8') as f:
    for belong in SST_valid_belong:
        f.write(belong + '\n')

with open('raw_data/SST_test.input', 'w', encoding='utf-8') as f:
    for sentence in SST_test_input:
        f.write(sentence + '\n')

with open('raw_data/SST_test.emotion', 'w', encoding='utf-8') as f:
    for score in SST_test_emotion:
        f.write(score + '\n')

with open('raw_data/SST_test.belong', 'w', encoding='utf-8') as f:
    for belong in SST_test_belong:
        f.write(belong + '\n')

# =================== ROC ==================

with open('ROC_data/integrated_data.json', 'r', encoding='utf-8') as f:
    ROC_data = json.load(f)
    tgts = ROC_data['tgts']
    all_number = len(tgts)
    train_number = int(all_number * 0.9)
    valid_number = int(all_number * 0.05)
    test_number = all_number - train_number - valid_number
    print('ROC:')
    print(all_number)
    print(train_number, valid_number, test_number)
    ROC_train_input = tgts[:train_number]
    ROC_valid_input = tgts[train_number:train_number + valid_number]
    ROC_test_input = tgts[train_number + valid_number:]
    ROC_train_belong = ['target' for i in range(train_number)]
    ROC_valid_belong = ['target' for i in range(valid_number)]
    ROC_test_belong = ['target' for i in range(test_number)]

with open('raw_data/ROC_train.input', 'w', encoding='utf-8') as f:
    for sentence in ROC_train_input:
        f.write(sentence + '\n')

with open('raw_data/ROC_train.belong', 'w', encoding='utf-8') as f:
    for belong in ROC_train_belong:
        f.write(belong + '\n')

with open('raw_data/ROC_valid.input', 'w', encoding='utf-8') as f:
    for sentence in ROC_valid_input:
        f.write(sentence + '\n')

with open('raw_data/ROC_valid.belong', 'w', encoding='utf-8') as f:
    for belong in ROC_valid_belong:
        f.write(belong + '\n')

with open('raw_data/ROC_test.input', 'w', encoding='utf-8') as f:
    for sentence in ROC_test_input:
        f.write(sentence + '\n')

with open('raw_data/ROC_test.belong', 'w', encoding='utf-8') as f:
    for belong in ROC_test_belong:
        f.write(belong + '\n')

# ======================== IMDB ======================

# IMDB_input = []
# IMDB_emotion = []
# IMDB_belong = []

# def walk_dir(root_dir):
    # file_names = os.listdir(root_dir)
    # for file_name in file_names:
        # file_path = os.path.join(root_dir, file_name)            
        # if os.path.isdir(file_path):
            # walk_dir(file_path)                  
        # else:
            # with open(file_path, 'r', encoding='utf-8') as f:
                # input = f.read().strip()
                # input = re.sub('<(.*?)>', '', input)
                # input = re.sub('\\((.*?)\\)', '', input)
                # input = ' '.join(WordPunctTokenizer().tokenize(input)[:50])
                # input = input.lower()
                # emotion = file_path.split('/')[-1].split('.')[-2].split('_')[-1]
                # belong = 'source'
                # IMDB_input.append(input)
                # IMDB_emotion.append(emotion)
                # IMDB_belong.append(belong)

# walk_dir('IMDB_data/')
                
# comp = list(zip(IMDB_input, IMDB_emotion, IMDB_belong))
# random.shuffle(comp)
# IMDB_input[:], IMDB_emotion[:], IMDB_belong[:] = zip(*comp)

# IMDB_number = len(IMDB_input)
# print('IMDB:')
# print(IMDB_number)
# IMDB_train_number = int(IMDB_number * 0.9)
# IMDB_valid_number = int(IMDB_number * 0.05)
# IMDB_test_number = IMDB_number - IMDB_train_number - IMDB_valid_number
# print(IMDB_train_number, IMDB_valid_number, IMDB_test_number)

# with open('raw_data/SST_train.input', 'w', encoding='utf-8') as f:
    # for sentence in IMDB_input[:IMDB_train_number]:
        # f.write(sentence + '\n')

# with open('raw_data/SST_train.emotion', 'w', encoding='utf-8') as f:
    # for score in IMDB_emotion[:IMDB_train_number]:
        # f.write(score + '\n')

# with open('raw_data/SST_train.belong', 'w', encoding='utf-8') as f:
    # for belong in IMDB_belong[:IMDB_train_number]:
        # f.write(belong + '\n')

# with open('raw_data/SST_valid.input', 'w', encoding='utf-8') as f:
    # for sentence in IMDB_input[IMDB_train_number:IMDB_train_number + IMDB_valid_number]:
        # f.write(sentence + '\n')

# with open('raw_data/SST_valid.emotion', 'w', encoding='utf-8') as f:
    # for score in IMDB_emotion[IMDB_train_number:IMDB_train_number + IMDB_valid_number]:
        # f.write(score + '\n')

# with open('raw_data/SST_valid.belong', 'w', encoding='utf-8') as f:
    # for belong in IMDB_belong[IMDB_train_number:IMDB_train_number + IMDB_valid_number]:
        # f.write(belong + '\n')

# with open('raw_data/SST_test.input', 'w', encoding='utf-8') as f:
    # for sentence in IMDB_input[IMDB_train_number + IMDB_valid_number:]:
        # f.write(sentence + '\n')

# with open('raw_data/SST_test.emotion', 'w', encoding='utf-8') as f:
    # for score in IMDB_emotion[IMDB_train_number + IMDB_valid_number:]:
        # f.write(score + '\n')

# with open('raw_data/SST_test.belong', 'w', encoding='utf-8') as f:
    # for belong in IMDB_belong[IMDB_train_number + IMDB_valid_number:]:
        # f.write(belong + '\n')
