import linecache
import torch
import torch.utils.data as torch_data
from random import Random
import utils

num_samples = 1


class MonoDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):

        self.srcF = infos['srcF']
        self.original_srcF = infos['original_srcF']
        self.length = infos['length']
        self.infos = infos
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()

        return src, original_src

    def __len__(self):
        return len(self.indexes)


class ROCDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False):

        self.inputF = infos['inputF']
        self.belongF = infos['belongF']
        self.original_inputF = infos['original_inputF']
        self.original_belongF = infos['original_belongF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        input = list(map(int, linecache.getline(self.inputF, index + 1).strip().split()))
        belong = linecache.getline(self.belongF, index + 1).strip()
        original_input = linecache.getline(self.original_inputF, index + 1).strip().split()
        original_belong = linecache.getline(self.original_belongF, index + 1).strip()

        return input, belong, original_input, original_belong

    def __len__(self):
        return len(self.indexes)


class SSTDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False):

        self.inputF = infos['inputF']
        self.belongF = infos['belongF']
        self.emotionF = infos['emotionF']
        self.original_inputF = infos['original_inputF']
        self.original_belongF = infos['original_belongF']
        self.original_emotionF = infos['original_emotionF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        input = list(map(int, linecache.getline(self.inputF, index + 1).strip().split()))
        belong = linecache.getline(self.belongF, index + 1).strip()
        emotion = linecache.getline(self.emotionF, index + 1).strip()
        original_input = linecache.getline(self.original_inputF, index + 1).strip().split()
        original_belong = linecache.getline(self.original_belongF, index + 1).strip()
        original_emotion = linecache.getline(self.original_emotionF, index + 1).strip()

        return input, belong, emotion, original_input, original_belong, original_emotion

    def __len__(self):
        return len(self.indexes)


def splitDataset(data_set, sizes):
    length = len(data_set)
    indexes = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indexes)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(ROCDataset(data_set.infos, indexes[0:part_len]))
        indexes = indexes[part_len:]
    data_sets.append(ROCDataset(data_set.infos, indexes))
    return data_sets


def ROCpadding(data):
    input, belong, original_input, original_belong = zip(*data)

    input_len = [len(s) for s in input]
    # input_pad = torch.zeros(len(input), max(input_len)).long()
    input_pad = torch.zeros(len(input), 50).long()
    for i, s in enumerate(input):
        end = input_len[i]
        input_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    domain_map = {
        'source': 0,
        'target': 1
    }
    score_map = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
    }

    belong_id = [domain_map[x] for x in belong]
    belong_id = torch.LongTensor(belong_id)

    input_len = torch.LongTensor(input_len)

    return input_pad, belong_id, input_len, original_input, original_belong


def SSTpadding(data):
    input, belong, emotion, original_input, original_belong, original_emotion = zip(*data)

    input_len = [len(s) for s in input]
    # input_pad = torch.zeros(len(input), max(input_len)).long()
    input_pad = torch.zeros(len(input), 50).long()
    for i, s in enumerate(input):
        end = input_len[i]
        input_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    domain_map = {
        'source': 0,
        'target': 1
    }
    score_map = {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4
    }

    belong_id = [domain_map[x] for x in belong]
    belong_id = torch.LongTensor(belong_id)
    emotion_id = [float(x) for x in emotion]
    emotion_id = torch.FloatTensor(emotion_id)

    input_len = torch.LongTensor(input_len)

    return input_pad, belong_id, emotion_id, input_len, original_input, original_belong, original_emotion


def ae_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    ae_len = [len(s)+2 for s in src]
    ae_pad = torch.zeros(len(src), max(ae_len)).long()
    for i, s in enumerate(src):
        end = ae_len[i]
        ae_pad[i, 0] = utils.BOS
        ae_pad[i, 1:end-1] = torch.LongTensor(s)[:end-2]
        ae_pad[i, end-1] = utils.EOS

    return src_pad, tgt_pad, ae_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), torch.LongTensor(ae_len), \
           original_src, original_tgt


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / utils.num_samples)

    for i in range(utils.num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i * num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i * num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])

    return split_samples