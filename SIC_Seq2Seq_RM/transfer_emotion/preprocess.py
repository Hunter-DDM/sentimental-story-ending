import argparse
import utils
import pickle

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('-load_data', required=True,
                    help="input file for the data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-input_vocab_size', type=int, default=50000,
                    help="Size of the input vocabulary")
parser.add_argument('-input_filter', type=int, default=0,
                    help="Maximum input sequence length")
parser.add_argument('-input_trun', type=int, default=50,
                    help="Truncate input sequence length")
parser.add_argument('-input_char', action='store_true', help='character based encoding')

parser.add_argument('-input_suf', default='input',
                    help="the suffix of the input filename")
parser.add_argument('-belong_suf', default='belong',
                    help="the suffix of the belong filename")
parser.add_argument('-emotion_suf', default='emotion',
                    help="the suffix of the emotion filename")
parser.add_argument('-ROC_pre', default='ROC_',
                    help="the prefix of the ROC filename")
parser.add_argument('-SST_pre', default='SST_',
                    help="the prefix of the SST filename")

parser.add_argument('-share', action='store_true', help='share the vocabulary between source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, trun_length, filter_length, char, vocab, size):

    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding='utf8') as f:
        for sent in f.readlines():
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            max_length = max(max_length, len(tokens))
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeDataROC(ROC_input, ROC_belong, inputDicts, save_ROC_input, save_ROC_belong, lim=0):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (ROC_input, ROC_belong))
    inputF = open(ROC_input, encoding='utf8')
    belongF = open(ROC_belong, encoding='utf8')

    inputIdF = open(save_ROC_input + '.id', 'w')
    inputStrF = open(save_ROC_input + '.str', 'w', encoding='utf8')
    belongStrF = open(save_ROC_belong + '.str', 'w', encoding='utf8')

    while True:
        input_line = inputF.readline()
        belong_line = belongF.readline()

        # normal end of file
        if input_line == "" and belong_line == "":
            break

        # source or target does not have same number of lines
        if input_line == "" or belong_line == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        input_line = input_line.strip()
        belong_line = belong_line.strip()

        # source and/or target are empty
        if input_line == "" or belong_line == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        input_line = input_line.lower()
        belong_line = belong_line.lower()

        inputWords = input_line.split() if not opt.input_char else list(input_line)

        if opt.input_filter == 0 or len(input_line.split()) <= opt.input_filter:
            if opt.input_trun > 0:
                inputWords = inputWords[:opt.input_trun]

            srcIds = inputDicts.convertToIdx(inputWords, utils.UNK_WORD)

            inputIdF.write(" ".join(list(map(str, srcIds)))+'\n')

            if not opt.input_char:
                inputStrF.write(" ".join(inputWords) + '\n')
            else:
                inputStrF.write("".join(inputWords) + '\n')
            belongStrF.write(belong_line + '\n')

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    inputF.close()
    belongF.close()
    inputStrF.close()
    belongStrF.close()
    inputIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'inputF': save_ROC_input + '.id', 'belongF': save_ROC_belong + '.str',
            'original_inputF': save_ROC_input + '.str', 'original_belongF': save_ROC_belong + '.str',
            'length': sizes}


def makeDataSST(SST_input, SST_belong, SST_emotion, inputDicts, save_SST_input, save_SST_belong, save_SST_emotion, lim=0):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s & %s ...' % (SST_input, SST_belong, SST_emotion))
    inputF = open(SST_input, encoding='utf8')
    belongF = open(SST_belong, encoding='utf8')
    emotionF = open(SST_emotion, encoding='utf8')

    inputIdF = open(save_SST_input + '.id', 'w')
    inputStrF = open(save_SST_input + '.str', 'w', encoding='utf8')
    belongStrF = open(save_SST_belong + '.str', 'w', encoding='utf8')
    emotionStrF = open(save_SST_emotion + '.str', 'w', encoding='utf8')

    while True:
        input_line = inputF.readline()
        belong_line = belongF.readline()
        emotion_line = emotionF.readline()

        # normal end of file
        if input_line == "" and belong_line == "" and emotion_line == "":
            break

        # source or target does not have same number of lines
        if input_line == "" or belong_line == "" or emotion_line == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        input_line = input_line.strip()
        belong_line = belong_line.strip()
        emotion_line = emotion_line.strip()

        # source and/or target are empty
        if input_line == "" or belong_line == "" or emotion_line == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        input_line = input_line.lower()
        belong_line = belong_line.lower()
        emotion_line = emotion_line.lower()

        inputWords = input_line.split() if not opt.input_char else list(input_line)

        if opt.input_filter == 0 or len(input_line.split()) <= opt.input_filter:
            if opt.input_trun > 0:
                inputWords = inputWords[:opt.input_trun]

            srcIds = inputDicts.convertToIdx(inputWords, utils.UNK_WORD)

            inputIdF.write(" ".join(list(map(str, srcIds)))+'\n')

            if not opt.input_char:
                inputStrF.write(" ".join(inputWords) + '\n')
            else:
                inputStrF.write("".join(inputWords) + '\n')
            belongStrF.write(belong_line + '\n')
            emotionStrF.write(emotion_line + '\n')

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    inputF.close()
    belongF.close()
    emotionF.close()
    inputStrF.close()
    belongStrF.close()
    emotionStrF.close()
    inputIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'inputF': save_SST_input + '.id', 'belongF': save_SST_belong + '.str',
            'emotionF': save_SST_emotion + '.str', 'original_inputF': save_SST_input + '.str',
            'original_belongF': save_SST_belong + '.str', 'original_emotionF': save_SST_emotion + '.str',
            'length': sizes}


def main():

    dicts = {}

    # ====================== load ==========================

    SST_train_input = opt.load_data + opt.SST_pre + 'train.' + opt.input_suf
    SST_train_emotion = opt.load_data + opt.SST_pre + 'train.' + opt.emotion_suf
    SST_train_belong = opt.load_data + opt.SST_pre + 'train.' + opt.belong_suf

    SST_valid_input = opt.load_data + opt.SST_pre + 'valid.' + opt.input_suf
    SST_valid_emotion = opt.load_data + opt.SST_pre + 'valid.' + opt.emotion_suf
    SST_valid_belong = opt.load_data + opt.SST_pre + 'valid.' + opt.belong_suf

    SST_test_input = opt.load_data + opt.SST_pre + 'test.' + opt.input_suf
    SST_test_emotion = opt.load_data + opt.SST_pre + 'test.' + opt.emotion_suf
    SST_test_belong = opt.load_data + opt.SST_pre + 'test.' + opt.belong_suf

    ROC_train_input = opt.load_data + opt.ROC_pre + 'train.' + opt.input_suf
    ROC_train_belong = opt.load_data + opt.ROC_pre + 'train.' + opt.belong_suf

    ROC_valid_input = opt.load_data + opt.ROC_pre + 'valid.' + opt.input_suf
    ROC_valid_belong = opt.load_data + opt.ROC_pre + 'valid.' + opt.belong_suf

    ROC_test_input = opt.load_data + opt.ROC_pre + 'test.' + opt.input_suf
    ROC_test_belong = opt.load_data + opt.ROC_pre + 'test.' + opt.belong_suf

    # ====================== save ==========================

    save_SST_train_input = opt.save_data + opt.SST_pre + 'train.' + opt.input_suf
    save_SST_train_emotion = opt.save_data + opt.SST_pre + 'train.' + opt.emotion_suf
    save_SST_train_belong = opt.save_data + opt.SST_pre + 'train.' + opt.belong_suf

    save_SST_valid_input = opt.save_data + opt.SST_pre + 'valid.' + opt.input_suf
    save_SST_valid_emotion = opt.save_data + opt.SST_pre + 'valid.' + opt.emotion_suf
    save_SST_valid_belong = opt.save_data + opt.SST_pre + 'valid.' + opt.belong_suf

    save_SST_test_input = opt.save_data + opt.SST_pre + 'test.' + opt.input_suf
    save_SST_test_emotion = opt.save_data + opt.SST_pre + 'test.' + opt.emotion_suf
    save_SST_test_belong = opt.save_data + opt.SST_pre + 'test.' + opt.belong_suf

    save_ROC_train_input = opt.save_data + opt.ROC_pre + 'train.' + opt.input_suf
    save_ROC_train_belong = opt.save_data + opt.ROC_pre + 'train.' + opt.belong_suf

    save_ROC_valid_input = opt.save_data + opt.ROC_pre + 'valid.' + opt.input_suf
    save_ROC_valid_belong = opt.save_data + opt.ROC_pre + 'valid.' + opt.belong_suf

    save_ROC_test_input = opt.save_data + opt.ROC_pre + 'test.' + opt.input_suf
    save_ROC_test_belong = opt.save_data + opt.ROC_pre + 'test.' + opt.belong_suf

    input_dict = opt.save_data + 'input.dict'

    print('Building input vocabulary...')
    dicts['input'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
    dicts['input'] = makeVocabulary(SST_train_input, opt.input_trun, opt.input_filter, opt.input_char,
                                    dicts['input'], opt.input_vocab_size)
    dicts['input'] = makeVocabulary(ROC_train_input, opt.input_trun, opt.input_filter, opt.input_char,
                                    dicts['input'], opt.input_vocab_size)

    print('Preparing training ...')
    ROC_train = makeDataROC(ROC_train_input, ROC_train_belong, dicts['input'], save_ROC_train_input, save_ROC_train_belong)
    SST_train = makeDataSST(SST_train_input, SST_train_belong, SST_train_emotion, dicts['input'], save_SST_train_input,
                            save_SST_train_belong, save_SST_train_emotion)

    print('Preparing validation ...')
    ROC_valid = makeDataROC(ROC_valid_input, ROC_valid_belong, dicts['input'], save_ROC_valid_input, save_ROC_valid_belong)
    SST_valid = makeDataSST(SST_valid_input, SST_valid_belong, SST_valid_emotion, dicts['input'], save_SST_valid_input,
                            save_SST_valid_belong, save_SST_valid_emotion)

    print('Preparing test ...')
    ROC_test = makeDataROC(ROC_test_input, ROC_test_belong, dicts['input'], save_ROC_test_input, save_ROC_test_belong)
    SST_test = makeDataSST(SST_test_input, SST_test_belong, SST_test_emotion, dicts['input'], save_SST_test_input,
                            save_SST_test_belong, save_SST_test_emotion)

    print('Saving source vocabulary to \'' + input_dict + '\'...')
    dicts['input'].writeFile(input_dict)

    data = {'ROC_train': ROC_train, 'ROC_valid': ROC_valid,
            'ROC_test': ROC_test, 'SST_train': SST_train,
            'SST_valid': SST_valid, 'SST_test': SST_test,
            'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))


if __name__ == "__main__":
    main()
