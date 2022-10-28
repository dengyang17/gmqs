import argparse
import pickle

import utils

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('--load_data', required=True,
                    help="input file for the data")

parser.add_argument('--save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('--vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")

parser.add_argument('--src_trun', type=int, default=0,
                    help="Truncate source sequence length")
parser.add_argument('--tgt_trun', type=int, default=0,
                    help="Truncate target sequence length")
parser.add_argument('--src_dict', help='')
parser.add_argument('--tgt_dict', help='')
parser.add_argument('--src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('--tgt_suf', default='tgt',
                    help="the suffix of the target filename")
parser.add_argument('--lower', action='store_false',
                    help='lower the case')
parser.add_argument('--freq', type=int, default=0,
                    help="remove words less frequent")

parser.add_argument('--report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, src_trun, tgt_trun, vocab, size, freq=0):

    print("%s: source truncate length = %d, target truncate length = %d," %
          (filename, src_trun, tgt_trun))
    src_max_length = 0
    tgt_max_length = 0
    with open(filename, encoding='utf8') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) != 2:
                print(items)
                continue
            src = ' '.join(eval(items[0])).split(' ')
            tgt = items[1].split(' ')
            
            src_max_length = max(src_max_length, len(src))
            tgt_max_length = max(tgt_max_length, len(tgt))
            if src_trun > 0:
                src = src[:src_trun]
            if tgt_trun > 0:
                tgt = tgt[:tgt_trun]
            for word in src:
                vocab.add(word)
            for word in tgt:
                vocab.add(word)

    print('Max length of %s = %d, %d' % (filename, src_max_length, tgt_max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size, freq)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(filename, dicts, save_srcFile, save_tgtFile, trun=True):
    count, empty_ignored = 0, 0

    print('Processing %s ...' % filename)
    F = open(filename, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')

    for line in F:
        if opt.lower:
            line = line.lower()

        items = line.strip().split('\t')
        if len(items)!=2:
            print(items)
            continue
        sline, tline = items[0], items[1]

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        srcWords = eval(sline) 
        tgtWords = tline.split() 

        if opt.src_trun > 0 and trun:
            srcWords = srcWords[:opt.src_trun]
        srcIds = []
        for srcWord in srcWords:
            srcId = dicts.convertToIdx(srcWord.split(), utils.UNK_WORD)
            srcIds.append(" ".join(list(map(str, srcId))))

        if opt.tgt_trun > 0 and trun:
            tgtWords = tgtWords[:opt.tgt_trun]
        tgtIds = dicts.convertToIdx(
            tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

        srcIdF.write(str(srcIds)+'\n')
        tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')
        srcStrF.write(str(srcWords)+'\n')
        tgtStrF.write(" ".join(tgtWords)+'\n')

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    F.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d ignored due to length == 0)' %
          (count, empty_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id', 
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': count}


def main():

    train_data = opt.load_data + 'train.txt' 
    valid_data = opt.load_data + 'valid.txt'
    test_data = opt.load_data + 'test.txt'

    save_train_src, save_train_tgt = opt.save_data + \
        'train.src', opt.save_data + 'train.tgt'
    save_valid_src, save_valid_tgt = opt.save_data + \
        'valid.src', opt.save_data + 'valid.tgt'
    save_test_src, save_test_tgt = opt.save_data + \
        'test.src', opt.save_data + 'test.tgt'

    dict_path = opt.save_data + 'dict'

    print('Building source and target vocabulary...')
    dicts = utils.Dict(
        [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
    dicts = makeVocabulary(
        train_data, opt.src_trun, opt.tgt_trun, dicts, opt.vocab_size, freq=opt.freq)


    print('Preparing training ...')
    train = makeData(train_data, dicts, save_train_src, save_train_tgt)

    print('Preparing validation ...')
    valid = makeData(valid_data, dicts, save_valid_src, save_valid_tgt, trun=False)

    print('Preparing test ...')
    test = makeData(test_data, dicts, save_test_src, save_test_tgt, trun=False)

    print('Saving vocabulary to \'' + dict_path + '\'...')
    dicts.writeFile(dict_path)

    data = {'train': train, 'valid': valid,
            'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))


if __name__ == "__main__":
    main()
