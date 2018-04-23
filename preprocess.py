import utilities as util
import argparse

def 

def preprocess(args):
    #tokenize
    #clean?corpus
    #multi_bleu
    #ここまでで作成したtraining corpusをどっか保存しておく
    #した二つはその作成したcorpusに対して実行する
    #vocab
    make_vocab()
    #word2vec
    if args.word2vec:
        util.make_word2vec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, see the readme.', description='arg description', epilog='end')
    parser.add_argument('sourcelang')
    parser.add_argument('targetlang')
    parser.add_argument('-srcfile', help='source corpus file path')
    parser.add_argument('-tgtfile', help='source corpus file path')
    parser.add_argument('-savedir', help='save directory for training corpus')
    parser.add_argument('-word2vec', help='use word2vec or not', action='store_true')
    args = parser.parse_args()
    
    preprocess(args)
    util.trace('finish preprocess')
