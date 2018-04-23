import utilities as util
import argparse
import collections
from gensim.models import word2vec

def preprocess(args):
    #tokenize
    #clean?corpus
    #ここまでで作成したtraining corpusをどっか保存しておく
    #した二つはその作成したcorpusに対して実行する
    #vocab
    #src_corpus = args.savedir+'/train.'+args.sourcelang
    #tgt_corpus = args.savedir+'/train.'+args.targetlang
    src_vocab = util.Vocabulary.make(args.savedir+'/train.'+args.sourcelang, args.srcvocab_size)
    #src_vocab = util.Vocabulary.make(src_corpus, args.srcvocab_size)
    tgt_vocab = util.Vocabulary.make(args.savedir+'/train.'+args.targetlang, args.tgtvocab_size)
    #tgt_vocab = util.Vocabulary.make(tgt_corpus, args.tgtvocab_size)
    src_vocab_file = args.savedir+'/vocabulary.'+args.sourcelang
    tgt_vocab_file = args.savedir+'/vocabulary.'+args.targetlang
    src_vocab.save(src_vocab_file)
    tgt_vocab.save(tgt_vocab_file)
    #word2vec
    if args.word2vec:
        #default; worker is 5.
        src_w2v=util.make_word2vec(args.savedir+'/train.'+args.sourcelang, args.edim)
        tgt_w2v=util.make_word2vec(args.savedir+'/train.'+args.targetlang, args.edim)
        util.save_word2vec(src_w2v, args.savedir+'/src_word2vec.'+args.sourcelang)
        util.save_word2vec(tgt_w2v, args.savedir+'/tgt_word2vec.'+args.targetlang)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, see the readme.', description='arg description', epilog='end')
    parser.add_argument('sourcelang')
    parser.add_argument('targetlang')
    parser.add_argument('-srcfile', help='source corpus file path')
    parser.add_argument('-tgtfile', help='source corpus file path')
    parser.add_argument('-savedir', help='save directory for training corpus')
    parser.add_argument('-word2vec', help='use word2vec or not', action='store_true')
    parser.add_argument('-edim', help='embedding size', type=int)
    parser.add_argument('-srcvocab_size', help='vocabulary size for src', type=int)
    parser.add_argument('-tgtvocab_size', help='vocabulary size for tgt', type=int)
    #parser.add_argument('-vocab_thred', help='under threthold, not use word for vocabulary', type=int)
    args = parser.parse_args()
    
    preprocess(args)
    util.trace('finish preprocess')
