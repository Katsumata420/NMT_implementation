import chainer.functions as chainFunc
import chainer.links as chainLink
import chainer
import itertools
import datetime
import collections
import sys
import random
from gensim.models import word2vec
import cupy as xp

class Vocabulary:
    def make(file_path, vocab_size):
        self = Vocabulary()
        self.word2id = collections.defaultdict(int)
        self.id2word = dict()
        with open(file_path) as i_f:
            word_frequency = collections.defaultdict(int)
            for line in i_f:
                for word in line.strip().split():
                    word_frequency[word] += 1
        self.word2id['<unk>'] = 0
        self.word2id['<bos>'] = 1
        self.word2id['<eos>'] = 2
        self.word2id[''] = -1
        self.id2word[0] = '<unk>'
        self.id2word[1] = '<bos>'
        self.id2word[2] = '<eos>'
        self.id2word[-1] = ''
        for i, (word, freq) in zip(range(vocab_size-3), \
            sorted(sorted(word_frequency.items()), key=lambda x:x[1], reverse=True)):
            self.word2id[word] = i + 3
            self.id2word[i+3] = word
        self.size = len(self.word2id) - 1

        return self

    def load(file_path):
        self = Vocabulary()
        self.word2id = collections.defaultdict(int)
        self.id2word = dict()
        self.word2id['<unk>'] = 0 
        self.word2id['<bos>'] = 1
        self.word2id['<eos>'] = 2
        self.word2id[''] = -1
        self.id2word[0] = '<unk>'
        self.id2word[1] = '<bos>'
        self.id2word[2] = '<eos>'
        self.id2word[-1] = ''
        with open(file_path) as vocab_file:
            for i, word in enumerate(vocab_file):
                self.word2id[word.strip()] = i+3
                self.id2word[i+3] = word.strip()
        self.size = i+4
        trace('vocab size: {}'.format(self.size))
        return self

    def save(self, file_path):
        with open(file_path, 'w') as output_file:
            for i in range(self.size):
                output_file.write(self.id2word[i]+'\n')

def miniBatch(src_corpus, tgt_corpus, src_vocab, tgt_vocab, batch_size, pooling):
    batches = list()
    mini_batch = list()
    for Npairs in gen_Npairs(src_corpus, tgt_corpus, src_vocab, tgt_vocab, batch_size*pooling):
        for sent_pair in sorted(Npairs, key=lambda x:len(x[0]), reverse=True):
            mini_batch.append(sent_pair)
            if len(mini_batch) == batch_size:
                batches.append(mini_batch)
                mini_batch = list()
    if mini_batch:
        batches.append(mini_batch)
    random.shuffle(batches)
    for mini_batch in batches:
        batch_src = [mini_batch[i][0] for i in range(len(mini_batch))]
        batch_tgt = [mini_batch[i][1] for i in range(len(mini_batch))]
        yield ([chainer.Variable(xp.array(list(x), dtype=xp.int32)) for x in itertools.zip_longest(*batch_src, fillvalue=-1)], 
        [chainer.Variable(xp.array(list(y), dtype=xp.int32)) for y in itertools.zip_longest(*batch_tgt, fillvalue=-1)])

def make_word2vec(corpus, embed_size):
    word2vec_model = word2vec.Word2Vec(word2vec.LineSentence(corpus), size=embed_size, min_count=1, workers=5)
    return word2vec_model

def save_word2vec(model, save_path):
    model.save(save_path)

def load_word2vec(path):
    return word2vec.Word2Vec.load(path)

def convert_b2w(batch, vocab):
    for sentence in list(chainer.cuda.to_cpu(chainFunc.transpose(chainFunc.vstack(batch)).data)):
        word_list = list()
        for i in list(sentence):
            word_list.append(vocab.id2word[i])
        if '<eos>' in word_list:
            yield ' '.join(word_list[:word_list.index('<eos>')])
        else:
            yield ' '.join(word_list)

def gen_Npairs(src_corpus, tgt_corpus, src_vocab, tgt_vocab, N):
    flag = True
    with open(src_corpus) as src, open(tgt_corpus) as tgt:
        Npairs = list()
        for line_src, line_tgt in zip(src, tgt):
            sent_src = list()
            sent_tgt = [tgt_vocab.word2id['<bos>']]
            for word in line_src.strip().split():
                sent_src.append(src_vocab.word2id[word])
            sent_src.append(src_vocab.word2id['<eos>'])
            for word in line_tgt.strip().split():
                sent_tgt.append(tgt_vocab.word2id[word])
            sent_tgt.append(tgt_vocab.word2id['<eos>'])
            Npairs.append([sent_src, sent_tgt])
            if len(Npairs) == N:
                yield Npairs
                Npairs = list()
        if Npairs:
            yield Npairs

def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()

