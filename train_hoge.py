import argparse
import utilize as util
import neural_model
import chainer
import cupy
import random

def train(args):
    start_epoch = 0

    trace('start training...')

    trace('random seed: {}'.format(args.seed_num))
    np.random.seed(args.seed_num)
    xp.random.seed(args.seed_num)
    random.seed(args.seed_num)

    chainer.global_config.train = True
    chainer.global_config.use_cudnn = True
    chainer.global_config.type_check = True
    trace('chainer config: {}'.format(chainer.global_config.__dict__))

    trace('load vocab...')
    source_vocab = util.Vocab.load()
    target_vocab = util.Vocab.load()
    if args.gensim == 'make':
        trace('making word2vec...')
        source_word2vec = util.Embedding.make()
        target_word2vec = util.Embedding.make()
        source_word2vec.save()
        target_word2vec.save()
    elif args.gensim == 'load':
        trace('loading word2vec...')
        source_word2vec = util.Embedding.load()
        target_word2vec = util.Embedding.load()
    elif args.gensim == 'not':
        trace('do not use word2vec')
        source_word2vec = None
        target_word2vec = None
    
    trace('making model...')
    #initialize model
    nmt_model = neural_model.hogehoge 
    if args.gpunum >= 0:
        import cupy as xp
        cuda.check_cuda_available() 
        cuda.get_device(args.gpunum).use()
        nmt_model.to_gpu()
        trace('use GPU id: {}'.format(args.gpunum))
    else:
        import numpy as xp
        args.gpunum = -1
        trace('without GPU')

    optim = args.optim
    optim.setup(nmt_model)
    optim.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    for epoch in range(start_epoch, args.epoch):
        trace('Epoch {}/{}'.format(epoch+1. args.epoch))
        accum_loss = 0.0
        num_sent = 0
        for batch_src, batch_tgt in util.make_batch(hoge):
            nmt_model.zerograds()
            loss, batch_hyp = nmt_model(batch_src, batch_tgt)
            accum_loss += loss.data
            loss.backward()
            optim.update()

            for src, tgt, hyp in convert batch to sents:
                trace('Epoch {}/{}, {} sent'.format(epoch+1, args.epoch, num_sent+1))
                trace('src: {}'.format(src))
                trace('tgt: {}'.format(tgt))
                trace('hyp: {}'.format(hyp))
                num_sent += 1
        trace('accum_loss: {}'.format(accum_loss))
        trace('Save model ...')
        model_name = '{}.{03d}'.format(args.name, epoch+1)
        chainer.serializers.save_npz('{}.weights'.format(model_name), nmt)
        chainer.serializers.save_npz('{}.optimizer'.format(model_name), optim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, look at readme.', description='arg description', epilog='end')
    parser.add_argument('sourcelang')
    parser.add_argument('targetlang')
    parser.add_argument('-datadir', help='data directory to use corpus and vocab', default='')
    parser.add_argument('-savedir', help='save directory for weight', default='')
    parser.add_argument('-model', help='model for neural MT', default='bahdanau')
    parser.add_argument('-edim', help='embedding size for model'. type=int, default=512)
    parser.add_argument('-nhid', help='hidden size for model'. type=int, default=512)
    parser.add_argument('-gensim_mode', help='use gensim for embedding, make, load, or not?'. default='not', choices=['make', 'load', 'not'])
    parser.add_argument('-gensimfileS', help='gensim file for source'. default='')
    parser.add_argument('-gensimfileS', help='gensim file for target'. default='')
    parser.add_argument('-nlayer', help='hidden layer for model, attention: 1layer using gensim is 2layer without gensim'. type=int, default=2)
    parser.add_argument('-optim', help='select optimizer', default='AdaGrad')
    parser.add_argument('-lr', help='learning rate for optimizer', type=float, default=0.01)
    parser.add_argument('-gpunum', help='GPU number (negative value is using CPU)', type=int, default=-1)
    parser.add_argument('-epoch', help='max epoch during training', type=int, default=50)
    parser.add_argument('-useDropout', help='max epoch during training', action='store_true')
    parser.add_argument('-dlr', help='dropout rate', type=float, default=0.2)
    parser.add_argument('-batch', help='batch size', type=int, default=100)
    parser.add_argument('-pooling', help='pooling size', type=int, default=100)
    parser.add_argument('-genlimit', help='generation limit', type=int, default=60)
    parser.add_argument('-useBeam', help='use beamsearch or not?', action='store_true')
    parser.add_argument('-beamsize', help='beam size', type=int, default=2)
    parser.add_argument('-grad_clip', help='gradient cliping', type=float, default=5.0)
    parser.add_argument('-useSeed', help='use random seed or not?', action='store_true')
    parser.add_argument('-seed_num', help='random seed number', type=int, default=2434)
    parser.add_argument('-name', help='model name, default is "sample"', default='sample') 
    args = parser.parse_args()
    
    train(args)
    trace('finish training!')
