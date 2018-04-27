import argparse
import utilities as util
import nmt_model
import chainer
import cupy
import random
import numpy as np

def train(args):
    start_epoch = 0
    corpus_file = args.datadir+'/train.'

    util.trace('start training...')

    chainer.global_config.train = True
    chainer.global_config.use_cudnn = 'always'
    chainer.global_config.type_check = True
    util.trace('chainer config: {}'.format(chainer.global_config.__dict__))

    util.trace('load vocab...')
    vocab_file = args.datadir+'/vocabulary.'
    source_vocab = util.Vocabulary.load(vocab_file+args.sourcelang)
    target_vocab = util.Vocabulary.load(vocab_file+args.targetlang)
    """
    util.trace('make vocab...')
    source_vocab = util.Vocabulary.make(corpus_file+args.sourcelang, 3000)
    target_vocab = util.Vocabulary.make(corpus_file+args.targetlang, 3000)
    """

    if args.gensim_mode == 'make':
        util.trace('making word2vec...')
        src_word2vec = util.make_word2vec(corpus_file+args.sourcelang, args.edim)
        tgt_word2vec = util.make_word2vec(corpus_file+args.targetlang, args.edim)
        util.save(src_word2vec, args.datadir+'/src_word2vec.'+args.sourcelang)
        util.save(tgt_word2vec, args.datadir+'/tgt_word2vec.'+args.targetlang)
    elif args.gensim_mode == 'load':
        util.trace('loading word2vec...')
        src_word2vec = util.load_word2vec(args.datadir+'/src_word2vec.'+args.sourcelang)
        tgt_word2vec = util.load_word2vec(args.datadir+'/tgt_word2vec.'+args.targetlang)
    elif args.gensim_mode == 'not':
        util.trace('do not use word2vec')
        src_word2vec = None
        tgt_word2vec = None
    
    util.trace('making model...')
    #initialize model
    NMTmodel = nmt_model.BahdanauNMT(source_vocab, target_vocab, args, src_word2vec, tgt_word2vec)

    if args.gpunum >= 0:
        import cupy as xp
        chainer.cuda.check_cuda_available() 
        chainer.cuda.get_device(args.gpunum).use()
        NMTmodel.to_gpu()
        util.trace('use GPU id: {}'.format(args.gpunum))
    else:
        import numpy as xp
        args.gpunum = -1
        util.trace('without GPU')
    
    util.trace('random seed: {}'.format(args.seed_num))
    np.random.seed(args.seed_num)
    xp.random.seed(args.seed_num)
    random.seed(args.seed_num)

    optim = args.optim
    #this is change
    optim = chainer.optimizers.AdaGrad(lr=args.lr)
    optim.setup(NMTmodel)
    optim.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    for epoch in range(start_epoch, args.epoch):
        util.trace('Epoch {}/{}'.format(epoch+1, args.epoch))
        accum_loss = 0.0
        num_sent = 0
        for batch_src, batch_tgt in util.miniBatch(corpus_file+args.sourcelang, corpus_file+args.targetlang,\
                                    source_vocab, target_vocab, args.batch, args.pooling):
            NMTmodel.zerograds()
            loss, batch_hyp = NMTmodel(batch_src, batch_tgt)
            accum_loss += loss.data
            loss.backward()
            optim.update()

            for src, tgt, hyp in zip(util.convert_b2w(batch_src, source_vocab), util.convert_b2w(batch_tgt, target_vocab), \
                util.convert_b2w(batch_hyp, target_vocab)):
                util.trace('Epoch {}/{}, {} sent'.format(epoch+1, args.epoch, num_sent+1))
                util.trace('src: {}'.format(src))
                util.trace('tgt: {}'.format(tgt))
                util.trace('hyp: {}'.format(hyp))
                num_sent += 1
        util.trace('accum_loss: {}'.format(accum_loss))
        util.trace('Save model ...')
        model_name = '{}.{:03d}'.format(args.name, epoch+1)
        chainer.serializers.save_npz(args.savedir+'/{}.weights'.format(model_name), NMTmodel)
        chainer.serializers.save_npz(args.savedir+'/{}.optimizer'.format(model_name), optim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, look at readme.', description='arg description', epilog='end')
    parser.add_argument('sourcelang')
    parser.add_argument('targetlang')
    parser.add_argument('-datadir', help='data directory to use corpus and vocab', default='')
    parser.add_argument('-savedir', help='save directory for weight', default='')
    #parser.add_argument('-model', help='model for neural MT', default='bahdanau')
    parser.add_argument('-edim', help='embedding size for model', type=int, default=512)
    parser.add_argument('-nhid', help='hidden size for model', type=int, default=512)
    parser.add_argument('-gensim_mode', help='use gensim for embedding, make, load, or not?', default='not', choices=['make', 'load', 'not'])
    #parser.add_argument('-gensimfileS', help='gensim file for source'. default='')
    #parser.add_argument('-gensimfileT', help='gensim file for target'. default='')
    #parser.add_argument('-nlayer', help='hidden layer for model, attention: 1layer using gensim is 2layer without gensim'. type=int, default=2)
    parser.add_argument('-optim', help='select optimizer', default='AdaGrad')
    parser.add_argument('-lr', help='learning rate for optimizer', type=float, default=0.01)
    parser.add_argument('-gpunum', help='GPU number (negative value is using CPU)', type=int, default=-1)
    parser.add_argument('-epoch', help='max epoch during training', type=int, default=50)
    parser.add_argument('-useDropout', help='max epoch during training', action='store_true')
    parser.add_argument('-dlr', help='dropout rate', type=float, default=0.2)
    parser.add_argument('-batch', help='batch size', type=int, default=100)
    parser.add_argument('-pooling', help='pooling size', type=int, default=100)
    parser.add_argument('-genlimit', help='generation limit', type=int, default=60)
    #parser.add_argument('-useBeam', help='use beamsearch or not?', action='store_true')
    #parser.add_argument('-beamsize', help='beam size', type=int, default=2)
    parser.add_argument('-grad_clip', help='gradient cliping', type=float, default=5.0)
    parser.add_argument('-useSeed', help='use random seed or not?', action='store_true')
    parser.add_argument('-seed_num', help='random seed number', type=int, default=2434)
    parser.add_argument('-name', help='model name, default is "sample"', default='sample') 
    args = parser.parse_args()
    
    train(args)
    util.trace('finish training!')
