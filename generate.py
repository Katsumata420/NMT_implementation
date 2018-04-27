import argparse
import utilities as util
import chainer.functions as chainFunc
import chainer.links as chainLink
import chainer
import nmt_model

def test(args):
    src_word2vec = None
    tgt_word2vec = None
    util.trace('start testing ...')
    corpus_file = args.vocabdir+'/test.' + args.sourcelang
    output_file = args.savedir+'/{}.generate.{}-{}'.format(args.name, args.sourcelang, args.targetlang)
    args.name = args.datadir+'/{}.{:03d}'.format(args.name, args.epochNum)

    chainer.global_config.train = False
    #chainer.global_config.debug = True
    util.trace('chainer config: {}'.format(chainer.global_config.__dict__))

    util.trace('load vocab...')
    vocab_file = args.vocabdir+'/vocabulary.'
    source_vocab = util.Vocabulary.load(vocab_file+args.sourcelang)
    target_vocab = util.Vocabulary.load(vocab_file+args.targetlang)

    util.trace('Loading Model ...') 

    NMTmodel = nmt_model.BahdanauNMT(source_vocab, target_vocab, args, src_word2vec, tgt_word2vec)

    if args.useGPU >= 0:
        import cupy as xp
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(args.useGPU).use()
        NMTmodel.to_gpu()
        util.trace('use GPU id: {}'.format(args.useGPU))
    else:
        import numpy as xp
        args.useGPU = -1
        util.trace('without GPU')
    
    chainer.serializers.load_npz('{}.weights'.format(args.name), NMTmodel)

    util.trace('Generating translation ...')
    finished = 0

    with open(output_file, 'w') as o_f:
        for src_sent in util.monoBatch(corpus_file, source_vocab, args):
            util.trace('Sample {} ...'.format(finished + 1))
            prds = NMTmodel.generate(src_sent)
            for predict in util.convert_b2w(prds, target_vocab):
                o_f.write('{}\n'.format(predict))
                finished += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, see the readme', description='arg description', epilog='end')
    parser.add_argument('sourcelang')
    parser.add_argument('targetlang')
    parser.add_argument('-datadir', help='saved the weight file in this folder', default='')
    parser.add_argument('-vocabdir', help='saved the vocab file in this folder', default='')
    parser.add_argument('-savedir', help='save the output sentence in this folder', default='')
    parser.add_argument('-useGPU', type=int, default=-1)
    parser.add_argument('-epochNum', type=int, help='point at the model you want to use')
    parser.add_argument('-genlimit', type=int, help='generation limit')
    parser.add_argument('-name', default='sample', help='model name')
    parser.add_argument('-edim', default=512, type=int, help='embedding size for model')
    parser.add_argument('-nhid', default=1024, type=int, help='hidden size for model')
    parser.add_argument('-useDropout', action='store_true')
    args = parser.parse_args()
    test(args)
    util.trace('finish generation')

