import cupy
import chainer
import chainer.functions as chainFunc
import chainer.links as chainLinks
import utilities as util
import cupy as xp

class BahdanauNMT(chainer.Chain):
    def __init__(self, src_vocab, tgt_vocab, args, src_w2v, tgt_w2v):
        super(BahdanauNMT, self).__init__()
        with self.init_scope():
            self.encoder = biGRU_encoder(src_vocab, args, src_w2v)
            self.decoder = GRUDecoder(tgt_vocab, args, tgt_w2v)

    def __call__(self, batch_src, batch_tgt):
        self.reset_state()
        hidden_encoder = self.encoder(batch_src)
        last_h_enc = self.encoder.getHidden()
        self.decoder.setHidden(last_h_enc)
        loss, hyp = self.decoder(hidden_encoder, batch_tgt)
        return loss, hyp

    def generate(self, batch_src):
        self.reset_state()
        hidden_encoder = self.encoder(batch_src)
        last_h_enc = self.encoder.getHidden()
        self.decoder.setHidden(last_h_enc)
        hyp = self.decoder.generateHyp(hidden_encoder)
        return hyp

    def reset_state(self):
        self.encoder.reset_states()
        self.decoder.reset_states()

class biGRU_encoder(chainer.Chain):
    def __init__(self, src_vocab, args, src_w2v):
        super(biGRU_encoder, self).__init__()
        with self.init_scope():
            self.word2embed = chainLinks.EmbedID(src_vocab.size, args.edim, ignore_label=-1)
            self.embed2hiddenf = chainLinks.GRU(args.edim, args.nhid)
            self.embed2hiddenb = chainLinks.GRU(args.edim, args.nhid)

        #embedding weight is intialized by w2v.
        if src_w2v is not None:
            for i in range(src_vocab.size):
                word = src_vocab.id2word[i]
                if word in src_w2v:
                    self.word2embed.W.data[i] = src_w2v[word]
        self.vocab_size = src_vocab.size
        self.embedding_size = args.edim
        self.hidden_size = args.nhid
        if args.useDropout:
            self.use_dropout = args.useDropout
            self.dropoutr = args.dlr
        else:
            self.use_dropout = None
            self.dropoutr = 0
            util.trace('{}'.format(chainer.global_config.__dict__))

    def __call__(self, batch):
        first_states = list()
        backward_states = list()
        concate_states = list()
        #filstlayer   
        for word in batch:
            first_states.append(chainFunc.dropout(self.word2embed(word), ratio=self.dropoutr))
        #util.trace('first_states length:{}'.format(len(first_states)))
        #util.trace('first state embedding:{}'.format(first_states[0].shape))
        #backward
        for first_hidden in first_states[::-1]:
            backward_states.append(chainFunc.dropout(self.embed2hiddenb(first_hidden), self.dropoutr))
        #util.trace('second state embedding:{}'.format(backward_states[0].shape))
        #cocate(forward,backward)
        for first_hidden, hidden_b in zip(first_states, backward_states[::-1]):
            forward_hidden = chainFunc.dropout(self.embed2hiddenf(first_hidden), self.dropoutr)
            concate_states.append(chainFunc.concat((forward_hidden, hidden_b), axis=1))
        #util.trace('top state embedding:{}'.format(concate_states[0].shape))
        return concate_states
    
    def reset_states(self):
        self.embed2hiddenf.reset_state()
        self.embed2hiddenb.reset_state()
    
    def getHidden(self):
        #return backward hidden for decoder hidden initialization
        return self.embed2hiddenb.h

class GRUDecoder(chainer.Chain):
    def __init__(self, tgt_vocab, args, tgt_w2v):
        super(GRUDecoder, self).__init__()
        with self.init_scope():
            self.word2embedding = chainLinks.EmbedID(tgt_vocab.size, args.edim, ignore_label=-1)
            self.gru = BahadanauGRU(args.edim, args.nhid)
            self.U_o = chainLinks.Linear(args.nhid, args.nhid)
            self.V_o = chainLinks.Linear(args.edim, args.nhid)
            self.C_o = chainLinks.Linear(2*args.nhid, args.nhid)
            self.W_o = chainLinks.Linear(args.nhid//2, tgt_vocab.size)
            self.attention = additiveAttention(args.nhid)
        
        if tgt_w2v is not None:
            for i in range(tgt_vocab.size):
                word = tgt_vocab.id2word[i]
                if word in tgt_w2v:
                    self.word2embedding.W.data[i] = tgt_w2v[word]
        self.vocab_size = tgt_vocab.size
        self.embedding_size = args.edim
        self.hidden_size = args.nhid
        self.gen_limit = args.genlimit
        if args.useDropout:
            self.use_dropout = args.useDropout
            self.dropoutr = args.dlr
        else:
            self.use_dropout = None
            self.dropoutr = 0
            util.trace('{}'.format(chainer.global_config.__dict__))

    def __call__(self, enc_states, batch_tgt):
        loss = chainer.Variable(self.xp.zeros((), dtype = self.xp.float32))
        predicts = list()
        for previous_wordID, word in enumerate(batch_tgt[1:]):
            previous_hidden = self.gru.h
            embedding = self.word2embedding(batch_tgt[previous_wordID])
            #util.trace('embedding size decside: {}'.format(embedding.shape))
            context = self.attention(previous_hidden, enc_states)
            hidden = chainFunc.dropout(self.gru(embedding, context),self.dropoutr)
            t = self.U_o(previous_hidden) + self.V_o(embedding) + self.C_o(context)
            t = chainFunc.maxout(t, 2)
            score = self.W_o(t)
            predict = chainFunc.argmax(score, axis=1)
            loss += chainFunc.softmax_cross_entropy(score, word, ignore_label=-1)

            predicts.append(predict.data)
        return loss, predicts

    def generateHyp(self, enc_states):
        bosID = 1
        bos = self.xp.array([bosID], dtype=self.xp.int32)
        predicts = [bos]
        while len(predicts)-1 < self.gen_limit:
            embedding = self.word2embedding(predicts[-1])
            previous_hidden = self.gru.h
            context = self.attention(previous_hidden, enc_states)
            hidden = self.gru(embedding, context)
            t = self.U_o(previous_hidden) + self.V_o(embedding) + self.C_o(context)
            t = chainFunc.maxout(t, 2)
            score = self.W_o(t)
            predict = chainFunc.argmax(score, axis=1)

            predicts.append(predict)
        del predicts[0]
        return predicts
    
    def setHidden(self, h):
        self.gru.set_state(h)

    def reset_states(self):
        self.gru.reset_state()

        
class additiveAttention(chainer.Chain):
    def __init__(self, hidden_size):
        super(additiveAttention, self).__init__()
        with self.init_scope():
            self.W_a = chainLinks.Linear(hidden_size, hidden_size)
            self.U_a = chainLinks.Linear(2*hidden_size, hidden_size)
            self.V_a = chainLinks.Linear(hidden_size, 1)
    
    def __call__(self, previous_hidden, enc_states):
        weighted_hidden = self.W_a(previous_hidden)
        #util.trace('W_a calc hidden: {}'.format(weighted_hidden.shape))
        scores = [self.V_a(chainFunc.tanh(weighted_hidden + self.U_a(hidden))) for hidden in enc_states]
        #ここでbatch*sourcelength*1の形にする
        scores = chainFunc.stack(scores, axis=1)
        """
        util.trace('scores type; {}'.format(type(scores)))
        util.trace('scores length; {}'.format(len(scores)))
        util.trace('score type; {}'.format(type(scores[0])))
        util.trace('scores shape: {}'.format(scores.shape))
        util.trace('score shape; {}'.format(scores[0].shape))
        """
        align = chainFunc.softmax(scores, axis=1)
        #util.trace('align shape: {}'.format(align.shape))
        stackenc_hidden = chainFunc.stack(enc_states, axis=1)
        #util.trace('stacking encder state shape: {}'.format(stackenc_hidden.shape))
        align_cast = chainFunc.broadcast_to(align, stackenc_hidden.shape)
        #util.trace('align cast shape: {}'.format(align_cast.shape))
        context = chainFunc.sum(align_cast*stackenc_hidden, axis=1)
        return context
        

class BahadanauGRU(chainLinks.StatefulGRU):
    def __init__(self, in_size, out_size):
        super(BahadanauGRU, self).__init__(in_size, out_size)
        with self.init_scope():
            self.C = chainLinks.Linear(2*out_size, out_size)
            self.C_z = chainLinks.Linear(2*out_size, out_size)
            self.C_r = chainLinks.Linear(2*out_size, out_size)

    def __call__(self, word, context):
        #like a statefulGRU code
        z = self.W_z(word)
        h_bar = self.W(word)
        if self.h is not None:
            r = chainFunc.sigmoid(self.W_r(word)+self.U_r(self.h)+self.C_r(context))
            z += self.U_z(self.h) + self.C_z(context)
            h_bar += self.U(r*self.h) + self.C(context)
        z = chainFunc.sigmoid(z)
        h_bar = chainFunc.tanh(h_bar)

        if self.h is not None:
            h_new = chainFunc.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar
        self.h = h_new
        return self.h
