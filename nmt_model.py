import cupy
import chainer
import chainer.function as chainFunc
import chainer.links as chainLinks
import utility

class BahdanauNMT(chainer.Chain):
    def __init__(self):
        super(BahdanauNMT, self).__init__()
        with self.init_scope():
            self.encoder = biGRU_encoder()
            self.decoder = GRUDecoder()

    def __call__(self, batch_src, batch_tgt):
        self.reset_states()
        hidden_encoder = self.encoder(batch_src)
        last_h_enc = self.encoder.getHidden()
        self.decoder.setHidden(last_h_enc)
        loss, hyp = self.decoder(hidden_encoder, batch_tgt)
        return loss, hyp

    def generate(self, batch_src):
        self.reset_states()
        hidden_encoder = self.encoder(batch_src)
        last_h_enc = self.encoder.getHidden()
        self.decoder.setHidden(last_h_enc)
        hyp = self.decoder(hidden_encoder)
        return hyp

    def reset_state(self):
        self.encoder.reset_states()
        self.decoder.reset_states()

class biGRU_encoder(chainer.Chain):
    def __init__(self, args):
        super(biGRU_encoder, self).__init__()
        with self.init_scope():
            self.word2embed = chainLinks.EmbedID()
            self.embed2hiddenf = chainLinks.GRU()
            self.embed2hiddenb = chainLinks.GRU()
        word2vec周りの処理
        variable init
        #self.dropoutr = args.dropoutrate

    def __call__(self, batch):
        first_states = list()
        #forward_states = list()
        backward_states = list()
        concate_states = list()
        #filstlayer   
        for word in batch:
            first_states.append(chainFunc.dropout(self.word2embed(word), self.dropoutr))
        #backward
        for first_hidden in first_states[::-1]:
            backward_states.append(chainFunc.dropout(self.embed2hiddenb(first_hidden), self.dropoutr))
        #cocate(forward,backward)
        for first_hidden, hidden_b in zip(first_states, backward_states[::-1]):
            forward_hidden = chainFunc.dropout(self.embed2hiddenf(first_hidden), self.dropoutr)
            concate_states.append(chainFunc.concat((forward_hidden, hidden_b), axis=1))
        return concate_states
    
    def reset_states(self):
        self.embed2hiddenf.reset_state()
        self.embed2hiddenb.reset_state()
    
    def getHidden(self):
        #return backward hidden for decoder hidden initialization
        return self.embed2hiddenb.h

class GRUDecoder(chainer.Chain):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        with self.init_scope():
            word2embedding = chainLinks.EmbedID(ignore_label=-1)
            self.gru = BahadanauGRU()
            self.U_o = chainLinks.Linear()
            self.V_o = chainLinks.Linear()
            self.C_o = chainLinks.Linear()
            self.W_o = chainLinks.Linear()
            self.attention = addictiveAttention()

    def __call__(self,enc_states, batch_tgt):
        loss = chainer.Variable(self.xp.zeros((), dtype = self.xp.float32))
        predicts = list()
        for previous_wordID, word in enumerate(batch_tgt[1:]):
            previous_hidden = self.gru.h
            embedding = self.word2embedding(batch_tgt[previous_wordID])
            context = self.attention(previous_hidden, enc_states)
            hidden = self.gru(embedding, context)
            t = self.U_o(previous_hidden) + self.V_o(embedding) + self.C_o(context)
            t = self.chainFunc.maxout(t, 2)
            score = self.W_o(t)
            predict = chainFunc.argmax(score, axis=1)
            # ここがずれてる
            loss += chainFunc.softmax_cross_entropy(score, word, ignore_label=-1)

            predicts.append(predict.data)
        return loss, predicts

    def generate():
        bos = self.xp.array([bosID]*batch_size, dtype=self.xp.int32)
        predicts = [bos]
        while len(predicts) < :
            embedding = self.word2embedding(predicts[-1])
            previous_hidden = self.gru.h
            context = self.attention(previous_hidden, enc_states)
            hidden = self.gru(embedding, context)
            t = self.U_o(previous_hidden) + self.V_o(embedding) + self.C_o(context)
            t = self.chainFunc.maxout(t, 2)
            score = self.W_o(t)
            predict = chainFunc.argmax(score, axis=1)

            predicts.append(predict)
        return predicts
    
    def setHidden(self, h):
        self.gru.set_state(h)

    def reset_states(self):
        self.gru.reset_state()

        
class addictiveAttention(chainer.Chain):
    def __init__(self):
        super(addictiveAttention, self).__init__()
        with self.init_scope():
            self.W_a = chainLinks()
            self.U_a = chainLinks()
            self.V_a = chainLinks()
    
    def __call__(self, previous_hidden, enc_states):
        weighted_hidden = self.W_a(previous_hidden)
        scores = [self.V_a(chainFunc.tanh(weighted_hidden + self.U_a(hidden))) for hidden in enc_states]
        align = chainFunc.softmax(scores)
        context = chainFunc.sum(align*enc_states)
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
