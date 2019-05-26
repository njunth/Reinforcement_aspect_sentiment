import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class origin_TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(origin_TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005
        self.l2_weight = 0.001

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        # self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn_a = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False)

        # self.attn = nn.Linear(self.hidden_size, self.max_length)
        # self.attn_softmax = nn.Softmax()
        # self.W1 = nn.Linear(hidden_size, hidden_size)
        # self.W2 = nn.Linear(self.input_size, self.input_size)
        # self.combine = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)

        self.decoder_a = nn.Linear(self.hidden_size+self.input_size, output_size)  # TODO
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.LogSoftmax()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        # self.optimizer = torch.optim.Adadelta(self.parameters())

    def getembedding(self, l_indicat):
        aspect_e_l = []
        for l in l_indicat:
            aspect_a = np.array( [l_indicat[l]], dtype=np.int64 )
            aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
            # print self.AE(Variable(aspect_tensor))
            as_e = self.AE(Variable(aspect_tensor))
            aspect_e_l.append(as_e)
        aspect_embeds = torch.cat( aspect_e_l, 0 )
        return aspect_embeds.data

    def forward(self, inputs, aspect=None):
        # print(aspect)
        # assert aspect.data.size(0) == 1

        # print("fuck")
        # print(input)

        # length = inputs.size()[0]
        aspect_embedding = self.AE( aspect )
        aspect_embedding = aspect_embedding.view(1, -1)

        output, (ht, ct) = self.rnn_a(inputs)
        # ht = ht.view(1, -1)
        # print(output.size())
        # last_i = Variable(torch.LongTensor([output.size()[0] - 1]))
        # last = torch.index_select(output, 0, last_i).view(1, -1)
        # print(last)
        # output = output.view(output.size()[0], -1)
        # s = self.attn(output)
        # s = s.view(1,-1)
        # attn_weights = F.softmax(s)
        # # print(attn_weights)
        # output = torch.mm(attn_weights, output)
        # r = F.tanh(torch.add(self.W1(output), self.W2(last)))

        # print(hidden)
        r = ht.view(1, -1)
        r = self.dropout(r)
        decoded = self.decoder_a(torch.cat((r, aspect_embedding), dim=1))
        # decoded = self.dropout(decoded)
        # output = F.softmax(decoded, dim=1)
        # output = decoded
        # output = F.sigmoid(decoded)
        return decoded, (None, None)

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        self.train()

        # aspect_tensor = torch.LongTensor( aspect ).view( -1 )
        # print aspect
        aspect_a = np.array( [aspect], dtype=np.int64 )
        aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
        # print(aspect_tensor)

        self.optimizer.zero_grad()
        output, _ = self.forward( line_tensor, Variable( aspect_tensor ) )
        # print "o"
        # print(output)
        # print "c"
        # print(category_tensor)
        loss = F.cross_entropy( output, category_tensor )
        loss.backward()

        self.optimizer.step()

        # print "opt finish"

        return output, loss.data[0], (0, 0)

class TextRNN(nn.Module):  # just a lstm with attention
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005
        self.l2_weight = 0.0005
        # self.lr_decay = 0.1

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        self.W_h = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_v = nn.Linear( self.input_size, self.input_size )
        self.w = nn.Linear( self.hidden_size + self.input_size, 1 )
        self.W_p = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_x = nn.Linear( self.hidden_size, self.hidden_size )

        # self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers = 1, bidirectional=False)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        # self.softmax = nn.LogSoftmax()
        # self.softmax= nn.Softmax(dim=1)
        self.attn_softmax = nn.Softmax( dim=0 )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        # self.optimizer_o = torch.optim.Adagrad(self.parameters(), lr=self.lr,  lr_decay=self.lr_decay, weight_decay=self.l2_weight)

    def forward(self, inputs, aspect):
        length = inputs.size()[0]
        output, hidden = self.rnn( inputs )

        hidden = hidden[0].view( 1, -1 )
        output = output.view( output.size()[0], -1 )
        aspect_embedding = self.AE( aspect )
        aspect_embedding = aspect_embedding.view( 1, -1 )
        # print(aspect)
        aspect_embedding = aspect_embedding.expand( length, -1 )
        M = F.tanh( torch.cat( (self.W_h( output ), self.W_v( aspect_embedding )), dim=1 ) )
        weights = self.attn_softmax( self.w( M ) ).t()
        # print(weights)
        r = torch.matmul( weights, output )
        r = F.tanh( torch.add( self.W_p( r ), self.W_x( hidden ) ) )
        r = self.dropout( r )
        decoded = self.decoder( r )
        output = decoded
        return output, (None, None)

    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable( torch.zeros(1, self.hidden_size ) ), Variable( torch.zeros( (1, self.hidden_size) ) ))

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        # self.zero_grad()
        # global output, loss
        self.train()

        # aspect_tensor = torch.LongTensor( aspect ).view( -1 )
        # print aspect
        aspect_a= np.array([aspect], dtype=np.int64)
        aspect_tensor = torch.from_numpy(aspect_a).view(-1)
        # print(aspect_tensor)

        self.optimizer.zero_grad()
        output, _ = self.forward( line_tensor, Variable( aspect_tensor ) )
        # print "o"
        # print(output)
        # print "c"
        # print(category_tensor)
        loss = F.cross_entropy( output, category_tensor )
        loss.backward()

        self.optimizer.step()

        # print "opt finish"

        return output, loss.data[0], (0, 0)

class AE_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(AE_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005
        self.l2_weight = 0.001

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        # self.W_h = nn.Linear( self.hidden_size, self.hidden_size )
        # self.W_v = nn.Linear( self.input_size, self.input_size )
        # self.w = nn.Linear( self.hidden_size + self.input_size, 1 )
        # self.W_p = nn.Linear( self.hidden_size, self.hidden_size )
        # self.W_x = nn.Linear( self.hidden_size, self.hidden_size )

        # self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn = nn.LSTM(2*input_size, hidden_size, num_layers = 1, bidirectional=False)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        # self.softmax = nn.LogSoftmax()
        # self.softmax= nn.Softmax(dim=1)
        self.attn_softmax = nn.Softmax( dim=0 )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  weight_decay=self.l2_weight)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs, aspect):
        length = inputs.size()[0]
        # print inputs.size()

        aspect_embedding = self.AE( aspect )
        aspect_embedding = aspect_embedding.view( 1, -1 )
        # print(aspect)
        aspect_embedding = aspect_embedding.expand( length, 1, -1 )
        # print aspect_embedding.size()

        output, hidden = self.rnn( torch.cat((inputs, aspect_embedding), dim=2) )

        hidden = hidden[0].view( 1, -1 )
        # output = output.view( output.size()[0], -1 )
        #
        # aspect_embedding_a = self.AE( aspect )
        # aspect_embedding_a = aspect_embedding_a.view( 1, -1 )
        # # print(aspect)
        # aspect_embedding_a = aspect_embedding_a.expand( length, -1 )
        #
        # M = F.tanh( torch.cat( (self.W_h( output ), self.W_v( aspect_embedding_a )), dim=1 ) )
        # weights = self.attn_softmax( self.w( M ) ).t()
        # # print(weights)
        # r = torch.matmul( weights, output )
        # r = F.tanh( torch.add( self.W_p( r ), self.W_x( hidden ) ) )
        decoded = self.decoder( hidden )
        output = decoded
        return output, (None, None)

    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable( torch.zeros(1, self.hidden_size ) ), Variable( torch.zeros( (1, self.hidden_size) ) ))

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        # self.zero_grad()
        # global output, loss
        self.train()

        # aspect_tensor = torch.LongTensor( aspect ).view( -1 )
        # print aspect
        aspect_a= np.array([aspect], dtype=np.int64)
        aspect_tensor = torch.from_numpy(aspect_a).view(-1)
        # print(aspect_tensor)

        self.optimizer.zero_grad()
        output, _ = self.forward( line_tensor, Variable( aspect_tensor ) )
        # print "o"
        # print(output)
        # print "c"
        # print(category_tensor)
        loss = F.cross_entropy( output, category_tensor )
        loss.backward()

        self.optimizer.step()

        # print "opt finish"

        return output, loss.data[0], (0, 0)


class ATAE_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(ATAE_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005
        self.l2_weight = 0.001

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        self.W_h = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_v = nn.Linear( self.input_size, self.input_size )
        self.w = nn.Linear( self.hidden_size + self.input_size, 1 )
        self.W_p = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_x = nn.Linear( self.hidden_size, self.hidden_size )

        # self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn = nn.LSTM(2*input_size, hidden_size, num_layers = 1, bidirectional=False)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        # self.softmax = nn.LogSoftmax()
        # self.softmax= nn.Softmax(dim=1)
        self.attn_softmax = nn.Softmax( dim=0 )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  weight_decay=self.l2_weight)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs, aspect):
        length = inputs.size()[0]
        # print inputs.size()

        aspect_embedding = self.AE( aspect )
        aspect_embedding = aspect_embedding.view( 1, -1 )
        # print(aspect)
        aspect_embedding = aspect_embedding.expand( length, 1, -1 )
        # print aspect_embedding.size()

        output, hidden = self.rnn( torch.cat((inputs, aspect_embedding), dim=2) )

        hidden = hidden[0].view( 1, -1 )
        output = output.view( output.size()[0], -1 )

        aspect_embedding_a = self.AE( aspect )
        aspect_embedding_a = aspect_embedding_a.view( 1, -1 )
        # print(aspect)
        aspect_embedding_a = aspect_embedding_a.expand( length, -1 )

        M = F.tanh( torch.cat( (self.W_h( output ), self.W_v( aspect_embedding_a )), dim=1 ) )
        weights = self.attn_softmax( self.w( M ) ).t()
        # print(weights)
        r = torch.matmul( weights, output )
        r = F.tanh( torch.add( self.W_p( r ), self.W_x( hidden ) ) )
        decoded = self.decoder( r )
        output = decoded
        return output, (None, None)

    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable( torch.zeros(1, self.hidden_size ) ), Variable( torch.zeros( (1, self.hidden_size) ) ))

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        # self.zero_grad()
        # global output, loss
        self.train()

        # aspect_tensor = torch.LongTensor( aspect ).view( -1 )
        # print aspect
        aspect_a= np.array([aspect], dtype=np.int64)
        aspect_tensor = torch.from_numpy(aspect_a).view(-1)
        # print(aspect_tensor)

        self.optimizer.zero_grad()
        output, _ = self.forward( line_tensor, Variable( aspect_tensor ) )
        # print "o"
        # print(output)
        # print "c"
        # print(category_tensor)
        loss = F.cross_entropy( output, category_tensor )
        loss.backward()

        self.optimizer.step()

        # print "opt finish"

        return output, loss.data[0], (0, 0)


class origin_ID_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(origin_ID_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.actions = 2
        self.lr = 0.0005
        self.l2_weight = 0.001

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        self.rnn = nn.LSTMCell(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(hidden_size+self.input_size, output_size)
        self.dropout = nn.Dropout(0.3)
        # self.softmax = nn.LogSoftmax()
        self.softmax= nn.Softmax(dim=1)

        self.PNet = nn.Linear(2 * self.hidden_size + 2*self.input_size, self.actions)
        self.saved_actions = []
        self.saved_log_probs = []
        self.gamma = 1
        self.miu = 0.1

        self.optimizerRL = torch.optim.Adam([{'params': self.PNet.parameters()},
                                             {'params': self.AE.parameters()}], lr=self.lr)
        self.optimizerC = torch.optim.Adam([
            {'params': self.AE.parameters()},
            {'params': self.rnn.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.dropout.parameters()},
            {'params': self.softmax.parameters()}
            ], lr=self.lr, weight_decay=self.l2_weight)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    def forward(self, inputs, aspect=None):
        # print(input)
        del self.saved_log_probs[:]
        del self.saved_actions[:]
        length = inputs.size()[0]
        ht, ct = self.init_hidden(inputs.data.is_cuda)

        aspect_embedding = self.AE( aspect )
        # aspect_embedding = aspect_embedding.view(-1)

        retain_len = 0
        for t in range(length):
            state = torch.cat((ct.view(-1), ht.view(-1), inputs[t].view(-1), aspect_embedding.view(-1))).data
            action = self.select_action(state)
            # print(action)
            if action == 1:  # the word is retained.
                ht, ct = self.rnn(inputs[t], (ht, ct))
                retain_len += 1
            # if action == 1 or action == 0:
            #     ht, ct = self.rnn(inputs[t], (ht, ct))
        # print(retain_len)
        r = ht.view(1, -1)  # just reshape

        r = self.dropout(r)
        decoded = self.decoder(torch.cat((r, aspect_embedding.view(1, -1)), dim=1))
        # print(decoded)
        # output = self.softmax(decoded)
        output = decoded
        return output, (length, retain_len)



    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable(torch.zeros(1, self.hidden_size)), Variable(torch.zeros((1, self.hidden_size))))

    def select_action(self, state):
        # print(state.view(1, -1))
        # state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.view(1, -1)
        probs = F.softmax(self.PNet(Variable(state)), dim=1)
        action = probs.multinomial()  # sample an action from the computed distribution.
        self.saved_actions.append(action.data[0, 0])
        log_prob = torch.gather(probs, 1, action).log()
        self.saved_log_probs.append(log_prob)
        return action.data[0, 0]

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        # print(line_tensor)
        aspect_a = np.array( [aspect], dtype=np.int64 )
        aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
        output, (l, retained_) = self.forward(line_tensor, Variable( aspect_tensor ))
        self.train()

        suppression = 0.1
        # suppression = 1
        L = line_tensor.size(0)
        retained = sum(self.saved_actions)
        L_ = L - retained
        assert retained == retained_

        loss = Variable(torch.zeros(1))
        if not freeze:
            # optimize Classification Part
            loss = F.cross_entropy(output, category_tensor)
            self.optimizerC.zero_grad()
            loss.backward()
            self.optimizerC.step()

        # Optimize Reinforcement Part:

        # RL = np.log(torch.index_select(output, 1, category_tensor).data[0, 0]) + self.miu * L_/L  # TODO
        RL = torch.index_select(output, 1, category_tensor).data[0, 0] + self.miu * L_ / L
        policy_loss = []
        rewards = []
        for r in range(line_tensor.size(0)):
            R = self.gamma * RL
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        # print(rewards)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            try:
                policy_loss.append(-log_prob * reward)  # Multiply the probability of the taken action and reward.
            except RuntimeError:  # except for overflow problem.
                print(-log_prob)
                print(reward)
        self.optimizerRL.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() * suppression
        policy_loss.backward()
        self.optimizerRL.step()

        del self.saved_log_probs[:]
        del self.saved_actions[:]

        return output, loss.data[0], (policy_loss.data[0], RL)


class ID_AT_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aspect_size, embeddings=None):
        super(ID_AT_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.actions = 2
        self.lr = 0.0005
        self.l2_weight = 0.001
        # self.lr_decay = 0.1

        self.AE = nn.Embedding( aspect_size, input_size )
        if not embeddings is None:
            self.AE.weight = nn.Parameter( embeddings )

        self.W_h = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_v = nn.Linear( self.input_size, self.input_size )
        self.w = nn.Linear( self.hidden_size + self.input_size, 1 )
        self.W_p = nn.Linear( self.hidden_size, self.hidden_size )
        self.W_x = nn.Linear( self.hidden_size, self.hidden_size )

        self.rnn = nn.LSTMCell(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        # self.softmax = nn.LogSoftmax()
        # self.softmax= nn.Softmax(dim=1)
        self.attn_softmax = nn.Softmax( dim=0 )

        self.PNet = nn.Linear(2 * self.hidden_size + self.input_size, self.actions)
        self.saved_actions = []
        self.saved_log_probs = []
        self.gamma = 1
        self.miu = 0.1

        self.optimizerRL = torch.optim.Adam(self.PNet.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        self.optimizerC = torch.optim.Adam([
            {'params': self.rnn.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.dropout.parameters()},
            # {'params': self.softmax.parameters()},
            {'params': self.AE.parameters()},
            {'params': self.W_h.parameters()},
            {'params': self.W_v.parameters()},
            {'params': self.w.parameters()},
            {'params': self.W_p.parameters()},
            {'params': self.W_x.parameters()},
            {'params': self.attn_softmax.parameters()},
            ], lr=self.lr, weight_decay=self.l2_weight)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    def forward(self, inputs, aspect):
        # print(input)
        del self.saved_log_probs[:]
        del self.saved_actions[:]
        length = inputs.size()[0]
        # print length
        ht, ct = self.init_hidden(inputs.data.is_cuda)

        aspect_embedding = self.AE(aspect)
        aspect_embedding = aspect_embedding.view(-1)

        output = ht
        retain_len = 0
        for t in range(length):
            state = torch.cat((ct.view(-1), ht.view(-1), inputs[t].view(-1))).data
            action = self.select_action(state)
            # print(action)
            if action == 1:  # the word is retained.
                ht, ct = self.rnn(inputs[t], (ht, ct))
                retain_len += 1
                output = torch.cat((output, ht), dim=0)
            # if action == 1 or action == 0:
            #     ht, ct = self.rnn(inputs[t], (ht, ct))
        # print(retain_len)
        # r = ht.view(1, -1)  # just reshape

        hidden = ht.view( 1, -1 )
        output = output.view( output.size()[0], -1 )
        aspect_embedding = self.AE( aspect )
        aspect_embedding = aspect_embedding.view( 1, -1 )
        # print(aspect)
        length = output.size()[0]
        aspect_embedding = aspect_embedding.expand( length, -1 )
        M = F.tanh( torch.cat( (self.W_h( output ), self.W_v( aspect_embedding )), dim=1 ) )
        weights = self.attn_softmax( self.w( M ) ).t()
        # print(weights)
        r = torch.matmul( weights, output )
        r = F.tanh( torch.add( self.W_p( r ), self.W_x( hidden ) ) )

        r = self.dropout(r)
        decoded = self.decoder(r)
        # print(decoded)
        # output = self.softmax(decoded)
        output = decoded
        return output, (inputs.size()[0], retain_len)

    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable(torch.zeros(1, self.hidden_size)), Variable(torch.zeros((1, self.hidden_size))))

    def select_action(self, state):
        # print(state.view(1, -1))
        # state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.view(1, -1)
        probs = F.softmax(self.PNet(Variable(state)), dim=1)
        action = probs.multinomial()  # sample an action from the computed distribution.
        self.saved_actions.append(action.data[0, 0])
        log_prob = torch.gather(probs, 1, action).log()
        self.saved_log_probs.append(log_prob)
        return action.data[0, 0]

    def optimize_step(self, category_tensor, line_tensor, aspect, freeze=False):
        # print(line_tensor)
        aspect_a = np.array( [aspect], dtype=np.int64 )
        aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
        output, (l, retained_) = self.forward(line_tensor, Variable( aspect_tensor ))
        self.train()

        suppression = 0.1
        # suppression = 1
        L = line_tensor.size(0)
        retained = sum(self.saved_actions)
        L_ = L - retained
        assert retained == retained_

        loss = Variable(torch.zeros(1))
        if not freeze:
            # optimize Classification Part
            loss = F.cross_entropy(output, category_tensor)
            self.optimizerC.zero_grad()
            loss.backward()
            self.optimizerC.step()

        # Optimize Reinforcement Part:

        # RL = np.log(torch.index_select(output, 1, category_tensor).data[0, 0]) + self.miu * L_/L  # TODO
        RL = torch.index_select(output, 1, category_tensor).data[0, 0] + self.miu * L_ / L
        policy_loss = []
        rewards = []
        for r in range(line_tensor.size(0)):
            R = self.gamma * RL
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        # print(rewards)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            try:
                policy_loss.append(-log_prob * reward)  # Multiply the probability of the taken action and reward.
            except RuntimeError:  # except for overflow problem.
                print(-log_prob)
                print(reward)
        self.optimizerRL.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() * suppression
        policy_loss.backward()
        self.optimizerRL.step()

        del self.saved_log_probs[:]
        del self.saved_actions[:]

        return output, loss.data[0], (policy_loss.data[0], RL)


if __name__ == '__main__':
    # Just for debug:
    pretrainedModel = LSTM(300, 128, 2)
    pretrained_dict = pretrainedModel.state_dict()
    model = ID_LSTM(300, 128, 2)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    input_tesnor = Variable(torch.rand(8, 1, 300))
    category_tensor = Variable(torch.LongTensor([1]))
    # print(category_tensor)
    print(model(input_tesnor))
    for child in model.children():
        print(child)
    print(model.children())
    print(model.optimize_step(category_tensor, input_tesnor))
