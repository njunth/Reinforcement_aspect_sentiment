import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class TextRNN(nn.Module):  # just a lstm
    def __init__(self, input_size, hidden_size, output_size):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005
        self.W_h = Variable(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.w = Variable( torch.randn(1, hidden_size), requires_grad=True )
        self.W_p = Variable( torch.randn( hidden_size, hidden_size ), requires_grad=True )
        self.W_x = Variable( torch.randn( hidden_size, hidden_size ), requires_grad=True )

        self.rnn = nn.LSTMCell(input_size, hidden_size)  # TODO
        # self.rnn = nn.LSTM(input_size, hidden_size//2, num_layers = 1, bidirectional=True)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.LogSoftmax()
        self.softmax= nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs):

        # print("fuck")
        # print(input)

        ht, ct = self.init_hidden(inputs.data.is_cuda)
        length = inputs.size()[0]
        # print length
        H = []
        for t in range(length):
            ht, ct = self.rnn(inputs[t], (ht, ct))
            H.append(ht)

        H = np.array(H)
        H = torch.FloatTensor( H )
        H = H.view(-1, self.hidden_size)
        H = H.t()
        print H.size()

        M = F.tanh(self.W_h.mm(Variable(H)))

        alpha = self.softmax(self.w.mm(M))

        r = Variable(H).mm(alpha.t())

        h_star = F.tanh(self.W_p.mm(r)+self.W_x.mm(ht.t()))

        r = h_star.view(1, -1)
        # print(r)

        r = self.dropout(r)
        decoded = self.decoder(r)
        # decoded = self.dropout(decoded)
        # output = self.softmax(decoded)
        output = decoded
        return output, (None, None)

    def init_hidden(self, cuda=False):
        if cuda:
            return (Variable(torch.zeros(1, self.hidden_size)).cuda(), Variable(torch.zeros((1, self.hidden_size))).cuda())
        else:
            return (Variable(torch.zeros(1, self.hidden_size)), Variable(torch.zeros((1, self.hidden_size))))

    def optimize_step(self, category_tensor, line_tensor, freeze=False):
        # self.zero_grad()
        self.train()

        # print(line_tensor)
        output, _ = self.forward(line_tensor)
        # print(output)
        # print(category_tensor)
        loss = F.cross_entropy(output, category_tensor)

        self.optimizer.zero_grad()
        loss.backward()

        # self.W_h.data -= self.lr * self.W_h.grad.data
        # self.w.data -= self.lr * self.w.grad.data
        # self.W_p.data -= self.lr * self.W_p.grad.data
        # self.W_x.data -= self.lr * self.W_x.grad.data

        self.optimizer.step()

        return output, loss.data[0], (0, 0)


class ID_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ID_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.actions = 2
        self.lr = 0.0005

        self.rnn = nn.LSTMCell(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.LogSoftmax()
        self.softmax= nn.Softmax(dim=1)

        self.PNet = nn.Linear(2 * self.hidden_size + self.input_size, self.actions)
        self.saved_actions = []
        self.saved_log_probs = []
        self.gamma = 1
        self.miu = 0.1

        self.optimizerRL = torch.optim.Adam(self.PNet.parameters(), lr=self.lr)
        self.optimizerC = torch.optim.Adam([
            {'params': self.rnn.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.dropout.parameters()},
            {'params': self.softmax.parameters()}
            ], lr=self.lr)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    def forward(self, inputs):
        # print(input)
        del self.saved_log_probs[:]
        del self.saved_actions[:]
        length = inputs.size()[0]
        ht, ct = self.init_hidden(inputs.data.is_cuda)

        retain_len = 0
        for t in range(length):
            state = torch.cat((ct.view(-1), ht.view(-1), inputs[t].view(-1))).data
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
        decoded = self.decoder(r)
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

    def optimize_step(self, category_tensor, line_tensor, freeze=False):
        # print(line_tensor)
        output, (l, retained_) = self.forward(line_tensor)
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
