import codecs
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
from gensim.models.word2vec import Word2Vec
import torch.nn.functional as F
from torch.optim import lr_scheduler

import re
from model_origin import *
import sys

w2v_model = Word2Vec.load_word2vec_format('embedding_all_Glove300.txt')

max_length = 80 # TODO
cuda_flag = True and torch.cuda.is_available()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def line_to_tensor(words, model, padding=False):
    size = model.vector_size
    # tensor = torch.zeros(1, 1, max_length, size)
    vec_list = []
    retained_words = []
    for li, w in enumerate(words):
        w = w.lower()
        if w in model:
            retained_words.append(w)
            vec = model[w]
            # vec_tensor = torch.from_numpy(vec)
        else:
            # print(w)
            vec = model['<unk>']
        vec_list.append(vec)

    if len(vec_list) == 0:
        # print(words)
        return torch.zeros(1, 1, size)
    # length = len(vec_list)
    feature_array = np.asarray(vec_list)
    # print(feature_array)
    if padding:
        feature_array = np.lib.pad(feature_array, ((0, max_length - len(feature_array)), (0, 0)), 'constant')
    # print(feature_array)
    tensor = torch.from_numpy(feature_array)  # N*d
    # print(tensor)
    tensor = tensor.view(tensor.size()[0], 1, -1)   # 1*1*max_length*dim
    return tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(x_feature_train, train_y, x_feature_dev, dev_y, rnn, epochs, savename, freeze=False):
    n_epochs = epochs

    print(cuda_flag)
    if cuda_flag:
        rnn = rnn.cuda()

    # print_every = 100
    plot_every = 200
    # plot_every = 10

    # Keep track of losses for plotting
    current_loss1 = []
    current_loss2 = []
    current_reward = []
    all_losses1 = []
    all_losses2 = []
    all_reward = []
    # start = time.time()
    best_acc = 0

    for epoch in range(1, n_epochs + 1):
        # print("****New epoch***")
        index_list = np.arange(len(x_feature_train))
        np.random.shuffle(index_list)
        # print(index_list)
        for index in index_list:
            line_tensor = x_feature_train[index]
            category_tensor = train_y[index]
            if cuda_flag:
                line_tensor = line_tensor.cuda()
                category_tensor = category_tensor.cuda()
            # print(line_tensor)
            # print(category_tensor)
            output, loss1, loss2 = rnn.optimize_step(Variable(category_tensor), Variable(line_tensor), freeze=freeze)
            current_loss1.append(loss1)
            current_loss2.append(loss2[0])
            current_reward.append(loss2[1])

            # Add current loss avg to list of losses
            if (index+1) % plot_every == 0:
                # print(batch_epoch)
                all_losses1.append(sum(current_loss1) / len(current_loss1))
                current_loss1 = []
                all_losses2.append(sum(current_loss2) / len(current_loss2))
                current_loss2 = []
                all_reward.append(sum(current_reward) / len(current_reward))
                current_reward = []

        test_predict, (origin_mean, retain_mean) = predict(rnn, x_feature_dev)
        # for i in range(len(test_predict)):
        #     print test_predict[i], dev_y[i]
        # print len(test_predict), len(dev_y)
        pred_acc = float(sum(test_predict == dev_y)) / float(len(dev_y))
        # print sum(test_predict == dev_y), pred_acc
        if pred_acc > best_acc:
            best_acc = pred_acc
            torch.save(rnn, "backup/%s_%.3f.pt" % (savename, pred_acc))
        print("Epoch %d:  Acc:%f   retained:(%.3f, %.3f)" % (epoch, pred_acc, origin_mean, retain_mean))
        # test_acc.append(pred_acc)

    # torch.save(rnn, "backup/polarity_cnn_last.pt")
    # print(max_acc)
    plt.figure()
    plt.plot(all_losses1)
    time_stamp = time.asctime().replace(':', '_').split()
    plt.savefig("fig/fooLoss_%s.png" % '_'.join(time_stamp))
    plt.figure()
    plt.plot(all_losses2)
    plt.savefig("fig/fooPloss_%s.png" % '_'.join(time_stamp))
    plt.figure()
    plt.plot(all_reward)
    plt.savefig("fig/fooReward_%s.png" % '_'.join(time_stamp))
    print(time_stamp)
    # plt.show()
    return best_acc


def category_from_output(output):
    # print(output)
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i[0][0]
    return category_i


def predict(rnn, x_feature_dev):
    rnn.eval()
    predicted = []
    o_l = []
    r_l = []
    for i in range(len(x_feature_dev)):
        # category_tensor = Variable(torch.LongTensor([dev_y[i] - 1]))
        line_tensor = x_feature_dev[i]
        if cuda_flag:
            line_tensor = line_tensor.cuda()
        output, (origin_len, retain_len) = rnn(Variable(line_tensor))
        # print(output)
        category_i = category_from_output(output)
        predicted.append(category_i)
        if not origin_len == None:
            o_l.append(origin_len)
            r_l.append(retain_len)

    origin_mean = np.mean(o_l)
    retain_mean = np.mean(r_l)
    # print(predicted)
    predicted = np.array(predicted)
    # print(predicted)
    # print(dev_y)

    return predicted, (origin_mean, retain_mean)


def load_pretrained_model(pretrainedModel, Newmodel):
    pretrained_dict = pretrainedModel.state_dict()
    model_dict = Newmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    Newmodel.load_state_dict(model_dict)


def pre_train():
    # f1 = codecs.open("dataset2/train_x.txt", encoding='utf-8')
    # f2 = open("dataset2/train_label.txt")
    # f3 = codecs.open("dataset2/dev_x.txt", encoding='utf-8')
    # f4 = open("dataset2/dev_label.txt")
    f1 = codecs.open( "SemEval_2015/Laptops_Train_Data_noasp.txt", encoding='utf-8' )
    f2 = open( "SemEval_2015/Laptops_Train_label_Data.txt" )
    f3 = codecs.open( "SemEval_2015/Laptops_Test_Data_noasp.txt", encoding='utf-8' )
    f4 = open( "SemEval_2015/Laptops_Test_label_Data.txt" )


    train_x_text = [clean_str(line) for line in f1]
    train_y = [int(line.strip('\r\n'))+1 for line in f2]
    dev_x_text = [clean_str(line) for line in f3]
    dev_y = [int(line.strip('\r\n'))+1 for line in f4]

    train_y_tensor = [torch.LongTensor([i]) for i in train_y]
    x_feature_train = [line_to_tensor(s.split(), w2v_model) for s in train_x_text] #N*1*dim
    x_feature_dev = [line_to_tensor(s.split(), w2v_model) for s in dev_x_text]

    ## PreCNet
    
    PreCNet = TextRNN(300, 128, 3)
    acc = 0
    while(acc < 0.80): # For a good pre-train CNet.
        print("Pre-training CNet")
        acc = train(x_feature_train, train_y_tensor, x_feature_dev, dev_y, PreCNet, epochs=1, savename="origin_PreCNet")

        # predicted, (origin_mean, retain_mean) = predict(PreCNet, x_feature_train)
        # acc = float(sum(predicted == train_y)) / float(len(train_y))
        print("On dev set: %f   " % (acc))
    
    # PrePNet
    # PreCNet = torch.load("backup/HalfCNet_0.739.pt")
    print("Pre-training PNet")
    PrePNet = ID_LSTM(300, 128, 3)
    load_pretrained_model(PreCNet, PrePNet)
    for param in PrePNet.parameters():
        param.requires_grad = False  # Freeze feature and CNet layers.
    for param in PrePNet.PNet.parameters():
        param.requires_grad = True
    train(x_feature_train, train_y_tensor, x_feature_dev, dev_y, PrePNet, epochs=50, savename="PrePNet", freeze=True)

def joint_train(pretrained_model):
    ## JointModel
    f1 = codecs.open("dataset2/train_x.txt", encoding='utf-8')
    f2 = open("dataset2/train_label.txt")
    f3 = codecs.open("dataset2/dev_x.txt", encoding='utf-8')
    f4 = open("dataset2/dev_label.txt")

    train_x_text = [clean_str(line) for line in f1]
    train_y = [int(line.strip('\r\n')) for line in f2]
    dev_x_text = [clean_str(line) for line in f3]
    dev_y = [int(line.strip('\r\n')) for line in f4]
    train_y_tensor = [torch.LongTensor([i]) for i in train_y]
    x_feature_train = [line_to_tensor(s.split(), w2v_model) for s in train_x_text]  # N*1*dim
    x_feature_dev = [line_to_tensor(s.split(), w2v_model) for s in dev_x_text]

    PrePNet = torch.load(pretrained_model)
    print("Joint Training")
    predicted, (origin_mean, retain_mean) = predict(PrePNet, x_feature_dev)
    acc = float(sum(predicted == dev_y)) / float(len(dev_y))
    print("%f   (%.3f, %.3f)" % (acc, origin_mean, retain_mean))
    JointModel = ID_LSTM(300, 128, 2)
    
    load_pretrained_model(PrePNet, JointModel)
    for param in JointModel.parameters():
        param.requires_grad = True
    train(x_feature_train, train_y_tensor, x_feature_dev, dev_y, JointModel, epochs=50, savename="JointModel", freeze=False)
    predicted, (origin_mean, retain_mean) = predict(JointModel, x_feature_train)
    acc = float(sum(predicted == train_y)) / float(len(train_y))
    print("%f   (%.3f, %.3f)" % (acc, origin_mean, retain_mean))
    torch.save(JointModel, "backup/744_JM_last%.3f.pt" % acc)

    print("Finish")

def test(pretrained_model):  # load a model and test on a sentence.
    test_model = torch.load(pretrained_model)
    # sentence = "even if the ring has a familiar ring , it's still unusually crafty and intelligent for hollywood horror ."
    # sentence = "if you sometimes like to go to the movies to have fun , wasabi is a good place to start ."
    sentence = "with a cast that includes some of the top actors working in independent film , lovely & amazing involves us because it is so incisive , so bleakly amusing about how we go about our lives ."
    x_text_list = clean_str(sentence).split()
    x_feature_tensor = line_to_tensor(x_text_list, w2v_model)
    test_model.eval()
    output = test_model(Variable(x_feature_tensor))
    print(x_text_list)
    retained_s = []
    for i, w in enumerate(x_text_list):
        if test_model.saved_actions[i] == 1:
            retained_s.append(w)
        else:
            retained_s.append('---')
    print(retained_s)
    print(test_model.saved_actions)

    probs = [np.exp(log_p.data[0,0]) for log_p in test_model.saved_log_probs]
    print(probs)
    print(output)


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 1: # FOr IDE Running.
        mode = "joint-train"
        pretrained_model_name = "backup/" + "PrePNet_0.777.pt"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        pretrained_model_name = "backup/" + sys.argv[2]
        print(pretrained_model_name)
    
    if mode == "pre-train":
        pre_train()
    if mode == "joint-train":
        joint_train(pretrained_model_name)
    if mode == "test":
        test(pretrained_model_name)
