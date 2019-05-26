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
from LSTM_attention import *
import sys

# w2v_model = Word2Vec.load_word2vec_format('embedding_all_Glove300.txt')
w2v_model = Word2Vec.load_word2vec_format('vectors300.bin', binary=True)
w2v_model_2 = Word2Vec.load_word2vec_format('embedding_all_Glove300.txt')

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

def label_indicator(sentence, dev_x_text):
    label_indicat = {}
    i = 0
    for s in sentence:
        words = s.split()
        if words[0] not in label_indicat:
            # print words[0]
            label_indicat[words[0]] = i
            i = i+1
    for s in dev_x_text:
        words = s.split()
        if words[0] not in label_indicat:
            # print words[0]
            label_indicat[words[0]] = i
            i = i+1
    return label_indicat

def text_aspect(l_indicat, train_x_text):
    aspect = []
    for s in train_x_text:
        words = s.split()
        if words[0] in l_indicat:
            aspect.append(l_indicat[words[0]])
        # else:
        #     print words[0]
    return aspect


def line_to_tensor(words, model, padding=False):
    size = model.vector_size
    # tensor = torch.zeros(1, 1, max_length, size)
    vec_list = []
    retained_words = []
    for li, w in enumerate(words):
        w = w.lower()
        if li == 0:
            # l = label_indicat[w]
            continue
        vec = []
        if w in model:
            retained_words.append( w )
            for m in model[w]:
                vec.append( float( m ) )
            # vec_tensor = torch.from_numpy(vec)
        else:
            # print(w)
            for m in w2v_model_2['<unk>']:
                vec.append( float( m ) )
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
    tensor = torch.FloatTensor(feature_array)  # N*d
    # print(tensor)
    tensor = tensor.view(tensor.size()[0], 1, -1)   # 1*1*max_length*dim
    return tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(aspect, x_feature_train, train_y, aspect_test, x_feature_dev, dev_y, rnn, epochs, savename, freeze=False):
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
            # print index
            line_tensor = x_feature_train[index]
            category_tensor = train_y[index]
            # print category_tensor
            if cuda_flag:
                line_tensor = line_tensor.cuda()
                category_tensor = category_tensor.cuda()
            # print(line_tensor)
            # print(category_tensor)
            # aspect_tensor = torch.LongTensor( aspect[index] ).view( 1, -1 )
            output, loss1, loss2 = rnn.optimize_step(Variable(category_tensor),
                                                     Variable(line_tensor), aspect[index], freeze=freeze)
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

        test_predict, (origin_mean, retain_mean) = predict(rnn, x_feature_dev, aspect_test)
        # for i in range(len(test_predict)):
        #     print test_predict[i], dev_y[i]
        # print test_predict
        pred_acc = score_polarity(test_predict, dev_y)
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


def score_polarity(predicted, golden):
    labels = 0.
    correct = 0.
    for i in range(len(predicted)):
        # print predicted[i]
        # print golden[i]
        correct += sum([predicted[i] == golden[i]])
        # print correct
        labels += 1.
    acc = float(correct)/float(labels)
    return acc

def category_from_output(output):
    # print(output)
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i[0][0]
    return category_i


def predict(rnn, x_feature_dev, aspect_test):
    rnn.eval()
    if cuda_flag:
        rnn = rnn.cuda()
    predicted = []
    o_l = []
    r_l = []
    for i in range(len(aspect_test)):
        # category_tensor = Variable(torch.LongTensor([dev_y[i] - 1]))
        line_tensor = x_feature_dev[i]
        # aspect_tensor = torch.LongTensor( aspect_test[i] )
        if cuda_flag:
            line_tensor = line_tensor.cuda()
            aspect_tensor = aspect_tensor.cuda()
        # print aspect_test[i]
        # print aspect_tensor
        aspect_a = np.array( [aspect_test[i]], dtype=np.int64 )
        aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
        # print len(line_tensor), aspect_tensor
        output, (origin_len, retain_len) = rnn(Variable(line_tensor), Variable( aspect_tensor ))
        # print (origin_len, retain_len)
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
    f1 = codecs.open("SemEval_2014/Restaurant_Train_Data.txt", encoding='utf-8')
    f2 = open("SemEval_2014/Restaurant_Train_label_Data.txt")
    f3 = codecs.open("SemEval_2014/Restaurant_test_Data.txt", encoding='utf-8')
    f4 = open("SemEval_2014/Restaurant_test_label_Data.txt")
    # f1 = codecs.open( "SemEval_2015/Laptops_Train_Data.txt", encoding='utf-8' )
    # f2 = open( "SemEval_2015/Laptops_Train_label_Data.txt" )
    # f3 = codecs.open( "SemEval_2015/Laptops_Test_Data.txt", encoding='utf-8' )
    # f4 = open( "SemEval_2015/Laptops_Test_label_Data.txt" )


    # train_x_text = [clean_str(line) for line in f1]
    train_x_text = [line for line in f1]
    train_y = [int(line.strip('\r\n')) for line in f2]
    # dev_x_text = [clean_str(line) for line in f3]
    dev_x_text = [line for line in f3]
    dev_y = [int(line.strip('\r\n')) for line in f4]

    l_indicat = label_indicator( train_x_text, dev_x_text)
    aspect = text_aspect(l_indicat, train_x_text)
    aspect_test = text_aspect(l_indicat, dev_x_text)

    print len(aspect_test), len(aspect), len(train_x_text), len(dev_x_text), len(l_indicat)
    aspect_e_l = []
    for a in l_indicat:
        print(a)
        # a_e = torch.sum( line_to_tensor( clean_str( a ).split(), w2v_model ), 0 )
        if a in w2v_model:
            # print w2v_model[a]
            a_e = w2v_model[a]
        # a_e = torch.rand(1, 300)
        a_e = torch.FloatTensor(a_e)
        aspect_e_l.append( a_e.view(1, -1) )
    aspect_embeds = torch.cat( aspect_e_l, 0 )

    train_y_tensor = [torch.LongTensor([i]) for i in train_y]
    x_feature_train = [line_to_tensor(s.split(), w2v_model) for s in train_x_text] #N*1*dim
    x_feature_dev = [line_to_tensor(s.split(), w2v_model) for s in dev_x_text]

    ## PreCNet
    
    PreCNet = TextRNN(300, 300, 3, len(l_indicat), aspect_embeds)
    # PreCNet = AE_LSTM( 300, 500, 3, len( l_indicat ), aspect_embeds )
    acc = 0
    print(PreCNet.AE.weight)
    while(acc < 0.76): # For a good pre-train CNet.
        print("Pre-training CNet")
        acc = train(aspect, x_feature_train, train_y_tensor, aspect_test, x_feature_dev, dev_y, PreCNet, epochs=30, savename="origin_PreCNet")
        # predicted, (origin_mean, retain_mean) = predict(PreCNet, x_feature_train)
        # acc = float(sum(predicted == train_y)) / float(len(train_y))
        print("On dev set: %f   " % (acc))
    # print(PreCNet.AE.weight)
    # PrePNet
    # PreCNet = torch.load("backup/origin_PreCNet_0.824.pt")
    # aspect_embeds = PreCNet.getembedding(l_indicat)
    # print aspect_embeds
    print("Pre-training PNet")
    # PrePNet = ID_LSTM(300, 300, 3)
    PrePNet = ID_AT_LSTM( 300, 300, 3, len(l_indicat), aspect_embeds )
    load_pretrained_model(PreCNet, PrePNet)
    for param in PrePNet.parameters():
        param.requires_grad = False  # Freeze feature and CNet layers.
    for param in PrePNet.PNet.parameters():
        param.requires_grad = True
    train(aspect, x_feature_train, train_y_tensor, aspect_test, x_feature_dev, dev_y, PrePNet, epochs=50, savename="PrePNet", freeze=True)

def joint_train(pretrained_model):
    ## JointModel
    # f1 = codecs.open("dataset2/train_x.txt", encoding='utf-8')
    # f2 = open("dataset2/train_label.txt")
    # f3 = codecs.open("dataset2/dev_x.txt", encoding='utf-8')
    # f4 = open("dataset2/dev_label.txt")
    #
    # train_x_text = [clean_str(line) for line in f1]
    # train_y = [int(line.strip('\r\n')) for line in f2]
    # dev_x_text = [clean_str(line) for line in f3]
    # dev_y = [int(line.strip('\r\n')) for line in f4]
    # train_y_tensor = [torch.LongTensor([i]) for i in train_y]
    # x_feature_train = [line_to_tensor(s.split(), w2v_model) for s in train_x_text]  # N*1*dim
    # x_feature_dev = [line_to_tensor(s.split(), w2v_model) for s in dev_x_text]
    f1 = codecs.open( "SemEval_2014/Restaurant_Train_Data.txt", encoding='utf-8' )
    f2 = open( "SemEval_2014/Restaurant_Train_label_Data.txt" )
    f3 = codecs.open( "SemEval_2014/Restaurant_test_Data.txt", encoding='utf-8' )
    f4 = open( "SemEval_2014/Restaurant_test_label_Data.txt" )
    # f1 = codecs.open( "SemEval_2015/Laptops_Train_Data.txt", encoding='utf-8' )
    # f2 = open( "SemEval_2015/Laptops_Train_label_Data.txt" )
    # f3 = codecs.open( "SemEval_2015/Laptops_Test_Data.txt", encoding='utf-8' )
    # f4 = open( "SemEval_2015/Laptops_Test_label_Data.txt" )


    # train_x_text = [clean_str(line) for line in f1]
    train_x_text = [line for line in f1]
    train_y = [int( line.strip( '\r\n' ) ) for line in f2]
    # dev_x_text = [clean_str(line) for line in f3]
    dev_x_text = [line for line in f3]
    dev_y = [int( line.strip( '\r\n' ) ) for line in f4]

    l_indicat = label_indicator( train_x_text, dev_x_text )
    aspect = text_aspect( l_indicat, train_x_text )
    aspect_test = text_aspect( l_indicat, dev_x_text )

    print len( aspect_test ), len( aspect ), len( train_x_text ), len( dev_x_text ), len( l_indicat )
    aspect_e_l = []
    for a in l_indicat:
        print(a)
        # a_e = torch.sum( line_to_tensor( clean_str( a ).split(), w2v_model ), 0 )
        if a in w2v_model:
            # print w2v_model[a]
            a_e = w2v_model[a]
        # a_e = torch.rand(1, 300)
        a_e = torch.FloatTensor( a_e )
        aspect_e_l.append( a_e.view( 1, -1 ) )
    aspect_embeds = torch.cat( aspect_e_l, 0 )

    train_y_tensor = [torch.LongTensor( [i] ) for i in train_y]
    x_feature_train = [line_to_tensor( s.split(), w2v_model ) for s in train_x_text]  # N*1*dim
    x_feature_dev = [line_to_tensor( s.split(), w2v_model ) for s in dev_x_text]

    PrePNet = torch.load(pretrained_model)
    print("Joint Training")
    predicted, (origin_mean, retain_mean) = predict(PrePNet, x_feature_dev, aspect_test)
    acc = float(sum(predicted == dev_y)) / float(len(dev_y))
    print("%f   (%.3f, %.3f)" % (acc, origin_mean, retain_mean))
    JointModel = ID_AT_LSTM( 300, 300, 3, len( l_indicat ), aspect_embeds )
    # JointModel = ID_AT_LSTM( 300, 300, 3, len( l_indicat ), aspect_embeds )

    load_pretrained_model(PrePNet, JointModel)
    for param in JointModel.parameters():
        param.requires_grad = True
    train(aspect, x_feature_train, train_y_tensor, aspect_test, x_feature_dev, dev_y, JointModel, epochs=50, savename="JointModel", freeze=False)
    predicted, (origin_mean, retain_mean) = predict(JointModel, x_feature_train)
    acc = float(sum(predicted == train_y)) / float(len(train_y))
    print("%f   (%.3f, %.3f)" % (acc, origin_mean, retain_mean))
    torch.save(JointModel, "backup/744_JM_last%.3f.pt" % acc)

    print("Finish")

def test(pretrained_model):  # load a model and test on a sentence.
    test_model = torch.load(pretrained_model)
    f1 = codecs.open( "SemEval_2014/Restaurant_Train_Data.txt", encoding='utf-8' )
    f3 = codecs.open( "SemEval_2014/Restaurant_test_Data.txt", encoding='utf-8' )
    train_x_text = [line for line in f1]
    dev_x_text = [line for line in f3]

    l_indicat = label_indicator( train_x_text, dev_x_text )
    # sentence = "even if the ring has a familiar ring , it's still unusually crafty and intelligent for hollywood horror ."
    # sentence = "if you sometimes like to go to the movies to have fun , wasabi is a good place to start ."
    delete_sentence("service great pizza and fantastic service.", l_indicat, test_model)
    delete_sentence("food great pizza and fantastic service.", l_indicat, test_model)

    # sentence = "ambience best of all is the warm vibe, the owner is super friendly and service is fast."
    # origin_len = 0.0
    # retain_len = 0.0
    # delete_len = 0.0
    # asp = "price"
    # target = "service"
    # for sentence in dev_x_text:
    #     aspect = clean_str( sentence ).split()[0]
    #     aspect_test = l_indicat[aspect]
    #     if asp != aspect:
    #         continue
    #     x_text_list = clean_str(sentence).split()
    #     x_feature_tensor = line_to_tensor(x_text_list, w2v_model)
    #     test_model.eval()
    #     aspect_a = np.array( [aspect_test], dtype=np.int64 )
    #     aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
    #     output = test_model(Variable(x_feature_tensor), Variable(aspect_tensor))
    #     print(x_text_list)
    #     print len(x_text_list), len(x_feature_tensor)
    #     # origin_len = origin_len + len(x_text_list)
    #     retained_s = []
    #     for i, w in enumerate(x_text_list[1:]):
    #         if w == target:
    #             origin_len = origin_len + 1
    #         if test_model.saved_actions[i] == 1:
    #             retained_s.append(w)
    #             if w == target:
    #                 retain_len = retain_len + 1
    #         else:
    #             retained_s.append('---')
    #             if w == target:
    #                 delete_len = delete_len + 1
    #     print(retained_s)
    # print origin_len, retain_len, delete_len, delete_len/origin_len
    # print origin_len/len(dev_x_text), retain_len/len(dev_x_text), delete_len/len(dev_x_text)




def delete_sentence(sentence, l_indicat, test_model):
    x_text_list = clean_str( sentence ).split()
    x_feature_tensor = line_to_tensor( x_text_list, w2v_model )
    test_model.eval()
    aspect_test = l_indicat[clean_str( sentence ).split()[0]]
    aspect_a = np.array( [aspect_test], dtype=np.int64 )
    aspect_tensor = torch.from_numpy( aspect_a ).view( -1 )
    output = test_model( Variable( x_feature_tensor ), Variable( aspect_tensor ) )
    print(x_text_list)
    print len( x_text_list ), len( x_feature_tensor )
    retained_s = []
    for i, w in enumerate( x_text_list[1:] ):
        if test_model.saved_actions[i] == 1:
            retained_s.append( w )
        else:
            retained_s.append( '---' )
    print(retained_s)
    print(test_model.saved_actions)

    probs = [np.exp( log_p.data[0, 0] ) for log_p in test_model.saved_log_probs]
    print(probs)
    print(output)



if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 1: # FOr IDE Running.
        mode = "joint-train"
        pretrained_model_name = "backup/" + "PrePNet_0.638.pt"
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
