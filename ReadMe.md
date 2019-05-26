# Reinforcement learning for Text Classification

## 运行：
* 因为该模型需要先预训练CNet和PNet然后再联合训练，联合训练时需要指定用到的预训练好的的模型。所以需要确认预训练模型的名称。
模型被保存在backup文件夹中。建议选用acc最高的预训练模型。（acc写在了模型的名称之中）
* 命令行模式：
    * 预训练： ```python main.py pre-train```
    * 联合训练： ```python main.py joint-train NAME_OF_PRETRAINED_MODEL```
    * 测试对一句话的删词效果： ```python main.py test NAME_OF_PRETRAINED_MODEL``` 比如PrePNet_0.777.pt
* 也可以在main.py代码中的```if __name__ == '__main__':```下面自行调整。然后使用IDE运行
* 也可以直接使用使用backup_saved文件夹中已经训练好的PrePNet进行联合训练。
* 训练过程中，会输出每个回合的在dev集上的准确率以及dev集句子的平均长度和删过词的平均长度。

## 代码简介：
* 本代码实现了“Learning Structured Representation for Text Classiﬁcation via Reinforcement Learning”论文中的ID-LSTM模型。
训练过程在main.py文件中，模型实现在model.py文件中。
* main.py
    * train():训练函数
    * pre_train():预训练CNet和PNet
    * Joint_train():最后联合训练
* model.py
    * TextRNN: 一个简单的LSTM，用来预训练CNet.
    * ID_LSTM: 使用强化学习做删词动作的文本分类模型， 在预训练PNet和最后联合训练时会用到。 
    基本框架也是用lstm最后一个隐状态接全连接做二分类，只是在LSTM传递过程中加入删词动作，删词动作的概率由一个PNet产生，
    然后根据概率采样出一个动作，PNet就是一个简单的全连接层加softmax。其optimize_step()函数优化强化学习的部分实现了policy gradient。
* 数据集：MR数据集
* embedding： GloVe40B。

## 实验结果
* 最终JointModel的准确率应该会在0.810以上，但是由于强化学习非常不稳定，所以有可能达不到这么高。实验发现这可能和预训练有很强的关系。我的建议是预训练时PreCNet达到0.76以上就停止，预训练PrePNet时训练到最好结果。这样的话，最后JointModel会稳定一些。
* 但是效果好的模型删词数量不太多，平均每句话只会删2到4个词。尽管我已经在reward里加上了惩罚。

## 注意事项：
* 本代码实现和论文中的公式有一些不同，主要是计算RL是没有加上log，感兴趣的话，可以自行实验加不加log的效果。
* 因为只是为了快速复现，有些地方不是很严谨，比如没有区分dev集和test集。
* pre-train和joint-train各需约半小时。

## 一些观察（Keng）
* 如果将一开始的CNet就训练到完全收敛，那么在最后联合训练时最终的结果虽然很好，但是几乎不会删词。而在论文中的结果则删掉了近一半的词。
* 在pre-train PNet时，还是要训练到收敛，不然效果很差。
* 其实简单不加RL的LSTM已经效果很好了(0.81+)，所以在这个数据集上其实引入强化学习带来的提高很有限。

## 可以进一步研究的问(da)题(keng):
* 预训练问题：
    * 最后的结果其实和预训练有很大的关系，但是论文也没有说明到底该如何预训练，预训练到什么程度？
* 联合训练问题：
    * 联合训练时，我的实现是step-by-step训练两个部分，但是也许有更好的方式。
* reward问题：
    * 文章中的reward在最后分类的概率上加了一个log，但是我个人不是很理解这种做法，做了一些实验，发现去掉log没有太大区别，而且减少了出现overflow的数值问题。
    * 对每一步的动作都加相同的reward，也是一个问题。
    * 在很多强化学习的优化技巧中，会将reward减去一个baseline，这一点值得尝试。

## Copyright
fanzf(AT)nlp.nju.edu.cn
2018/1/31