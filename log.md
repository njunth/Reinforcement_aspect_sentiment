# MR Baseline:
* BiLSTM: 0.78
* LSTM: 0.787 / 0.802   0.792

# Experiment:
## Ex1: not best pretrained.+ no smooth gradient.
* pre-CNet: 0.778 / 0.8
* pre-PNet: 0.78 / 0.79
* Joint: 0.799 / 0.807
## Ex2: best pretrained + smooth gradient.
* pre-CNet : 0.801
* pre-PNet : 0.805
* JointNet : 0.808
## Ex2: best pretrained + smooth gradient.
* pre-CNet : 0.799
* pre-PNet : 0.812
* JointModel: 0.813
## Ex3: better PNet + dropout
* preCNet : 0.801
* PrePNet : 0.815
* JointModel: 0.809
## Ex4: best CNet + dropout
## Ex5: preCNet Lr = 0.0005

# Joint Training:
* add loss
* step by step (次序)

# reward
* miu = 0

## Learning Rate:
* 在最后fine-tune时，如果lr很大，就会回退到0.5

# DropOut

# Test Delete
* Ex1: PrePNet_0.815
    * action: 0.804/0.805/0.81/0.806/0.809/0.809/0.807/0.794/0.805/0.807
    * no action: 0.801
* Ex2: Counting:
    * 未处理的句子平均长度
        * train: 20.9
        * dev: 21.3
    * clean 之后：
        * train: 20.35
        * dev: 20.614
    * word2vec之后：
        * 16.3301680856
        * 16.609
    * Glove:
        * 预处理： 20.3858563121 20.2338210467
        * 20.2063959834 / 20.449 (train/dev)
        * add UNK : 20.3580004366 / 20.614
        
        
    * PrePNet:dev:(16.600999999999999, 15.063000000000001) train:(16.318, 14.367)
* Ex3: remove relu in policy:
    0.780000   (16.601, 15.610)
* Glove:
    * preprocessing: 18764 / 18002
* UNK:
* test:['if', 'you', 'sometimes', 'like', 'to', 'go', 'to', 'the', 'movies', 'to', 'have', 'fun', ',', 'wasabi', 'is', 'a', 'good', 'place', 'to', 'start']
    * ['if', 'you', 'sometimes', 'like', '---', 'go', 'to', '---', '---', 'to', 'have', 'fun', ',', 'wasabi', 'is', '---', 'good', 'place', 'to', 'start']
    * [0.99999284744274075, 0.95050918941508955, 0.99974435567753672, 0.99505335079479018, 0.46360448751327071, 0.9301480681280927, 0.65892506517659233, 0.93403411200751185, 0.99776726969268104, 0.66401904258032951, 0.98291516323709738, 0.98862916228343722, 0.999999940395357, 0.94141113878969962, 0.99998676776909512, 0.97805237681137513, 0.99999624490738093, 0.99999547004679812, 0.99992239475169908, 0.99951863289168086]
    * ['if', 'you', 'sometimes', 'like', '---', 'go', 'to', '---', '---', '---', 'have', 'fun', ',', 'wasabi', 'is', '---', 'good', 'place', 'to', 'start']
    * [0.99999284744274075, 0.95050918941508955, 0.99974435567753672, 0.99505335079479018, 0.46360448751327071, 0.9301480681280927, 0.65892506517659233, 0.93403411200751185, 0.99776726969268104, 0.33598093396125672, 0.9727410679351689, 0.98491102449986379, 0.999999940395357, 0.9405256517143753, 0.99998843669846649, 0.97568774215366549, 0.99999660253533484, 0.9999959468841837, 0.99993032216770061, 0.99956470728450941]
    * ['if', 'you', 'sometimes', 'like', 'to', 'go', 'to', '---', '---', '---', 'have', 'fun', ',', 'wasabi', 'is', '---', 'good', 'place', 'to', 'start']
    * [0.99999284744274075, 0.95050918941508955, 0.99974435567753672, 0.99505335079479018, 0.53639554173560666, 0.96036034805783765, 0.77761458500071945, 0.90528565966787033, 0.99669593561129388, 0.25459624936123293, 0.98143476325746504, 0.98826611041591383, 0.999999940395357, 0.94263774229819319, 0.9999874234199122, 0.97737967941374571, 0.99999636411668469, 0.99999564886084791, 0.99992603063594199, 0.99954020977192559]
    * ['with', 'a', 'cast', 'that', 'includes', 'some', 'of', 'the', 'top', 'actors', 'working', 'in', 'independent', 'film', ',', 'lovely', 'amazing', 'involves', 'us', 'because', 'it', 'is', 'so', 'incisive', ',', 'so', 'bleakly', 'amusing', 'about', 'how', 'we', 'go', 'about', 'our', 'lives']
    * ['---', '---', '---', 'that', 'includes', 'some', '---', '---', 'top', '---', '---', '---', '---', '---', ',', 'lovely', 'amazing', 'involves', 'us', 'because', 'it', 'is', 'so', 'incisive', ',', 'so', 'bleakly', 'amusing', 'about', 'how', 'we', 'go', 'about', '---', 'lives']
    * [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    * [0.77358072457921678, 0.99574667220107449, 0.9857958551661512, 0.99999922513963391, 0.22955763948425587, 0.99976873397796151, 0.97368240439453124, 0.39340620983168867, 0.99601322410578175, 0.99999529123323505, 0.99952208996105663, 0.99577081217341967, 0.99998170137412878, 0.99999582767492967, 1.0, 0.99932295086129275, 0.99999856948852539, 0.98714363599740851, 0.99996203184176702, 1.0, 1.0, 0.99999988079070334, 1.0, 0.99929553270141114, 1.0, 1.0, 0.999999940395357, 0.999999940395357, 0.99991279840164815, 0.99999380111705716, 0.99999117851237451, 0.99995303154066206, 0.9044802180215894, 0.96508240836652692, 0.14287331270113715]

GLove：
# 不收敛预训练：
## EX1:
* HalfCNet: 0.739000 / 0.744597 (dev/ train)
    * PrePNet:
        * (1) 0.742000   (20.614, 12.859):  / (train_set: 0.770028   (20.358, 12.637))
            * Joint: 0.793000   (20.614, 12.923) / 0.903624   (20.358, 12.654)
            * No suppresion Joint: 0.786000   (20.614, 15.927)
            * Long Joint:
                * 0.775000   (20.614, 14.430)
                * 0.778000   (20.614, 14.481)
                * 0.803000   (20.614, 15.160) / 0.974023   (20.358, 14.896)****
                * 0.784000   (20.614, 13.635)
                * 0.781000   (20.614, 14.886)

        * (2) 0.737000   (20.614, 11.857)
            * Joint:
        * (3) 0.731000   (20.614, 12.226)
    * no suppression PrePNet:
        * 0.750000   (20.614, 8.526) / 0.751364   (20.358, 8.538)
            * Long Joint: 0.812000   (20.614, 17.762)
            * $$$
    * 不收敛 PrePNet：
        * 0.689 / 0.696682   (20.358, 6.743)
            * Joint： 非常差
        * 0.733 (20.614, 6.767) / 0.679655   (20.358, 6.671)
            * Joint: 0.757000   (20.614, 9.227) / 0.914211   (20.358, 9.131) * loss在降
        * 0.742000(Half)   (20.614, 10.148) / 0.753329   (20.358, 9.929)
            * Joint: 0.796000   (20.614, 12.228) /  0.950775   (20.358, 11.959)
            * Long（100）: 0.801000   (20.614, 11.175)
        * 0.738000(Half)   (20.614, 12.349) / 0.761624   (20.358, 12.042) 
            * Joint : reward 升了， 但是PLoss也升了。Loss明显降了 last train: 0.970421   (20.358, 18.786)
                * 0.813000   (20.614, 19.388) / ?
                * 0.789000   (20.614, 17.290) / ?
                * $$$
            * Long(100): 
                * 0.795000   (20.614, 12.187)
## Ex2:
* HalfCNet: / > 0.8
# 收敛预训练：
## EX1:
* PreCNet: best_dev: 0.808 / 0.957869   last_train:0.966492 
    * PrePNet: 0.812000    (20.614, 20.090)  / 0.952303   (20.358, 19.811)
        * Long Joint: 0.813000    (20.614, 18.001) / 0.972495   (20.358, 17.633)
## Ex2: best PreCNet:
* 0.817: / 0.963108

## 1000 744 JM
* 279:  0.798000   (20.614, 16.967)

# change policy Loss： 
## Ex1: no log in reward. under 0.739CNEt
* 0.742000(Half)    
    * 0.809000   (20.614, 18.757): PLoss在降，reward在升，说明去掉log是有好处的。
* 0.744(Half)   
    * 95:  0.798000   (20.614, 14.435)
* 0.764(best)   (20.614, 12.351)
    * 25:  0.801000   (20.614, 11.456)
    * 24:  0.808000   (20.614, 16.679)
* 0.767(best)   (20.614, 14.317)
    * 28:  0.807000   (20.614, 16.514)
    * 21:  0.806000   (20.614, 14.933)

## best PrePNet under 0.739
26:  0.764000   (20.614, 12.351)
95:  0.767000   (20.614, 14.317)

## 收敛：no log
* 0.812:
    * 2:  0.753000   (20.614, 7.781) ??? 为什么降了。
        * Epoch 18:  Acc:0.812000   retained:(20.614, 20.210)
        * Epoch 50:  Acc:0.816000   retained:(20.614, 20.373)

## MIU in RL:

## no suppression: under 0.739


## another select_action

## reduction baseline

# Plan
## DONE:
* chek freeze
* check pretrained
* cuda
* 保存最佳pre-trained model

## Doing:
* 去掉log
* 去掉baseline
* 稳定到0.812
* L_/L到底能不能反向传播？还是被当做常数？

## TODO
* - L_/L
* 其他优化算法。
* 其他联合训练方法: 防止震荡。
* learning_rate
* different UNK
* Two layer policy.
* drop-out.
* rl = log(p - 0.5)
* 其他loss
* 其他强化学习算法。
* overflow : -inf : double?
* smooth gradient
* No PrePNet

* 除了 Adam
* 用强化学习最大化两个输出的差值。

# 观察：
* supression:
* pretrain stopping:
* 效果好的倾向于少删词， 多删词的效果略差
* 完全收敛的pretrain几乎不怎么删词。
* pretrain PNet时，如果不收敛，那么效果会很差。
* Joint 倒退问题。一般loss不收敛。
* 简单LSTM已经取得很好结果，删词未必有意义。

# Question:
* 如何预训练，分别预训练到什么程度。
* 最后联合训练怎么训练，是迭代还是别的？
* 为什么复现会倾向于不删词。
* 不稳定问题。
* 其他的RL算法
* 开源。
