# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from dataset import load_dataset, unwrap, build_vocab
import numpy as np

def Train_process(training_sentences, product):
    file = "../SemEval_2015/{}_Train_Data_noasp.txt".format( product )
    label_file = "../SemEval_2015/{}_Train_label_Data.txt".format( product )
    out_str = ""
    out_label = ""
    label=[]
    for sentence in training_sentences:
        # print sentence.raw_text
        if len(sentence.opinions) == 0:
            continue
        for opinion in sentence.opinions:
            # if opinion.category not in label:
            #     label.append(opinion.category)
            # temp = opinion.category.split( "#" )
            # out_str += temp[0].replace( "_", " " )
            # out_str += " "
            # out_str += temp[1].replace( "_", " " )
            # out_str += " "

            out_str += opinion.category
            out_str += " "
            out_str += sentence.raw_text
            out_str += "\n"
            out_label += str(opinion.polarity)
            out_label += "\n"

    # print len(label)
    output = open(file, 'w')
    output.write(out_str)

    output = open(label_file, 'w')
    output.write( out_label )

def Test_process(testing_sentences, product):
    file = "../SemEval_2015/{}_Test_Data_noasp.txt".format( product )
    label_file = "../SemEval_2015/{}_Test_label_Data.txt".format( product )
    out_str = ""
    out_label = ""
    for sentence in testing_sentences:
        # print sentence.raw_text
        if len(sentence.opinions) == 0:
            continue
        for opinion in sentence.opinions:
            # temp = opinion.category.split("#")
            # out_str += temp[0].replace("_", " ")
            # out_str += " "
            # out_str += temp[1].replace("_", " ")
            # out_str += " "

            out_str += opinion.category
            out_str += " "
            out_str += sentence.raw_text
            out_str += "\n"
            out_label += str(opinion.polarity)
            out_label += "\n"

    # output = open(file, 'w')
    # output.write(out_str)
    #
    # output = open(label_file, 'w')
    # output.write( out_label )

def train_14():
    file = "../SemEval_2014/train14_text_ml.txt"
    label_file = "../SemEval_2014/train14_label_ml.txt"
    aspect_file = "../SemEval_2014/train14_aspect_ml.txt"
    Restaurant_train_text = "../SemEval_2014/Restaurant_Train_Data.txt"
    Restaurant_train_label = "../SemEval_2014/Restaurant_Train_label_Data.txt"
    out_str = ""
    out_label = ""
    sentence = open(file).readlines()
    aspect = open(aspect_file).readlines()
    label = open(label_file).readlines()
    for i in range(len(sentence)):
        print sentence[i].strip("\n")
        temp_aspect = aspect[i].strip("\n").split(" ")
        temp_label = label[i].strip("\n").split(" ")
        print len(temp_aspect), len(temp_label)
        print temp_aspect, temp_label
        for j in range(len(temp_aspect)):
            print j
            print temp_aspect[j]
            print temp_label[j]
            out_str += temp_aspect[j].strip()
            out_str += " "
            out_str += sentence[i].strip("\n")
            out_str += "\n"
            out_label += temp_label[j].strip("\n")
            out_label += "\n"
    output = open( Restaurant_train_text, 'w' )
    output.write( out_str )

    output = open( Restaurant_train_label, 'w' )
    output.write( out_label )


def test_14():
    file = "../SemEval_2014/test14_text_ml.txt"
    label_file = "../SemEval_2014/test14_label_ml.txt"
    aspect_file = "../SemEval_2014/test14_aspect_ml.txt"
    Restaurant_train_text = "../SemEval_2014/Restaurant_test_Data.txt"
    Restaurant_train_label = "../SemEval_2014/Restaurant_test_label_Data.txt"
    out_str = ""
    out_label = ""
    sentence = open(file).readlines()
    aspect = open(aspect_file).readlines()
    label = open(label_file).readlines()
    s_l = []
    for i in range(len(sentence)):
        print sentence[i].strip("\n")
        s_l.append(len(sentence[i].strip("\n").split(" ")))
    #     temp_aspect = aspect[i].strip("\n").split(" ")
    #     temp_label = label[i].strip("\n").split(" ")
    #     print len(temp_aspect), len(temp_label)
    #     print temp_aspect, temp_label
    #     for j in range(len(temp_aspect)):
    #         print j
    #         print temp_aspect[j]
    #         print temp_label[j]
    #         out_str += temp_aspect[j].strip()
    #         out_str += " "
    #         out_str += sentence[i].strip("\n")
    #         out_str += "\n"
    #         out_label += temp_label[j].strip("\n")
    #         out_label += "\n"
    # output = open( Restaurant_train_text, 'w' )
    # output.write( out_str )
    #
    # output = open( Restaurant_train_label, 'w' )
    # output.write( out_label )
    print np.mean(s_l)



def main(product):
    # train_14()
    test_14()

    # TRAIN_FILE = "../SemEval_2015/ABSA-15_{}_Train_Data.xml".format( product )
    # training_reviews = load_dataset( TRAIN_FILE )
    # training_sentences = unwrap( training_reviews )
    # Train_process(training_sentences, product)
    #
    #
    # TEST_FILE = "../SemEval_2015/ABSA15_{}_Test.xml".format(product)
    # FILE = TEST_FILE
    # testing_reviews = load_dataset(FILE)
    # testing_sentences = unwrap(testing_reviews)
    # Test_process( testing_sentences, product )



if __name__ == "__main__":
    main("Laptops")
    # main("Restaurants")