from os import listdir
import numpy

def get_dir_files_arr(dirpath):
    dircontents = listdir(dirpath)
    dircontents.sort()
    return dircontents

def extract_features(filedata, wordcount_dict):
    tokens = filedata.split()
    for tok in tokens:
        if tok not in wordcount_dict:
            wordcount_dict[tok]=1;
        else:
            wordcount_dict[tok] = wordcount_dict[tok] + 1

def build_dir_word_dict(base_path, file_name_array, input_dict):
    for f in file_name_array:
        print("opening : " + f)
        filePath = base_path + "/" + f
        fHandler = open(filePath,'r')
        extract_features(fHandler.read(),input_dict)
        fHandler.close()


def interleave_pos_neg_files(neg_basepath, neg_file_arr, pos_basepath, pos_file_arr, combined_array, sentiment_flag):
    for i in range(len(pos_file_arr) + len(neg_file_arr)):
        if(len(pos_file_arr)==0 and len(neg_file_arr)==0):
            break
        elif(len(pos_file_arr)>0 and len(neg_file_arr)==0):
            while len(pos_file_arr) > 0 :
                pos_tuple = (pos_basepath + "/" + pos_file_arr.pop(),1)
                combined_array.append(pos_tuple)
            break
        elif(len(pos_file_arr)==0 and len(neg_file_arr)>0):
            while len(neg_file_arr) >0:
                neg_tuple = (neg_basepath + "/" + neg_file_arr.pop(),-1)
                combined_array.append(neg_tuple)
            break
        else:
            if(sentiment_flag == -1):
                neg_tuple = (neg_basepath + "/" + neg_file_arr.pop(),sentiment_flag)
                combined_array.append(neg_tuple)
                sentiment_flag = 1
                continue
            elif(sentiment_flag == 1):
                pos_tuple = (pos_basepath + "/" + pos_file_arr.pop(),sentiment_flag)
                combined_array.append(pos_tuple)
                sentiment_flag = -1
                continue

neg_files_basepath = "./data/review_polarity/txt_sentoken/neg"
pos_files_basepath = "./data/review_polarity/txt_sentoken/pos"
negative_files = get_dir_files_arr(neg_files_basepath)
positive_files = get_dir_files_arr(pos_files_basepath)

all_words = {}
build_dir_word_dict(neg_files_basepath,negative_files,all_words)
build_dir_word_dict(pos_files_basepath,positive_files,all_words)
