import os, sys, re, math
import glob,operator
import csv
import argparse
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here
def find_sub_directories(folder):
    d_paths = []
    classes = {}
    for paths,dirs,files in os.walk(folder):
        if paths not in d_paths:
            count = paths.count('/')
            if dirs:
                for ele in dirs:
                    abs_paths = os.path.join(paths,ele)
                    find_sub_directories(abs_paths)
                    d_paths.append(abs_paths)
                    classes.update({ele: abs_paths})
    return d_paths, classes
def find_all_files(foldername):

    directories, classes = find_sub_directories(foldername)
    abs_paths = []
    class_files={}
    for c,p in classes.items():
       for paths, dirs, files in os.walk(p):
           if files:
            for element in files:
                abs_paths.append(p+'/'+element)
       class_files.update({c:abs_paths})
    return class_files

def pre_processing(doc):
    f = open(doc, encoding="utf-8").readlines()
    words_list=[]
    for s in f:
        s = s.lower()  # Converting to lowercase
        clean = re.compile('<.*?>')
        s = re.sub(clean, ' ', s)  # Removing HTML tags
        s = re.sub(r'[?|!|\'|"|#]', r'', s)
        s = re.sub(r'[.|,|)|(|\|/]', r' ', s)  # Removing Punctuation
        s = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",s)
    
        # remove special characters and digits
        s = re.sub("(\\d|\\W)+"," ",s)
        words_list+=[word for word in s.split()]
    filtered_words = [word for word in words_list if word not in nltk.corpus.stopwords.words('english')]
    return filtered_words


def compute_doc_word_frequency(all, m=None):
    document_dict={}
    wordset = []
    final_freq_list=[]
    count=0
    all_words_set = set()
    all_words_list =[]
    for file in all:
        doc_name1 = file.split('/')[-1]
        doc_name2 =  file.split('/')[-2]
        doc_name= doc_name2 +'_' + doc_name1
        list_of_words = pre_processing(file)
        wordset.append({'file_id':count,'file_name': doc_name,'word_list': list_of_words, 'count': len(list_of_words)})
        count+=1
    if m is None:
        for i in wordset:
            all_words_set = all_words_set.union(set(i['word_list']))
            all_words_list+=i['word_list']
        all_words_dict = dict.fromkeys(all_words_set, 0)
        for i in wordset:
            new_dict= all_words_dict.copy()
            for word in i['word_list']:
                new_dict[word]+=1
            final_freq_list.append({'file_id':i['file_id'], 'file_name': i['file_name'], 'freq_dict': new_dict, 'count': i['count']})

    else:
        for i in wordset:
            freqd = dict(nltk.FreqDist(i['word_list']))
            sorted_d = sorted(freqd.items(), key=operator.itemgetter(1),reverse=True)[:m]
            temp_word_list=[key for key,value in sorted_d]
            all_words_set = all_words_set.union(set(temp_word_list))
        all_words_dict =  dict.fromkeys(all_words_set, 0)
        for i in wordset:
            new_dict= all_words_dict.copy()
            for word,value in sorted_d:
                new_dict[word]=value
            final_freq_list.append({'file_id':i['file_id'], 'file_name': i['file_name'], 'freq_dict': new_dict, 'count': i['count']})
    return all_words_dict, final_freq_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate term-document matrix.")
    parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
    parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                        default=None,
                        help="Use TruncatedSVD to truncate to N dimensions")
    parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                        type=int, default=None,
                        help="Use the top M dims from the raw counts before further processing")
    parser.add_argument("foldername", type=str,
                        help="The base folder name containing the two topic subfolders.")
    parser.add_argument("outputfile", type=str,
                        help="The name of the output file for the matrix data.")

    args = parser.parse_args()

    print("Loading data from directory {}.".format(args.foldername))
    class_files = find_all_files(args.foldername)
    all_files=[]
    for c,f in class_files.items():
        all_files+=f
    

    if not args.basedims:
        print("Using full vocabulary.")
        all_words_dict, final_freq_list = compute_doc_word_frequency(all_files)
        list_of_freq_dicts = [(i['freq_dict'],i['file_name']) for i in final_freq_list]
        from sklearn.feature_extraction import DictVectorizer
        v = DictVectorizer(sparse=False)
        freq_term_matrix = v.fit_transform([x[0] for x in list_of_freq_dicts])    
        #data=X.toarray()
        row_index= [x[1] for x in list_of_freq_dicts]
        df = pd.DataFrame( columns=v.get_feature_names())
        for count,row in enumerate(freq_term_matrix):
            df.loc[row_index[count]] = row
        df.to_csv('full_voc.csv', sep='\t')
        
    else:
        print("Using only top {} terms by raw count.".format(args.basedims))
        all_words_dict, final_freq_list=compute_doc_word_frequency(all_files,args.basedims)
        list_of_freq_dicts = [(i['freq_dict'],i['file_name']) for i in final_freq_list]
        from sklearn.feature_extraction import DictVectorizer
        v = DictVectorizer(sparse=False)
        freq_term_matrix = v.fit_transform([x[0] for x in list_of_freq_dicts])
        row_index= [x[1] for x in list_of_freq_dicts]
        print(len(row_index))
        #data=X.toarray()
        df = pd.DataFrame( columns=v.get_feature_names())
        for count,row in enumerate(freq_term_matrix):
            df.loc[row_index[count]] = row
        df.to_csv('not_full_voc.csv', sep='\t')
        

    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf = TfidfTransformer()
        tfidf_matrix = tfidf.fit_transform(freq_term_matrix)
        df = pd.DataFrame(tfidf_matrix.toarray())
        df = pd.DataFrame( columns=v.get_feature_names())

        for count,row in enumerate(tfidf_matrix.toarray()):
            df.loc[row_index[count]] = row
        df.to_csv('tfidf.csv', sep='\t')
  
        print(type(tfidf_matrix))
#        df_tfidf = pd.DataFrame( columns=tfidf_matrix.get_feature_names())
#        for count,row in enumerate(tfidf_matrix):
#            df.loc[row_index[count]] = row
#        df.to_csv('out_test1.csv', sep='\t')
       

    if args.svddims:
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
        svd = TruncatedSVD(n_components=args.svddims)
        svd1 = svd.fit_transform(freq_term_matrix)
        svd2 = svd.fit_transform(tfidf_matrix)
        df1 =  pd.DataFrame(svd1)
        df2 = pd.DataFrame(svd2)
        df1.to_csv('svd_1out_test3.csv', sep='\t')
        df2.to_csv('svd_2out_test4.csv', sep='\t')

        print(type(svd1))
        print(type(svd2))
        

    # THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
    # PARAMETERS.

    print("Writing matrix to {}.".format(args.outputfile))

