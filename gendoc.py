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


def find_sub_directories(folder):
    '''

    :param folder: take input as folder name
    :return:
          returns directory paths and sub directory absolute paths
    '''
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
    '''

    :param foldername:
            takes input as folder name
    :return:
         returns dictionary
         all sub directories names present in the directory mapped to
        list of files in the sub directories
    '''

    directories, classes = find_sub_directories(foldername)
    class_files={}
    for c,p in classes.items():
       for paths, dirs, files in os.walk(p):
           abs_paths = []
           if files:
            for element in files:
                abs_paths.append(p+'/'+element)
       class_files.update({c:abs_paths})
    return class_files

def pre_processing(docs):
    '''

    :param docs:
          Take absolute paths of file names list
    :return:
        returns list of strings with each element as the text in each document after
        applying the processing
    '''
    all_docs = []
    for file in docs:
        f = open(file, encoding="utf-8").readlines()
        docname = file.split('/')[-2]+'_'+file.split('/')[-1]
        words_list=[]
        for s in f:
            s = s.lower()  # Converting to lowercase
            clean = re.compile('<.*?>')
            s = re.sub(clean, ' ', s)  # Removing HTML tags
            s = re.sub(r'[?|!|\'|"|#]', r'', s)
            s = re.sub(r'[.|,|)|(|\|/]', r' ', s)  # Removing Punctuation
            s = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", s)
            # remove special characters and digits
            s = re.sub("(\\d|\\W)+"," ",s)
            words_list += [word for word in s.split()]
        filtered_words = [word for word in words_list if word not in nltk.corpus.stopwords.words('english')]
        all_docs.append((docname, ' '.join(filtered_words)))
    return all_docs




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
    file_contents = pre_processing(all_files)

    #filenames and their contents separated

    filenames = [x[0] for x in file_contents]
    documents = [x[1] for x in file_contents]
    if not args.basedims:
        print("Using full vocabulary.")
        vectorizer = CountVectorizer(max_features=None)
        count_vector = vectorizer.fit_transform(documents).toarray()
        count_vector_df =  pd.DataFrame(data=count_vector,    
                                        index= filenames,    
                                        columns= vectorizer.get_feature_names())
        print("Writing matrix to {}.".format(args.outputfile))
        count_vector_df.to_csv('cv_'+args.outputfile+'.csv', sep='\t', encoding='utf-8')
        
    else:
        print("Using only top {} terms by raw count.".format(args.basedims))
        #use 'max_features' to use the most common words in total vocabulary
        vectorizer = CountVectorizer(max_features = args.basedims)
        count_vector = vectorizer.fit_transform(documents).toarray()
        count_vector_df =  pd.DataFrame(data=count_vector,
                                        index= filenames,
                                        columns= vectorizer.get_feature_names())
        print("Writing matrix to {}.".format(args.outputfile))
        count_vector_df.to_csv('cv_'+args.outputfile+'.csv', sep='\t', encoding='utf-8')

    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        tfidfconverter = TfidfVectorizer(max_features= None)
        if args.basedims:
            tfidfconverter = TfidfVectorizer(max_features= args.basedims)  
        tfidf_vector = tfidfconverter.fit_transform(documents).toarray()
        tfidf_vector_df =  pd.DataFrame(data= tfidf_vector,
                                        index= filenames,
                                        columns= tfidfconverter.get_feature_names())  
        tfidf_vector_df.to_csv('tfidf_'+args.outputfile+'.csv', sep='\t', encoding='utf-8')

    if args.svddims:
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
        if args.svddims < len(vectorizer.get_feature_names()):
            svdT = TruncatedSVD(n_components=args.svddims)
            svdT_vector = svdT.fit_transform(count_vector)
            svdT_vector_df =  pd.DataFrame(data = svdT_vector,
                                           index = filenames,
                                           columns= [i for i in range(0,args.svddims)])
            svdT_vector_df.to_csv('svdT_countvector'+args.outputfile+'.csv', sep='\t', encoding='utf-8')
        else:
            print('Cannot truncate array as the svd dimensions greater than array columns')
        if args.tfidf:
            if args.svddims < len(tfidfconverter.get_feature_names()):
                svdT = TruncatedSVD(n_components = args.svddims)
                svdT_vector = svdT.fit_transform(tfidf_vector)
                svdT_vector_df =  pd.DataFrame(data = svdT_vector,
                                               index = filenames,
                                               columns =[i for i in  range(0,args.svddims)])
                svdT_vector_df.to_csv('svdT_tfidf'+args.outputfile+'.csv', sep='\t', encoding='utf-8')
 
            else:
                 print('Cannot truncate array as the svd dimensions greater than array columns')
              
        


