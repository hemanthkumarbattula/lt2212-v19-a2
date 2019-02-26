import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



def main():

    df_csv = pd.read_csv(args.vectorfile,sep="\t",index_col=0)
    df_dict = df_csv.to_dict('split')
    new_list = []
    columns = df_dict['columns']
    index = [i.split('_')[0] for i in df_dict['index']]
    data =  df_dict['data']
    for count,i in enumerate(data):
        new_list.append((index[count],i))
    class_dict={}
    for file,vector in new_list:
        if file  not in class_dict:
            class_dict[file]=[vector]
        else:
            class_dict[file]+=[vector]
    all_class_arrays=[]
    cosine_sim={}
    for clas,arr in class_dict.items():
        all_class_arrays.append((clas, np.array(arr)))
    for i, arr1 in all_class_arrays:
        for j,arr2 in all_class_arrays:
            cs = np.array(cosine_similarity(arr1,arr2))
            all_rows_cs = np.array(cs.mean(axis = 1))
            final_cosine =  all_rows_cs.mean(axis = 0)
            cosine_sim[(i,j)] = final_cosine
    print_cosine_similarities(cosine_sim)


def print_cosine_similarities(cosine_sim):

    f = open('outfile.txt', 'a')
    f.write('\tCosine Values\n')
    print('\tCosine Values\n')
    f.write('=========================================================\n')
    print('=========================================================\n')
    f.write('========%s========\n' % (args.vectorfile))
    print('========%s========\n' % (args.vectorfile))
    for key, value in cosine_sim.items():
        f.write('%s\t%s\n' % (key,value))
        print('%s\t%s\n' % (key,value))
    f.write('========'+'='*len(args.vectorfile)+'========\n' )
    print('========'+'='*len(args.vectorfile)+'========\n' )
    f.close()

if __name__ == "__main__":


    # simdoc.py -- Don't forget to put a reasonable amount code comments
    # in so that we better understand what you're doing when we grade!

    # add whatever additional imports you may need here

    parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
    parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

    args = parser.parse_args()

    print("Reading matrix from {}.".format(args.vectorfile))

    main()
