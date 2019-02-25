import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))
df_csv = pd.read_csv(args.vectorfile,sep="\t",index_col=0)
df_dict = df_csv.to_dict('split')
new_list = []
columns = df_dict['columns']
index = [i.split('_')[0] for i in df_dict['index']]
data =  df_dict['data']
for count,i in enumerate(data):
    new_list.append((index[count],i))
class_dict={}
for i,j in new_list:
    if i not in class_dict:
        class_dict[i]=[j]
    else:
        class_dict[i]+=j
for i in class_dict.keys():
    print(i)  

for i,j in class_dict.items():
    for l1 in j:
        for l2 in j:
            print(l1)
            print(l2)
            print(type(l1),len(l1))
            print(type(l2),len(l2))
