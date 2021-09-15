from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
#import seaborn as sns
import os
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from numpy import savetxt


maxlen = 1500
pad = "no"

GRCh38_latest_rna = open("/projects/btl_scratch/datasets/RefSeq_transcriptomes/homo_sapiens/GRCh38_latest_rna.fna")
GRCh38_array = GRCh38_latest_rna.readlines()

current_ind = 0
current_transcript = ""
GRCh38_array.append("> \/n")
length_vals = []
txt = []
while current_ind < len(GRCh38_array):
    current_line = GRCh38_array[current_ind]
    current_line = current_line[0:len(current_line)-1]
    if current_line.startswith(">"):
       txt.append(current_transcript)
       txt.append(current_line)
       if current_ind>1:
           length_vals.append(len(current_transcript))
       current_transcript = ""
    else:
        current_transcript += current_line[0:len(current_line)-1]
    current_ind += 1

#%%
txt_final = txt[1:len(txt)-1]
label_arr = []
print(len(txt_final))
for i in range(0,len(txt_final),2):
    txt_split = txt_final[i].split(';')
    trans_features = txt_split[0].split(',')
    label_arr.append(trans_features[len(trans_features)-1])



#%%
Counter(label_arr)

#%%%

print("hi")
percentile_list = pd.DataFrame(
    {'label': label_arr,
     'length': length_vals
     })
#print(percentile_list)
percentile_list.drop(percentile_list[percentile_list.label==" small nuclear RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" telomerase RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" Y RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" RNase P RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" antisense RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" guide RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" small cytoplasmic RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" RNase MRP RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" ribosomal RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" vault RNA"].index, inplace=True)
percentile_list.drop(percentile_list[percentile_list.label==" partial mRNA"].index, inplace=True)

print(percentile_list)
#print(type(percentile_list))


### sample 10000 mRNA and 10000 ncRNA
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['A'], ['T'], ['C'], ['G']]
enc.fit(X)


mRNA_indices = [i for i in range(len(label_arr)) if label_arr[i] == " mRNA" and length_vals[i] <= maxlen]
ncRNA_indices = [i for i in range(len(label_arr)) if label_arr[i] == " ncRNA" and length_vals[i] <= maxlen]

mRNA_data = []
mRNA_data_text = []
for i in range(0,len(mRNA_indices)):
    current = np.array(list(txt_final[mRNA_indices[i]*2+1])).reshape(-1,1)
    current = np.where(current == 'A', 1, current)
    current = np.where(current == 'T', 2, current)
    current = np.where(current == 'C', 3, current)
    current = np.where(current == 'G', 4, current)
    current = np.where(current == 'N', 5, current)
    current = np.where(current == 'W', 5, current)
    current = current.astype('int')
    mRNA_data.append(current)
    mRNA_data_text.append(txt_final[mRNA_indices[i]*2+1]);

#################
ncRNA_data = []
ncRNA_data_text = []
for i in range(0,len(ncRNA_indices)):
    current = np.array(list(txt_final[ncRNA_indices[i]*2+1])).reshape(-1,1)
    current = np.where(current == 'A', 1, current)
    current = np.where(current == 'T', 2, current)
    current = np.where(current == 'C', 3, current)
    current = np.where(current == 'G', 4, current)
    current = np.where(current == 'N', 5, current)
    current = np.where(current == 'W', 5, current)
    current = current.astype('int')
    ncRNA_data.append(current)
    ncRNA_data_text.append(txt_final[ncRNA_indices[i]*2+1])
#################

mRNA_data_arr = np.array(mRNA_data, dtype=object)
ncRNA_data_arr = np.array(ncRNA_data, dtype=object)

dataset_sequence_final = np.concatenate((mRNA_data_arr, ncRNA_data_arr), 0)
dataset_labels_final = np.vstack((np.full((len(mRNA_data),1), 1), np.full((len(ncRNA_data),1), 0)))


X_train, X_test, y_train, y_test = train_test_split(dataset_sequence_final,dataset_labels_final,test_size=0.2, random_state = 42)

np.save("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/data/X_train_notruncation_" + pad + "pad_maxlen" + str(maxlen), X_train)
np.save("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/data/X_test_notruncation_" + pad + "pad_maxlen" + str(maxlen), X_test)

savetxt("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/data/y_train_notruncation_" + pad + "pad_maxlen" + str(maxlen), y_train)
savetxt("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/data/y_test_notruncation_" + pad + "pad_maxlen" + str(maxlen), y_test)


