import numpy as np
import timeit
from itertools import product
import timeit
import math
import pandas as pd

from itertools import product
import timeit
import math

unique_char = 'ACGTN'

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1
    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers

print("started")
#seqs = []
s1 = np.load("./1000000_2000000_sequencess.npy", allow_pickle=True)
print("s1 loaded")
#s2 = np.load("./1000000_2000000_sequencess.npy", allow_pickle=True)
#print("s2 loaded")
#s3 = np.load("./2000000_3000000_sequencess.npy", allow_pickle=True)
#print("s3 loaded")
#s4 = np.load("./3000000_4072342_sequencess.npy", allow_pickle=True)
#print("s4 loaded")
#varss = []
#v1 = np.load("./0_1000000_variants.npy", allow_pickle=True)
#print("v1 loaded")
#v2 = np.load("./1000000_2000000_variants.npy", allow_pickle=True)
#print("v2 loaded")
#v3 = np.load("./2000000_3000000_variants.npy", allow_pickle=True)
#print("v3 loaded")
#v4 = np.load("./3000000_4072342_variants.npy", allow_pickle=True)
#print("v4 loaded")
#for i in range(len(s1)):
#    seqs.append(s1[i])
#for i in range(len(s2)):
#    seqs.append(s2[i])
#for i in range(len(s3)):
#    seqs.append(s3[i])
#for i in range(len(s4)):
#    seqs.append(s4[i])
#for i in range(len(v1)):
#    varss.append(v1[i])
#for i in range(len(v2)):
#    varss.append(v2[i])
#for i in range(len(v3)):
#    varss.append(v3[i])
#for i in range(len(v4)):
#    varss.append(v4[i])
#print("len of seqs", len(seqs))
#print("len of varss", len(varss))
seqs = s1
print('s1[0]',s1[0])
print('seqs[0]',seqs[0])
print("shape of seqs", seqs.shape)
for i in range(len(seqs)):
    seqs[i][seqs[i]=='-'] = 'N'
    seqs[i][seqs[i]=='B'] = 'N'
    seqs[i][seqs[i]=='D'] = 'N'
    seqs[i][seqs[i]=='E'] = 'N'
    seqs[i][seqs[i]=='F'] = 'N'
    seqs[i][seqs[i]=='H'] = 'N'
    seqs[i][seqs[i]=='I'] = 'N'
    seqs[i][seqs[i]=='J'] = 'N'
    seqs[i][seqs[i]=='K'] = 'N'
    seqs[i][seqs[i]=='L'] = 'N'
    seqs[i][seqs[i]=='M'] = 'N'
    seqs[i][seqs[i]=='O'] = 'N'
    seqs[i][seqs[i]=='P'] = 'N'
    seqs[i][seqs[i]=='Q'] = 'N'
    seqs[i][seqs[i]=='R'] = 'N'
    seqs[i][seqs[i]=='S'] = 'N'
    seqs[i][seqs[i]=='U'] = 'N'
    seqs[i][seqs[i]=='V'] = 'N'
    seqs[i][seqs[i]=='W'] = 'N'
    seqs[i][seqs[i]=='X'] = 'N'
    seqs[i][seqs[i]=='Y'] = 'N'
    seqs[i][seqs[i]=='Z'] = 'N'
    seqs[i][seqs[i]=='['] = 'N'
    seqs[i][seqs[i]==']'] = 'N'


seq_data = seqs
print('seq_data',seq_data)
print('seq_data[0]',seq_data[0])
spaced_kmer_length = 3
Kmer = spaced_kmer_length
unique_seq_kmers_final_list = [''.join(c) for c in product(unique_char, repeat=spaced_kmer_length)]


start = timeit.default_timer()
final_feature_vector = []

for seq_ind in range(len(seq_data)):
    print("index: ",seq_ind,"/",len(seq_data))
    se_temp = seq_data[seq_ind]
 #   print('se_temp',se_temp)
    k_mers_final = build_kmers(se_temp,spaced_kmer_length)
  #  print('k_mers_final',k_mers_final)
    character_val = len(unique_char)
    pwm_matrix = np.array([[0]*Kmer]*(character_val))
    count_lines = 0 # Initialize the total number of sequences to 0
    # Read line by line, stripping the end of line character and
    # updating the PWM with the frequencies of each base at the 9 positions
    for ii in range(len(k_mers_final)):
        line = k_mers_final[ii]
   #     print(line)
        count_lines += 1 # Keep counting the sequences
        for i in range(len(line)):
            if line[i]=='9':
                pwm_matrix[len(pwm_matrix)-1,i] = pwm_matrix[len(pwm_matrix)-1,i] + 1
            else:
    #            print(line[i])
                ind_tmp = unique_char.index(line[i])
                pwm_matrix[ind_tmp,i] = pwm_matrix[ind_tmp,i] + 1
    LaPlace_pseudocount = 0.1
    equal_prob_nucleotide = character_val/100
    for i in range(len(k_mers_final[0])):
        for x in range(len(pwm_matrix)):
            pwm_matrix[x,i] = round(math.log((pwm_matrix[x,i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    ################ Generate PWM (End) #########################

    ################ Assign Individual k-mers Score (Start) #########################
    each_k_mer_score = []
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for ii in range(len(k_mers_final)):
        line = k_mers_final[ii]
        score = 0
        for i in range(len(line)):
            if line[i]=='9':
                score += pwm_matrix[len(pwm_matrix)-1,i]
            else:
                ind_tmp = unique_char.index(line[i])
                score += pwm_matrix[ind_tmp,i]
        final_score_tmp = round(score, 3)
        each_k_mer_score.append(final_score_tmp)
        ###################### assign weughted k-mers frequency score ###############
        kmer_val_check = str(line)
        aa_lst_1 = kmer_val_check.replace(",","")
        aa_lst_2 = aa_lst_1.replace("[","")
        aa_lst_3 = aa_lst_2.replace("\"","")
        aa_lst_4 = aa_lst_3.replace("]","")
        aa_lst_5 = aa_lst_4.replace("'","")
        aa_lst_6 = aa_lst_5.replace(" ","")

        ind_tmp = unique_seq_kmers_final_list.index(aa_lst_6)
        listofzeros[ind_tmp] = listofzeros[ind_tmp] + (1 * final_score_tmp)

    final_feature_vector.append(listofzeros)
    ################ Assign Individual k-mers Score (end) #########################

stop = timeit.default_timer()
print("PWM Time : ", stop - start)

max_vec_length = 0

for t in range(len(final_feature_vector)):
    if len(final_feature_vector[t]) > max_vec_length:
        max_vec_length = len(final_feature_vector[t])

padded_pwm_vec = []

for t in range(len(final_feature_vector)):
    row_vec = final_feature_vector[t]
    if(len(row_vec)<max_vec_length):
        for k in range(len(row_vec),max_vec_length):
            row_vec.append(0)
    padded_pwm_vec.append(row_vec)
np.save("./1000000_2000000_PWM2Vec_embeddings_FullGenome_2582584.npy",padded_pwm_vec)


print("All Processing Done!!!")




