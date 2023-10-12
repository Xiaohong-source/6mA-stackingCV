import pandas as pd
import numpy as np



def read_seq(filepath):
    seq = []
    lst1 = []
    for line in open(filepath, 'r', encoding='utf-8'):
        line = line.splitlines()  # 去掉换行符
        lst1.append(line)
    for i in range(len(lst1)):  # 遍历lines
        if i % 2 != 0:  # 把双数行存到test里面，单数行存到train里面
            seq.append(*lst1[i])
    return seq

def Onehot(seq):
    result = []
    for s in seq:
        lst = []
        for i in range(len(s)):
            if s[i] == 'A':
                lst.extend([1, 0, 0, 0])
            if s[i] == 'G':
                lst.extend([0, 0, 1, 0])
            if s[i] == 'C':
                lst.extend([0, 1, 0, 0])
            if s[i] == 'T':
                lst.extend([0, 0, 0, 1])
            if s[i] == 'N':
                lst.extend([0, 0, 0, 0])
        result.append(lst)
    return np.array(result)


def EIIP(seq):
    result = []
    for s in seq:
        lst = []
        for i in range(len(s)):
            if s[i] == 'A':
                lst.extend([0.1260])
            if s[i] == 'C':
                lst.extend([0.1340])
            if s[i] == 'G':
                lst.extend([0.0806])
            if s[i] == 'T':
                lst.extend([0.1335])
            if s[i] == 'N':
                lst.extend([0])
        result.append(lst)
    return np.array(result)

def NCP(seq):
    result = []

    for s in seq:
        lst = []
        for i in range(len(s)):
            if s[i] == 'A':
                lst.extend([1, 1, 1])
            if s[i] == 'C':
                lst.extend([0, 1, 0])
            if s[i] == 'G':
                lst.extend([1, 0, 0])
            if s[i] == 'T':
                lst.extend([0, 0, 1])
            if s[i] == 'N':
                lst.extend([0, 0, 0])
        result.append(lst)
    return np.array(result)
def ENAC(seq,k):
    result = []
    for s in seq:
        ENAC = []
        k_frequency = []
        for i in range(len(s)-k+1):
            a = s[i:i+k]
            A_sum = a.count('A')
            A_fre = A_sum/k
            k_frequency.append(A_fre)
            C_sum = a.count('C')
            C_fre = C_sum/k
            k_frequency.append(C_fre)
            G_sum = a.count('G')
            G_fre = G_sum/k
            k_frequency.append(G_fre)
            T_sum = a.count('T')
            T_fre = T_sum/k
            k_frequency.append(T_fre)
        ENAC.extend(k_frequency)
        result.append(ENAC)
    return np.array(result)

def Onehot_EIIP(pos_filepath,neg_filepath):
    pos_seq = read_seq(pos_filepath)
    neg_seq = read_seq(neg_filepath)

    pos_feature = np.concatenate([Onehot(pos_seq),EIIP(pos_seq)], axis=1)
    neg_feature = np.concatenate([Onehot(neg_seq), EIIP(neg_seq)], axis=1)

    lable = np.concatenate([np.ones((len(pos_seq),)), np.zeros((len(neg_seq),))], axis=0)  # 竖向拼接
    return  np.concatenate([pos_feature, neg_feature], axis=0), lable

def Onehot_NCP(pos_filepath,neg_filepath):
    pos_seq = read_seq(pos_filepath)
    neg_seq = read_seq(neg_filepath)
    pos_feature = np.concatenate([Onehot(pos_seq),NCP(pos_seq)], axis=1)
    neg_feature = np.concatenate([Onehot(neg_seq), NCP(neg_seq)], axis=1)
    lable = np.concatenate([np.ones((len(pos_seq),)), np.zeros((len(neg_seq),))], axis=0)  # 竖向拼接
    return  np.concatenate([pos_feature, neg_feature], axis=0),lable


def Onehot_EIIP_NCP(pos_filepath,neg_filepath):
    pos_seq = read_seq(pos_filepath)
    neg_seq = read_seq(neg_filepath)
    pos_feature = np.concatenate([Onehot(pos_seq),EIIP(pos_seq) ,NCP(pos_seq)], axis=1)
    neg_feature = np.concatenate([Onehot(neg_seq),EIIP(neg_seq) , NCP(neg_seq)], axis=1)

    lable = np.concatenate([np.ones((len(pos_seq),)), np.zeros((len(neg_seq),))], axis=0)  # 竖向拼接
    return  np.concatenate([pos_feature, neg_feature], axis=0),lable


def Onehot_ENAC(pos_filepath,neg_filepath):
    pos_seq = read_seq(pos_filepath)
    neg_seq = read_seq(neg_filepath)

    pos_feature = np.concatenate([Onehot(pos_seq), ENAC(pos_seq,5)], axis=1)
    neg_feature = np.concatenate([Onehot(neg_seq), ENAC(neg_seq,5)], axis=1)

    lable = np.concatenate([np.ones((len(pos_seq),)), np.zeros((len(neg_seq),))], axis=0)  # 竖向拼接
    return  np.concatenate([pos_feature, neg_feature], axis=0),lable






