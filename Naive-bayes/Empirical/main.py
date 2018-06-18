# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#author: Divansh Arora 2015027
import csv
import random




def read_data(pathToFile):
    full_dataset = []
    with open(pathToFile, 'r') as data:
        text = csv.reader(data)
        for line in text:
            full_dataset.append([int(i) for i in line])
    return full_dataset



#training_precent = 0.7
#testing_percent = 0.3

def get_data(training_precent, testing_percent):
    dataset = read_data("/home/divansh/Desktop/Coursework/Sem6/PR_SML/Assignment1/tae.data.txt")
    aux_data=[]
    for d in dataset:
        if(d[5]!=2):
            aux_data.append(d)
    random.shuffle(aux_data)
    train_data = aux_data[0:int(len(aux_data)*training_precent)]
    test_data = aux_data[int(len(aux_data)*training_precent) : int(len(aux_data))]
    return (train_data, test_data)

def get_shuffle_data():
    dataset = read_data("/home/divansh/Desktop/Coursework/Sem6/PR_SML/Assignment1/tae.data.txt")
    aux_data=[]
    for d in dataset:
        if(d[5]!=2):
            aux_data.append(d)
    random.shuffle(aux_data)
    return aux_data

def partition_data(aux_data,l,r):
    test_data = aux_data[l:r]
    train_data = aux_data[0 : l] + aux_data[r : int(len(aux_data))]
#    print("train data len = "+str(len(train_data)))
#    print("test data len = "+str(len(test_data)))
    
    return (train_data, test_data)
    
    
    
    