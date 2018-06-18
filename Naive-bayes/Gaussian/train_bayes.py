# -*- coding: utf-8 -*-
import math
from collections import defaultdict
import json

def listMean(l):
    return sum(l)/len(l)

def sigma(l):
    mu = listMean(l);
    sm=0
    for x in l:
        sm+=(x-mu)*(x-mu)
    return math.sqrt(sm/len(l))

class model:
    data=[]
    mulist = []
    sigmalist=[]
#    ziplist=[]
#    def makeZiplist(self):
#        self.ziplist = list(zip(*(self.data)))

def get_model(training_data):
    high=model()
    low=model()
    hx=[]
    lx=[]
    for l in training_data:
        if l[5]==1:
            lx.append(l)
        else:
            hx.append(l)
    high.data = hx
    low.data = lx
    high_ziplist = list(zip(*(high.data)))
    low_ziplist= list(zip(*(low.data)))
    high.mulist=([listMean(x) for x in high_ziplist])
    low.mulist=([listMean(x) for x in low_ziplist])
    high.sigmalist=([sigma(x) for x in high_ziplist])
    low.sigmalist=([sigma(x) for x in low_ziplist])
    return[low, high]

def gauss_prob(mu, sigma,x):
    if(sigma==0):
        return -1
    return (1 / (math.sqrt(2*math.pi) * sigma))*math.exp(-(math.pow(x-mu,2)/(2*math.pow(sigma,2))))

def get_models(training_data):
    all_models=get_model(training_data)
    return all_models

def model_to_dict(modelx):
    d = defaultdict(list)
    d["mu"]=modelx.mulist
    d["sigma"]=modelx.sigmalist
    return d

def save_model(all_models):
    cnt=0
    ret={}
    for obj in all_models:
        ret[cnt]=model_to_dict(obj)
        cnt+=1        
    with open('model.josn', 'w') as outfile:
        json.dump(ret, outfile)
    return ret

def predict(all_models, feature_list):
    probLow=1
    cnt=0
    for i in feature_list:
        probLow=probLow*gauss_prob(all_models[0].mulist[cnt],all_models[0].sigmalist[cnt],i)
        cnt+=1
        
    probHigh=1
    cnt=0
    for i in feature_list:
        probHigh=probHigh*gauss_prob(all_models[1].mulist[cnt],all_models[1].sigmalist[cnt],i)
        cnt+=1        
    
    if probHigh>probLow:
        return 3
    else:
        return 1
        
        
        
        
        
        
        
        
        
        
        
        

    
