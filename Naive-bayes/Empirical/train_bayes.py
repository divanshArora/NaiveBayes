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
    

    ziplist=[]
#    def makeZiplist(self):
#        self.ziplist = list(zip(*(self.data)))


def getEmpiricalProb(ziplist, feature_id, feature_val ):
    ans=0
#    l = len(ziplist[feature_id])
    for i in ziplist[feature_id]:
        if(i==feature_val):
            ans+=1
    if(ans==0):
        return 1
    return ans
    
    
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
    high.ziplist = high_ziplist
    low.ziplist=low_ziplist
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
    d["ziplist"]=modelx.ziplist
    return d

def save_model(all_models):
    cnt=0
    ret={}
    for obj in all_models:
        ret[cnt]=model_to_dict(obj)
        cnt+=1        
    with open('model.json', 'w') as outfile:
        json.dump(ret, outfile)
    return ret

def predict(all_models, feature_list):
    probLow=1
    cnt=0
    for i in feature_list:
        p = getEmpiricalProb(all_models[0].ziplist,cnt, i)
        if(p!=-1):
            probLow=probLow*p
        cnt+=1
        
    probHigh=1
    cnt=0
    for i in feature_list:
        p = getEmpiricalProb(all_models[1].ziplist,cnt, i)
        if(p!=-1):
            probHigh=probHigh*p
        cnt+=1
    
    if probHigh>probLow:
        return 3
    else:
        return 1
        
def predict_roc(all_models, feature_list, delta):
    d = delta        
    probHigh=1
    probLow=1
    cnt=0
    for i in feature_list:
        p = getEmpiricalProb(all_models[0].ziplist,cnt, i)
        p1= p
        probLow=probLow*p
        p = getEmpiricalProb(all_models[1].ziplist,cnt, i)
        probHigh=probHigh*p
        probHigh= (2*probHigh)/(p1+p)
        probLow= (2*probLow)/(p1+p)
        cnt+=1
        d*=delta;
#        print("pppsum = ",probHigh+probLow," pl = ",probLow," -- ", probHigh )
    probLow=probLow/2
    probHigh=probHigh/2
#    print("psum = ",probHigh+probLow," pl = ",probLow," -- ", probHigh )
    
    if probHigh>delta:
 #        print("=== 3 ===")
        return 3
    else:
  #      print("===1===")
        return 1
        
        
        
        
        
        
        
        
        
        
        

    
