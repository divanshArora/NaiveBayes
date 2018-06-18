# -*- coding: utf-8 -*-
import train_bayes
import main
import matplotlib.pyplot as plt

def result(train_fraction, test_fraction):
    all_data = main.get_data(train_fraction, test_fraction)# 0 = train 1 = test
    testing_data=all_data[1]
    all_models = train_bayes.get_models(all_data[0])
    tp=0
    fp=0
    tn=0
    fn=0
    #p = high
    #n = low
    
    for i in testing_data:
        if train_bayes.predict(all_models,i[0:len(i)-1])==i[-1]:
            if i[-1]==3:
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]==3:
                fn+=1
            else:
                fp+=1
#    tpr = high_correct/(high_correct+high_wrong)
#    far = 
    confusion_matrix = [[tn, fp],[fn, tp]]
    return confusion_matrix

def result1(all_models, testing_data):
#    all_data = main.get_data(train_fraction, test_fraction)# 0 = train 1 = test
#    testing_data=all_data[1]
#    all_models = train_bayes.get_models(all_data[0])
    tp=0
    fp=0
    tn=0
    fn=0
    #p = high
    #n = low
    
    for i in testing_data:
        if train_bayes.predict(all_models,i[0:len(i)-1])==i[-1]:
            if i[-1]==3:
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]==3:
                fn+=1
            else:
                fp+=1
#    tpr = high_correct/(high_correct+high_wrong)
#    far = 
    confusion_matrix = [[tn, fp],[fn, tp]]
    return confusion_matrix


def print_ac(confusion_mat):
    tp=confusion_mat[1][1]
    fp=confusion_mat[0][1]
    tn=confusion_mat[0][0]
    fn=confusion_mat[1][0]
    
#    print(fn,tn,tp,fp)
    accuracy =  ((tp+tn)/(tp+fp+tn+fn))*100
    tpr = tp/(tp+fn)
    far=  fp/(fp+tn)
    print("tpr = ",str(tpr)," far = ",str(far))
    
    confusion_matrix = [[tn, fp],[fn, tp]]
    print("Accuracy: \n"+str(accuracy)+"%")
    print("Confusion matrix:\n"+str(confusion_matrix[0]),"\n",str(confusion_matrix[1]))
    

def plot_hundred_runs():
    tprList = []
    farList=[]
    ac_list =[]
    ml=[]
    x=[]
    for i in range(100):
        x.append(i)
        cm=result(0.7,0.3)
        tp=cm[1][1]
        fp=cm[0][1]
        tn=cm[0][0]
        fn=cm[1][0]
        ac_list.append(((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))*100)
#        tprList.append(tp/(tp+fn))
#        farList.append(fp/(fp+tn))
        ml.append( (fp/(fp+tn), tp/(tp+fn)    ))
    ml.sort()
    for i in range(100):
        farList.append(ml[i][0])
        tprList.append(ml[i][1])
    plt.figure(1)
    plt.plot(ac_list)
    plt.ylabel('Accuracies on random runs')
    plt.title("100 Run run vs accuracy")
    plt.xlabel('Run number')
    plt.show()
    
    plt.figure(2)
    plt.plot(farList,tprList)
    plt.ylabel('TPR')
    plt.title("100 Run FAR vs TPR")
    plt.xlabel('FAR')
    plt.show()
    
    
def calc_accuracy(cm):
    return ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))*100
    
def kFold():
    accuracy_list=[]
    aux_list = (main.get_shuffle_data()) # 0 = train 1 = test
    for i in range(0,len(aux_list),int(len(aux_list)/5)):
        if i+ int(len(aux_list)/5) <=len(aux_list):
            part = main.partition_data(aux_list,i, i+ int(len(aux_list)/5))
            all_models = train_bayes.get_models(part[0])
            cm = result1(all_models,part[1])
            accuracy_list.append(((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))*100)
    normal_acc = calc_accuracy(result(0.7,0.3))
    return [accuracy_list,normal_acc]    
    

def compare_thirty_fifty():
    thirty=[]
    fifty=[]
    t=[]
    for i in range(30):
        aux_list = (main.get_shuffle_data()) # 0 = train 1 = test
        part = main.partition_data(aux_list,0, int(len(aux_list)*0.3))
        all_models = train_bayes.get_models(part[0])
        cm1=result1(all_models,part[1])
        part = main.partition_data(aux_list,0, int(len(aux_list)*0.5))
        all_models = train_bayes.get_models(part[0])
        cm2=result1(all_models,part[1])
        thirty.append(calc_accuracy(cm1))
        fifty.append(calc_accuracy(cm2))
        t.append(i)
    plt.figure(3)
    plt.plot(t, thirty, label='70-30 split')
    plt.plot(t, fifty,label='50-50 split')
    plt.ylabel('Accuracy')
    plt.title("30 Run accuracy 70-30 and 50-50 split")
    plt.xlabel('Run number')
    plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
    plt.show()

        
    
if __name__=="__main__":

    s="""
                low(predicted)  high(predicted)
            
    low(actual)      tn             fp
        
    high(actual)     fn             tp
    """
    print(s)
    print("Part A. Simple Test 70% training results:")
    
    aux_list = (main.get_shuffle_data()) # 0 = train 1 = test
    part = main.partition_data(aux_list,0, int(len(aux_list)*0.3))
    all_models = train_bayes.get_models(part[0])
    train_bayes.save_model(all_models)
    cm1=result1(all_models,part[1])
    print_ac(cm1)
    
    
    print("Accuracy graph for 100 runs")
    plot_hundred_runs()
    compare_thirty_fifty()
    kfold_list= kFold()
    print("Mean 5 fold = ",str(train_bayes.listMean(kfold_list[0])))
    print("Std dev 5 fold = ",str(train_bayes.sigma(kfold_list[0])))
    print("70-30 accuracy = ",str(kfold_list[1]))
#    print(kFold())
#    result(0.7,0.3)
#    result(0.5,0.5)





















