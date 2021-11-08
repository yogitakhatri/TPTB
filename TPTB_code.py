#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading datasets

import numpy as np
import pandas as pd
from scipy.io import arff

df1 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.3.csv')
df2 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.4.csv')
df3 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.5.csv')
df4 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.6.csv')
df5 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.7.csv')
df6 = pd.read_csv(r'D:\datasets\JURECZKO\arc\arc.csv')
df7=pd.read_csv(r'D:\datasets\JURECZKO\berek\berek.csv')
df8=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.0.csv')
df9=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.2.csv')
df10=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.4.csv')
df11=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.6.csv')
df12=pd.read_csv(r'D:\datasets\JURECZKO\ckjm\ckjm.csv')
df13=pd.read_csv(r'D:\datasets\JURECZKO\elearning\e-learning.csv')
df14=pd.read_csv(r'D:\datasets\JURECZKO\forrest\forrest-0.7.csv')
df15=pd.read_csv(r'D:\datasets\JURECZKO\forrest\forrest-0.8.csv')
df16=pd.read_csv(r'D:\datasets\JURECZKO\intercafe\intercafe.csv')
df17=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-1.1.csv')
df18=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-1.4.csv')
df19=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-2.0.csv')
df20=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-3.2.csv')
df21=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.0.csv')
df22=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.1.csv')
df23=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.2.csv')
df24=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.3.csv')
df25=pd.read_csv(r'D:\datasets\JURECZKO\kalkulator\kalkulator.csv')
df26=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.0.csv')
df27=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.1.csv')
df28=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.2.csv')
df29=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.0.csv')
df30=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.2.csv')
df31=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.4.csv')
df32=pd.read_csv(r'D:\datasets\JURECZKO\nieruchomosci\nieruchomosci.csv')
df33=pd.read_csv(r'D:\datasets\JURECZKO\pbeans\pbeans1.csv')
df34=pd.read_csv(r'D:\datasets\JURECZKO\pbeans\pbeans2.csv')
df35=pd.read_csv(r'D:\datasets\JURECZKO\pdftranslator\pdftranslator.csv')
df36=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-1.5.csv')
df37=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-2.0.csv')
df38=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-2.5.csv')
df39=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-3.0.csv')
df40=pd.read_csv(r'D:\datasets\JURECZKO\redaktor\redaktor.csv')
df41=pd.read_csv(r'D:\datasets\JURECZKO\serapion\serapion.csv')
df42=pd.read_csv(r'D:\datasets\JURECZKO\skarbonka\skarbonka.csv')
df43=pd.read_csv(r'D:\datasets\JURECZKO\sklebagd\sklebagd.csv')
df44=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.0.csv')
df45=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.1.csv')
df46=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.2.csv')
df47=pd.read_csv(r'D:\datasets\JURECZKO\systemdata\systemdata.csv')
df48=pd.read_csv(r'D:\datasets\JURECZKO\szybkafucha\szybkafucha.csv')
df49=pd.read_csv(r'D:\datasets\JURECZKO\termoproject\termoproject.csv')
df50=pd.read_csv(r'D:\datasets\JURECZKO\tomcat\tomcat.csv')
df51=pd.read_csv(r'D:\datasets\JURECZKO\velocity\velocity-1.5.csv')
df52=pd.read_csv(r'D:\datasets\JURECZKO\velocity\velocity-1.6.csv')
df53=pd.read_csv(r'D:\datasets\JURECZKO\workflow\workflow.csv')
df54=pd.read_csv(r'D:\datasets\JURECZKO\wspomaganiepi\wspomaganiepi.csv')
df55=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.4.csv')
df56=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.5.csv')
df57=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.6.csv')
df58=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-init.csv')
df59=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.2.csv')
df60=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.3.csv')
df61=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.4.csv')
df62=pd.read_csv(r'D:\datasets\JURECZKO\zuzel\zuzel.csv')





# In[2]:


#  filtering the features and the defect count from each dataset as first three features are representing informations related to  name and version

df1=df1.iloc[:,3:24]
df2=df2.iloc[:,3:24]
df3=df3.iloc[:,3:24]
df4=df4.iloc[:,3:24]
df5=df5.iloc[:,3:24]
df6=df6.iloc[:,3:24]
df7=df7.iloc[:,3:24]
df8=df8.iloc[:,3:24]
df9=df9.iloc[:,3:24]
df10=df10.iloc[:,3:24]
df11=df11.iloc[:,3:24]
df12=df12.iloc[:,3:24]
df13=df13.iloc[:,3:24]
df14=df14.iloc[:,3:24]
df15=df15.iloc[:,3:24]
df16=df16.iloc[:,3:24]
df17=df17.iloc[:,3:24]
df18=df18.iloc[:,3:24]
df19=df19.iloc[:,3:24]
df20=df20.iloc[:,3:24]
df21=df21.iloc[:,3:24]
df22=df22.iloc[:,3:24]
df23=df23.iloc[:,3:24]
df24=df24.iloc[:,3:24]
df25=df25.iloc[:,3:24]
df26=df26.iloc[:,3:24]
df27=df27.iloc[:,3:24]
df28=df28.iloc[:,3:24]
df29=df29.iloc[:,3:24]
df30=df30.iloc[:,3:24]
df31=df31.iloc[:,3:24]
df32=df32.iloc[:,3:24]
df33=df33.iloc[:,3:24]
df34=df34.iloc[:,3:24]
df35=df35.iloc[:,3:24]
df36=df36.iloc[:,3:24]
df37=df37.iloc[:,3:24]
df38=df38.iloc[:,3:24]
df39=df39.iloc[:,3:24]
df40=df40.iloc[:,3:24]
df41=df41.iloc[:,3:24]
df42=df42.iloc[:,3:24]
df43=df43.iloc[:,3:24]
df44=df44.iloc[:,3:24]
df45=df45.iloc[:,3:24]
df46=df46.iloc[:,3:24]
df47=df47.iloc[:,3:24]
df48=df48.iloc[:,3:24]
df49=df49.iloc[:,3:24]
df50=df50.iloc[:,3:24]
df51=df51.iloc[:,3:24]
df52=df52.iloc[:,3:24]
df53=df53.iloc[:,3:24]
df54=df54.iloc[:,3:24]
df55=df55.iloc[:,3:24]
df56=df56.iloc[:,3:24]
df57=df57.iloc[:,3:24]
df58=df58.iloc[:,3:24]
df59=df59.iloc[:,3:24]
df60=df60.iloc[:,3:24]
df61=df61.iloc[:,3:24]
df62=df62.iloc[:,3:24]



# In[3]:



# labeling the modules as faulty ( with integer value "1") if their defect count value is greater than 0, otherwise as  non-faulty with integer value "0"

df1['bug']=(df1["bug"]> 0).astype(int)
df2['bug'] = (df2['bug'] > 0).astype(int)
df3['bug']=(df3["bug"]> 0).astype(int)
df4['bug'] = (df4['bug'] > 0).astype(int)
df5['bug'] = (df5['bug'] > 0).astype(int)
df6['bug'] = (df6['bug'] > 0).astype(int)
df7['bug'] = (df7['bug'] > 0).astype(int)
df8['bug'] = (df8['bug'] > 0).astype(int)
df9['bug']=(df9["bug"]> 0).astype(int)
df10['bug'] = (df10['bug'] > 0).astype(int)
df11['bug']=(df11["bug"]> 0).astype(int)
df12['bug'] = (df12['bug'] > 0).astype(int)
df13['bug'] = (df13['bug'] > 0).astype(int)
df14['bug'] = (df14['bug'] > 0).astype(int)
df15['bug'] = (df15['bug'] > 0).astype(int)
df16['bug'] = (df16['bug'] > 0).astype(int)
df17['bug']=(df17["bug"]> 0).astype(int)
df18['bug'] = (df18['bug'] > 0).astype(int)
df19['bug']=(df19["bug"]> 0).astype(int)
df20['bug'] = (df20['bug'] > 0).astype(int)
df21['bug'] = (df21['bug'] > 0).astype(int)
df22['bug'] = (df22['bug'] > 0).astype(int)
df23['bug'] = (df23['bug'] > 0).astype(int)
df24['bug'] = (df24['bug'] > 0).astype(int)
df25['bug']=(df25["bug"]> 0).astype(int)
df26['bug'] = (df26['bug'] > 0).astype(int)
df27['bug']=(df27["bug"]> 0).astype(int)
df28['bug'] = (df28['bug'] > 0).astype(int)
df29['bug'] = (df29['bug'] > 0).astype(int)
df30['bug'] = (df30['bug'] > 0).astype(int)
df31['bug'] = (df31['bug'] > 0).astype(int)
df32['bug'] = (df32['bug'] > 0).astype(int)
df33['bug']=(df33["bug"]> 0).astype(int)
df34['bug'] = (df34['bug'] > 0).astype(int)
df35['bug']=(df35["bug"]> 0).astype(int)
df36['bug'] = (df36['bug'] > 0).astype(int)
df37['bug'] = (df37['bug'] > 0).astype(int)
df38['bug'] = (df38['bug'] > 0).astype(int)
df39['bug'] = (df39['bug'] > 0).astype(int)
df40['bug'] = (df40['bug'] > 0).astype(int)
df41['bug']=(df41["bug"]> 0).astype(int)
df42['bug'] = (df42['bug'] > 0).astype(int)
df43['bug']=(df43["bug"]> 0).astype(int)
df44['bug'] = (df44['bug'] > 0).astype(int)
df45['bug'] = (df45['bug'] > 0).astype(int)
df46['bug'] = (df46['bug'] > 0).astype(int)
df47['bug'] = (df47['bug'] > 0).astype(int)
df48['bug'] = (df48['bug'] > 0).astype(int)
df49['bug'] = (df49['bug'] > 0).astype(int)
df50['bug']=(df50["bug"]> 0).astype(int)
df51['bug'] = (df51['bug'] > 0).astype(int)
df52['bug'] = (df52['bug'] > 0).astype(int)
df53['bug'] = (df53['bug'] > 0).astype(int)
df54['bug'] = (df54['bug'] > 0).astype(int)
df55['bug'] = (df55['bug'] > 0).astype(int)
df56['bug']=(df56["bug"]> 0).astype(int)
df57['bug'] = (df57['bug'] > 0).astype(int)
df58['bug']=(df58["bug"]> 0).astype(int)
df59['bug'] = (df59['bug'] > 0).astype(int)
df60['bug'] = (df60['bug'] > 0).astype(int)
df61['bug'] = (df61['bug'] > 0).astype(int)
df62['bug'] = (df62['bug'] > 0).astype(int)


# In[4]:


# Selecting the modules where loc (lines of code) is greater than zero

df1=df1[df1["loc"]>0]
df2=df2[df2["loc"]>0]
df3=df3[df3["loc"]>0]
df4=df4[df4["loc"]>0]
df5=df5[df5["loc"]>0]
df6=df6[df6["loc"]>0]
df7=df7[df7["loc"]>0]
df8=df8[df8["loc"]>0]
df9=df9[df9["loc"]>0]
df10=df10[df10["loc"]>0]
df11=df11[df11["loc"]>0]
df12=df12[df12["loc"]>0]
df13=df13[df13["loc"]>0]
df14=df14[df14["loc"]>0]
df15=df15[df15["loc"]>0]
df16=df16[df16["loc"]>0]
df17=df17[df17["loc"]>0]
df18=df18[df18["loc"]>0]
df19=df19[df19["loc"]>0]
df20=df20[df20["loc"]>0]
df21=df21[df21["loc"]>0]
df22=df22[df22["loc"]>0]
df23=df23[df23["loc"]>0]
df24=df24[df24["loc"]>0]
df25=df25[df25["loc"]>0]
df26=df26[df26["loc"]>0]
df27=df27[df27["loc"]>0]
df28=df28[df28["loc"]>0]
df29=df29[df29["loc"]>0]
df30=df30[df30["loc"]>0]

df31=df31[df31["loc"]>0]
df32=df32[df32["loc"]>0]
df33=df33[df33["loc"]>0]
df34=df34[df34["loc"]>0]
df35=df35[df35["loc"]>0]
df36=df36[df36["loc"]>0]
df37=df37[df37["loc"]>0]
df38=df38[df38["loc"]>0]
df39=df39[df39["loc"]>0]
df40=df40[df40["loc"]>0]
df41=df41[df41["loc"]>0]
df42=df42[df42["loc"]>0]
df43=df43[df43["loc"]>0]
df44=df44[df44["loc"]>0]
df45=df45[df45["loc"]>0]

df46=df46[df46["loc"]>0]
df47=df47[df47["loc"]>0]
df48=df48[df48["loc"]>0]
df49=df49[df49["loc"]>0]
df50=df50[df50["loc"]>0]
df51=df51[df51["loc"]>0]
df52=df52[df52["loc"]>0]
df53=df53[df53["loc"]>0]
df54=df54[df54["loc"]>0]
df55=df55[df55["loc"]>0]
df56=df56[df56["loc"]>0]
df57=df57[df57["loc"]>0]
df58=df58[df58["loc"]>0]
df59=df59[df59["loc"]>0]
df60=df60[df60["loc"]>0]
df61=df61[df61["loc"]>0]
df62=df62[df62["loc"]>0]


# In[5]:


# combining all the datasets into a single dataset with keys as their names

datasets=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df59,df60,df61,df62], keys=['ant1','ant2','ant3','ant4','ant5','arc','berek','camel1','camel2','camel3','camel4','ckjm','elearn','forrest2','forrest3','intercafe','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru','pbeans1','pbeans2','pdftranslator','poi1','poi2','poi3','poi4','redaktor','serapion','skarbonka','sklebagd','synapse1','synapse2','synapse3','systemdata','szybkafucha','termoproject','tomcat','velocity2','velocity3','workflow','wspomagani','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel'])
print(datasets)


# In[6]:


# creating a list of all datasets 

list2=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df59,df60,df61,df62]


# In[7]:


# function to calculate the Euclidean distance between two vectors "row1" and "row2"
from math import sqrt

def euclidean_distance(row1, row2):
    
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# function to select "num_neighbors" nearest neighbours of a vector "test_row" from a dataset "train"

def get_neighbors(train, test_row, num_neighbors):
	distances = []
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


# In[8]:


# This function will accept the name, index and the list of the earlier releases of a particular target project
# list1 (inside the function) contains the name of all the projects, where the number represents its version
# index of a target project means its location in list 1, for example, "ant1" target project has index value "0"
# Taking all inputs, it will generate the the cross-company data for that particular target project by removing all its previous versions, if any
# output will contain the cross-company dataset "cc_data", the target project dataset "X_test1" and its defect labels "y_test"


def KNN (target_data_label, target_data_index,list_earlier_release):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    get_ipython().run_line_magic('matplotlib', 'inline')
    import numpy as np
    from sklearn import metrics

    list1=['ant1','ant2','ant3','ant4','ant5','arc','berek','camel1','camel2','camel3','camel4','ckjm','elearn','forrest2','forrest3','intercafe','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru','pbeans1','pbeans2','pdftranslator','poi1','poi2','poi3','poi4','redaktor','serapion','skarbonka','sklebagd','synapse1','synapse2','synapse3','systemdata','szybkafucha','termoproject','tomcat','velocity2','velocity3','workflow','wspomagani','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel']
    j=target_data_index
    result=[]
    
    if list_earlier_release:
        X=datasets.drop(index=target_data_label, level=0)
        X=X.drop(index=list_earlier_release, level=0)
    else:
         X=datasets.drop(index=target_data_label, level=0)
    
    Z=X.copy(deep=True)
     
    X_test=list2[j] # target project 
    X_test1= X_test.drop("bug",axis=1)
    y_test = list2[j]['bug']

    X_train=X.values.tolist()
    
    n=len(X_test1)
    neighbour=[]

    for i in range(n):
        a=np.array(X_test1.iloc[i,:])
        b=get_neighbors(X_train,a,10)
        neighbour.append(b)
    listOfList = neighbour
 
   # Use list comprehension to convert a list of lists to a flat list 
    flatList = [ item for elem in listOfList for item in elem]
 
    l2=['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
       'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc',
       'avg_cc', 'bug']
    cc_data=pd.DataFrame(flatList,columns=l2)
    
    cc_data.drop_duplicates(keep='first',inplace=True) 
    cc_data=cc_data.reset_index(drop=True)
    
    return cc_data,X_test1,y_test


# In[9]:


def compute_tp_tn_modified(actual,pred):
    tp=sum((actual==1)&(pred==1))
    tn=sum((actual==0)&(pred==0))
    fp=sum((actual==0)&(pred==1))
    fn=sum((actual==1)&(pred==0))
    return tp,tn,fp,fn


# In[10]:


# function for applying SMOTE to handle class imbalance issue

def smote(X_train,y_train):
    
    from imblearn.over_sampling import SMOTE
    smt = SMOTE()
    X_train, y_train = smt.fit_sample(X_train, y_train)
    return X_train,y_train


# In[11]:


# function to calculate the MIC (Maximum Information Coefficient) of all the features of a dataset  "train_data"
# this function will return a list of MIC values for all the features

def mic(train_data):
    import numpy as np
    from minepy import MINE
    X=train_data.iloc[:,0:20]
    label=train_data.iloc[:,-1]
    col_list=X.columns
    MIC=[]
    for i in range(0,len(col_list)):
        x=X[col_list[i]]
        y=label
        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        MIC.append(mine.mic())
    return MIC


# In[12]:


# function for matrix multiplication
#X is (n,m) array and Y is (m,1) array 

def mat_mul(X,Y):
    import numpy as np  
    len1=X.shape
    
    rows=len1[0]
    m=len1[1]
    col=1
    Y=Y.reshape(-1,1)
    result=np.empty((rows,col))
    for i in range(0,rows):
        for j in range(0,col):
            result[i][j]=0
            for k in range(0,m):
                result[i][j] += X[i][k] * Y[k][j]
    result=result.reshape(-1,1)
    return result


# In[13]:


# function to calculate similarity weights of cross-company data "train_data" 
# train data with labels and test data without labels

def feature_similarity(train_data,test_data):
    import math
    MIC=mic(train_data)
   
    MIC=np.array(MIC)
    MIC.reshape(-1,1)
    train_data=train_data.iloc[:,0:20]
    len1=train_data.shape
    max1=np.empty(len1[1])
    min1=np.empty(len1[1])
    SW=np.empty(len1[0])
    disimilarity=[]
    a=np.empty((train_data.shape))
    
    max1=np.max(test_data,axis=0)
    min1=np.min(test_data,axis=0)
   
    col_list=test_data.columns
    n=len1[0]
    m=len1[1]
   
    for j in range(0,n):
        for k in range(0,m):
            if train_data.iat[j,k]<=max1[k] and train_data.iat[j,k]>=min1[k]:
                a[j][k]=1
            else:
                a[j][k]=0
              
    #multiplying two matrix a and MIC to claculate similarity based on feature distribution and feature correlation woith fault
    total_MIC=np.sum(MIC)
    Similarity = mat_mul(a,MIC)
    
    Similarity=Similarity.tolist()
    l=len(Similarity)
    for i in range(0,l):
        s=total_MIC-Similarity[i]+1
        disimilarity.append(np.power(s,2))
   
    SW=np.divide(Similarity,disimilarity)   
   
    return SW,total_MIC


# In[14]:


# This is the weight boosting module which provides implementation of Dynamic Transfer Adaboost Classifier  




from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.special import xlogy

from sklearn.ensemble._base import BaseEnsemble
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_array, check_random_state, _safe_indexing
from sklearn.utils.extmath import softmax
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _deprecate_positional_args



class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=0):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        return check_array(X, accept_sparse=['csr', 'csc'], ensure_2d=True,
                           allow_nd=True, dtype=None)

    def _validate_data(self,X, y):
        return check_X_y(X, y,accept_sparse=['csr', 'csc'],
                                   ensure_2d=True,
                                   allow_nd=True,
                                   dtype=None,
                                   y_numeric=is_regressor(self))
    def fit(self, X, y, len_s,len_t,sample_weight=None):
        """Build a boosted classifier from the training set (X, y), which is constructed by merging cross_company data along with target data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
        len_s : number of samples from the cross-company data in the training dataset (X,y)
        len_t : number of samples from the target data in the training dataset (X,y)
        sample_weight : array-like of shape (n_samples,)
        
        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = check_X_y(X, y,accept_sparse=['csr', 'csc'],
                                   ensure_2d=True,
                                   allow_nd=True,
                                   dtype=None,
                                   y_numeric=is_regressor(self))
        

        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        #sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")
        
        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initializion of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            sample_weight1= sample_weight/ sample_weight.sum()
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(iboost,
                X, y,len_s,len_t,sample_weight1,
                random_state )

            # Early termination
            if sample_weight is None:
                break
            # Stop if error is zero 
            if estimator_error == 0 :
                self.estimator_weights_=self.estimator_weights_[:iboost+1]
                self.n_estimators=iboost+1
                self.estimator_weights_[iboost] = estimator_weight
                self.estimator_errors_[iboost] = estimator_error
                break
            if estimator_error >= 0.5 :
                self.estimator_weights_=self.estimator_weights_[:iboost]
                self.n_estimators=iboost
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            #if iboost < self.n_estimators - 1:
                
                #print(" weight for the  next round",sample_weight)
                

        return self,self.n_estimators

    @abstractmethod
    def _boost(self, iboost, X, y,len_s,len_t,sample_weight, random_state):
        """Implement a single boost.
        Warning: This method needs to be overridden by subclasses.
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        len_s: number of samples from the cross-company data in the training dataset (X,y)
        len_t: number of samples from the traget data in the training dataset (X,y)
        sample_weight : array-like of shape (n_samples,)
            The current sample weights.
        random_state : RandomState
            The current random number generator
        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            
        estimator_weight : float
            The weight for the current boost.
            
        error : float
            The classification error for the current boost.
            
        pass
       """
    

class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    """ An AdaBoost classifier.
    
    This class implements the algorithm known as Dynamic Transfer AdaBoost-DTA.
    
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, default=1.
        
    algorithm : DTA, default='DTA'
        
        
    random_state : int or RandomState, default=None
        Controls the random seed given at each `base_estimator` at each
        boosting iteration.
        Thus, it is only used when `base_estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.
    
    """
    @_deprecate_positional_args
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='DTA',
                 random_state=0):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        
    def _validate_data(self,X, y):
        return check_X_y(X, y,accept_sparse=['csr', 'csc'],
                                   ensure_2d=True,
                                   allow_nd=True,
                                   dtype=None,
                                   y_numeric=is_regressor(self))
    
   
    def fit(self,X, y,len_s,len_t,sample_weight=None):
        """Build a boosted classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        len_s : number of samples from the  cross-company data in the training set (X,y)
        len_t : number of samples from the  target data in the training set (X,y)
        sample_weight : array-like of shape (n_samples,), default=None
           
        Returns
        -------
        self : object
            Fitted estimator.
        """
        
        return super().fit(X, y,len_s,len_t,sample_weight)

    
    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))
   

    def _boost(self,iboost, X, y,len_s,len_t,sample_weight, random_state):
        """Implement a single boost.
        
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        len_s : number of samples from the cross-company data in the training set (X,y)
        len_t : number of samples from the target data in the training set (X,y)
        sample_weight : array-like of shape (n_samples,)
            The current sample weights.
        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.
        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.
        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
         
        return self._boost_discrete_modified(iboost, X, y,len_s,len_t, sample_weight,
                                        random_state)
            
    

    
    def _boost_discrete_modified(self, iboost, X, y,len_s,len_t, sample_weight, random_state):
        """Implement a single boost using the Dynamic Transfer Adaboost algorithm ."""
       
        estimator = self._make_estimator(random_state=random_state)
        
        n_t= len_t 
        n_s=len_s 
    
        n = X.shape[0]
        
            
        curr_sample_weights = sample_weight
        bata = 1 / (1 + np.sqrt(2 * np.log( n_s/self.n_estimators))) # hedge parameter
        
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y
       
        # Error function
        
        estimator_error=np.sum(sample_weight[n_s:n_s+n_t]* np.abs(y_predict[n_s:n_s+n_t] - y[n_s:n_s+n_t]) / np.sum(sample_weight[n_s:n_s+n_t]))
        
        beta_t=estimator_error/(1-estimator_error)
        
        # calculated correction term for applying Dynamic TransferAdaboost
        correction_term=2*(1-estimator_error)
        estimator_weight = self.learning_rate * (np.log((1. - estimator_error) / estimator_error)) 
           
       

        # Stop if classification is perfect
        if estimator_error <= 0:
            
            return sample_weight,1, 0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
           
            if len(self.estimators_) == 0:
               
                print("base classifier not fitting")         
            return  sample_weight,estimator_weight, estimator_error

      
        if not iboost == self.n_estimators - 1:
            new_sample_weights=np.empty(curr_sample_weights.shape)
            for j in range(0,n_s):
               
                new_sample_weights[j]=curr_sample_weights[j]*np.power(bata,np.abs(y_predict[j]-y[j]))*correction_term
            for k in range(0,n_t):
                new_sample_weights[n_s+k]=curr_sample_weights[n_s+k]* np.power(beta_t,(-np.abs(y_predict[n_s+k]-y[n_s+k])))
               
           
            sample_weight=new_sample_weights
           
        
        return sample_weight, estimator_weight, estimator_error

    

    def predict_proba(self, X):
       
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            
        Returns
        -------
        prob : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. 
        """
        check_is_fitted(self)
        X = self._check_X(X)

        
        n=X.shape[0]
        N=len(self.estimators_)
        
        prob=np.empty((n,2))
        result_prob_neg=np.ones([n,N])
        result_prob_pos=np.ones([n,N])
        for i in range(0,N):
        
            result_prob_neg[:,i]=self.estimators_[i].predict_proba(X)[:,0]
            result_prob_pos[:,i]=self.estimators_[i].predict_proba(X)[:,1]
        
        
        #  probability calculation considering only second half of the fitted estimators  
        for j in range(0,n):
            if N>1:
                prob_neg=np.sum(result_prob_neg[j, int(np.ceil(N / 2)):N] * self.estimator_weights_[int(np.ceil(N / 2)):N])
                prob_pos=np.sum(result_prob_pos[j, int(np.ceil(N / 2)):N] * self.estimator_weights_[int(np.ceil(N / 2)):N])
                total=np.sum(self.estimator_weights_[int(np.ceil(N / 2)):N])
                prob[j,0]=prob_neg/total
                prob[j,1]=prob_pos/total
            if N==1:
                prob_neg=np.sum(result_prob_neg[j, :] * self.estimator_weights_)
                prob_pos=np.sum(result_prob_pos[j, :] * self.estimator_weights_)
                total=np.sum(self.estimator_weights_)
                prob[j,0]=prob_neg/total
                prob[j,1]=prob_pos/total
                 
        
        return prob

    
    



# In[15]:


# X_train contains the similarity weight also in the last column
def trainAda(n_estimators,  X_train, y_train,len_s,len_t):
    """Train a Dynamic Transfer AdaBoost ensemble. 
    
    Parameters:
            
    n_estimators: integer,  AdaBoost ensemble (maximum) size 
    
    
    
    X_train: array-like, shape (n_samples, n_features), training data
    
    y_train: array-like, shape (n_samples,), training labels
    len_s : number of samples from the cross-campany data in the training set (X_train, y_train)
    len_t : number of samples from the target data in the training set (X_train, y_train)
    Returns: 
        
    AdaBoost: object, a Dynamic Transfer AdaBoost classifier
    """
    
    # removing th similarity weight column from the X_train 
    X_train1 = X_train.iloc[:,0:20]
    sample_weight1=X_train.iloc[:,-1]
    
 
    #Train a Dynamic TrAdaBoost ensemble
    from sklearn.naive_bayes import GaussianNB
    mod=GaussianNB()
    TrAdaBoostDynamic = AdaBoostClassifier(base_estimator=mod,n_estimators=n_estimators,random_state=0)
    TrAdaBoostDynamic,no_fitted_estimators = TrAdaBoostDynamic.fit(X_train1, y_train,len_s,len_t,sample_weight=sample_weight1)
 
    
    return TrAdaBoostDynamic,no_fitted_estimators

 


def predict_prob_Ada(AdaBoostClassifier1,  X_test):
    """

          Parameters:

          AdaBoostClassifier1: object, a Dynamic Adaboost classifier object as
                                        returned by trainAda()


          X_test: array-like, shape (n_samples, n_features), test data

          Returns:

          

          scores_Ada: array-like, shape (n_samples), predicted scores for the positive
                                   class for the training data 
    
     """    
    scores_Ada =AdaBoostClassifier1.predict_proba(X_test)[:,1]#Positive Class scores
 
    

    return  scores_Ada


# In[16]:


""" This function will take target data "X_test", probability score "prob_score" generated by Dynamic Transfer Adaboost classifier , 
predicted labels "y_pred_Ada" generated by Dynamic Transfer Adaboost classifier and actual labels " y_test" 
for preparing a dataframe containing all faulty and all non-faulty modules in order, sorted by {prob_score/loc, avg_cc } individually.
This dataframe will be used in three functions namely "PII_20, costeffort20 and IFA20" to calculate effort based measures.
"""
def sorted_df_epm (X_test, prob_score,  y_pred_Ada,y_test):
    
    import pandas as pd
    
    import numpy as np
   
    test_df_pred=pd.DataFrame()
    test_df_pred=X_test.copy(deep=True)
    test_df_pred['score']=prob_score
    
    test_df_pred['score/loc']=np.divide(prob_score,X_test["loc"])
    test_df_pred['pred']=  y_pred_Ada
    test_df_pred['actual']=y_test
    
    defective=pd.DataFrame()
    non_defective=pd.DataFrame()
    defective=test_df_pred[test_df_pred.pred==1]
    
    
    non_defective=test_df_pred[test_df_pred.pred==0]
    
    
    defective.sort_values(["score/loc","avg_cc"] ,axis = 0, ascending = [False,False], 
                inplace = True)
    defective=defective.reset_index(drop=True)
    
    non_defective.sort_values(["score/loc","avg_cc"], axis = 0, ascending = [False,False], 
                 inplace = True)
    
    non_defective=non_defective.reset_index(drop=True)
    target=pd.DataFrame()
    target=pd.concat([defective,non_defective],axis=0) 
    target=target.reset_index(drop=True)
    
    return target


# In[17]:


# function to calculate PII@20 effort based measure
def PII_20(target_data):
    import pandas as pd
    d=pd.DataFrame()
    d= target_data.copy(deep=True)
    import numpy as np
    total_loc=np.sum(d["loc"])
   
    m=0.2*total_loc
   
    sum_loc=0
    count_module=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break
       
    M=d.shape[0] # total modules
    
    return count_module/M


# In[18]:


# function to calculate CostEffort@20 effort based measure

def costeffort20(target_data):
    import pandas as pd
    import numpy as np
    d=pd.DataFrame()
    d_defective=pd.DataFrame()
    d= target_data.copy(deep=True)
    
    total_loc=np.sum(d["loc"])
    m=0.2*total_loc
    sum_loc=0
    count_module=0
    count_actualdefective_module=0
    total_defective=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break
        
    for i in range(0,count_module):
        if d.loc[i,"actual"]==1:
            count_actualdefective_module+=1
    d_defective=d[d.actual==1]
    total_defective=len(d_defective)
    cost=count_actualdefective_module/ total_defective
    return cost


# In[19]:


# function to calculate IFA@20 effort based measure

def IFA20(target_data):
    import pandas as pd
    import numpy as np
    d=pd.DataFrame()
    d= target_data.copy(deep=True)
    
    total_loc=np.sum(d["loc"])
    m=0.2*total_loc
    sum_loc=0
    count_module=0
    count_actualdefective_module=0
    total_defective=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break
        else:
            continue
    false_alarm=0
    for i in range(0,count_module): 
        if d.loc[i,"pred"]==1 and d.loc[i,"actual"]==1:
            break
        else:
            false_alarm+=1
            
    return false_alarm


# In[20]:


""""  This function will take three inputs: 1) probability score" prob_score"   generated by the Dynamic Transfer Adaboost Classifier,
 2) Actual class labels "y_test" of the testing data
 3) testing data "X_test"
 OutPut: a list of NEPMs follwed by EPMs
"""
def predict_label(prob_score,y_test,X_test):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score,precision_score,roc_auc_score
    from sklearn.metrics import matthews_corrcoef

    target=pd.DataFrame()
   
    y_pred_Ada = np.zeros(y_test.shape)
    y_pred_Ada[np.where(prob_score >= 0.5)] = 1
    rec=recall_score(y_test,y_pred_Ada)
    prec=precision_score(y_test,y_pred_Ada)
    f1=2*prec*rec/(prec+rec)
    
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred_Ada).ravel()
    err1=fp/(fp+tn) # PF   
    a=1-err1
    g_measure=2*rec*a/(rec+a)
    mcc=matthews_corrcoef(y_test,y_pred_Ada)
    
    # calculating effort aware measures
       
        
    target=sorted_df_epm (X_test, prob_score,  y_pred_Ada, y_test) 
    PII=PII_20(target)
    
    costeffort=costeffort20(target)
   
    IFA=IFA20(target)
    
    t=[g_measure,mcc,err1,PII,costeffort,IFA]
    
    
    return t


# In[21]:


""""
This function will train the Dynamic Transfer Adaboost classifier on mixed project data

it will take following inputs:
1) X_train1, y_train1 : training data 
2) X_test1,y_test1: target data 
3) seed : a parameter to control random splitting of target data
Outputs: 
1) no_fitted_estimator: number of fitted estimators
2) t : a list of NEPMs followed by EPMs
"""
def TPTB(X_train1,X_test1,y_train1,y_test1,seed):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score,precision_score,roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import StratifiedShuffleSplit
    
    X_train12=pd.DataFrame()
    X_train12=X_train1.copy(deep=True)
    # constructing the train data set with labels to be passed in weight calculation function
    X_train_sw=X_train12.copy(deep=True)
    X_train_sw["bug"]=y_train1
    
   
    
    sw,total_MIC=feature_similarity(X_train_sw,X_test1)
    # including the sample weight in the training data to be passed to the Dynamic Adaboost classifier
    
    X_train12["sample_weight"]=sw
   
   
    row_s=X_train12.shape[0]
    #print(" no. of samples in cross-company data",row_s)
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed) 
    for train_index, test_index in sss.split(X_test1, y_test1):
           
        x_test11, x_test_train = X_test1.iloc[train_index], X_test1.iloc[test_index]
        y_test11, y_test_train = y_test1.iloc[train_index], y_test1.iloc[test_index] 
        x_test_train=x_test_train.reset_index(drop=True)
        y_test_train=y_test_train.reset_index(drop=True)
      
        row_t=x_test_train.shape[0]
        #print(" no of samples in the train_target",row_t)
        x_test_train["sample_weight"]=np.ones([row_t, 1]) * total_MIC
           
       
            
        X=pd.concat([X_train12,x_test_train],keys=["S","T"])
        y=pd.concat([y_train1,y_test_train],keys=["S","T"])
        
        X=X.reset_index(drop=True)
        y=y.reset_index(drop=True)
       
        x_test11=x_test11.reset_index(drop=True)
        y_test11=y_test11.reset_index(drop=True)
    
        ada,no_fitted_estimator=trainAda(n_estimators=50, X_train=X,y_train=y,len_s=row_s,len_t=row_t)
       
        
        if no_fitted_estimator>=1:
                    
            prob=predict_prob_Ada(ada, x_test11)
                
            t=predict_label(prob,y_test11,x_test11)
            
            return no_fitted_estimator ,t
        else:
            return 0,0


# In[22]:


""" This is the main function from where the process starts.

Run individually for every target data by  setting the valid range in the for loop such that 
it will execute only once individually for every target data. 
For example, setting the for loop range as (0,1) will make it run only once with value of "i" (i.e. the target index) as "0", for "ant1" target projec.
Similarly, setting the for loop range as as (1,2) will make it run for "ant2" target project and so on.

"""
import numpy as np
result=[]
list1=['ant1','ant2','ant3','ant4','ant5','arc','berek','camel1','camel2','camel3','camel4','ckjm','elearn','forrest2','forrest3','intercafe','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru','pbeans1','pbeans2','pdftranslator','poi1','poi2','poi3','poi4','redaktor','serapion','skarbonka','sklebagd','synapse1','synapse2','synapse3','systemdata','szybkafucha','termoproject','tomcat','velocity2','velocity3','workflow','wspomagani','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel']

# Function KNN will take the name of the target project, its label and the list of its previous versions, if any, otherwise pass an empty list
# Will generate the cross-company data "train_cc for it"
for i in range(0,1):
    train_cc,X_test_data,y_test=KNN(list1[i],i,['ant2','ant3','ant4','ant5'])
    
    


# In[23]:


# Applying SMOTE  after KNN
  
X_train12= train_cc.iloc[:,0:20]
y_train12= train_cc.iloc[:,-1]
X_train2,y_train2=smote(X_train12,y_train12)
X_train1=pd.DataFrame(X_train2,columns=['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
       'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc',
       'avg_cc'])
    
X_train1["bug"]=y_train2
final_CC_train_data= X_train1 # cross-company data after applying SMOTE


# In[28]:



# Passing the training data obtained after applying SMOTE in function "TPTB"
# if number of fitted estimators are zero in any round,then nothing is returned and that round will not be counted.
#Therefore, we are running a for loop for a greater number of rounds and stopping as soon as  20 successful rounds get completed.
# finally, recording the mean performance for the selected target project.   

import random
    
X_train11= final_CC_train_data.iloc[:,0:20]
y_train11= final_CC_train_data.iloc[:,-1]
    
seed=[]
my_result=[]

count=0

for i in range(0,200):
    result=[]
    
    #  generating random seed for splitting
    n = random.randint(0,1000)
    
    no_fit_estimators_my,result=TPTB(X_train11,X_test_data,y_train11,y_test,n)
    if  no_fit_estimators_my>=1:
        count=count+1
        my_result.append(result)
        seed.append(n)
    if count==20:
        break

        
            
            
        
    
    


# In[29]:



np.mean(my_result,axis=0)
           
            
            







