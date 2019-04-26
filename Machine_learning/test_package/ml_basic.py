'''
Created on 04-Aug-2018

@author: NIDHI
'''

import pandas as pd
import numpy as np
import tldextract 
from urllib import parse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm


urls = pd.read_csv("data.csv")
x = urls["url"]

print(x)
#print(list_url)

 
def countdots(url):
    dots = 0
    char = '.'
    for i in url:
        if char==i:
            dots=dots+1
    return dots
    
def counthyphen(url):
    return url.count('-')
def countat(urls):
    return urls.count('@')

def frames(x,y):
    all_features = []
    strurls = str(x)
    all_features.append(strurls)
    all_features.append(len(strurls))
    all_features.append(countdots(strurls))
    all_features.append(str(y))
    return all_features

dataset = pd.DataFrame(columns=('url','length of url','total dots','label'))

for i in range(len(urls)):
    output1 = frames(urls["url"].loc[i],urls["label"].loc[i])
    dataset.loc[i] = output1
    dataset.to_csv("output.csv")
    

X = dataset.drop(['url','label'],axis=1).values
print("X:",X)
y = dataset['label'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("X:\n",X,"x_train:\n",x_train)
lgs = LogisticRegression()
lgs.fit(x_train,y_train)
print("log reg:",end="")
print(lgs.score(x_test,y_test)*100)




   
        
        