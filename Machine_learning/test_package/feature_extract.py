import pandas as pd
import numpy as np
import tldextract 
import pygeoip
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from urllib.parse import urlparse
from sklearn.svm import LinearSVC

urls = pd.read_csv("data.csv")
x = urls["url"]
nf=-1

def num_deli(url):
    
    token = re.split('\W+',url)
    delim=0 
    for i in token:
        delim=delim+1
    return delim
     
def num_dots(url):
    dots = 0
    char = '.'
    for i in url:
        if char==i:
            dots=dots+1
    return dots
    
def num_hyphen(url):
    hyphen = 0
    char = '-'
    for i in url:
        if char==i:
            hyphen=hyphen+1
    return hyphen

def num_quesmark(url):
    quesmark = 0
    char = '?'
    for i in url:
        if char==i:
            quesmark=quesmark+1
    return quesmark
    
def getASN(host):
    try:
        g = pygeoip.GeoIP('GeoIPASNum.dat')
        asn =int(g.org_by_name(host).split()[0][2:])
        return asn
    except:
        return nf

def frames(x,y):
    all_features = []
    strurls = str(x)
    obj = urlparse(x)
    host = obj.netloc
    path = obj.path
    all_features.append(strurls)
    all_features.append(len(strurls))
    all_features.append(num_dots(strurls))
    all_features.append(num_hyphen(strurls))
    all_features.append(num_quesmark(strurls))
    all_features.append(num_deli(strurls))
    all_features.append(len(host))
    #all_features.append(getASN(host))
    all_features.append(str(y))
    return all_features

dataset = pd.DataFrame(columns=('url','length of url','total dots','total hyphens','total question marks','num_delim','length of host','label'))

for i in range(len(urls)):
    output1 = frames(urls["url"].loc[i],urls["label"].loc[i])
    dataset.loc[i] = output1
    dataset.to_csv("output2.csv")
    

X = dataset.drop(['url','label'],axis=1).values
print("X:",X)
y = dataset['label'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("X:\n",X,"x_train:\n",x_train)

######################################################
# Logistic Regression
lgs = LogisticRegression()
lgs.fit(x_train,y_train)
print("log reg:",end="")
print(lgs.score(x_test,y_test)*100)

#################################
# SVM 
s = LinearSVC()
s.fit(x_train,y_train)
print("svm:",end="")
print(s.score(x_test,y_test)*100)