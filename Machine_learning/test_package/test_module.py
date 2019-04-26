'''
Created on 04-Aug-2018

@author: NIDHI
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


print("hello world")

urls = pd.read_csv("data.csv")


def makeTokens(f):
    tokens_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    print("+++++++++++")    
    for i in tokens_BySlash:
        tokens = str(i).split('-')
        tokens_Bydot = []
        
        print("...........")
              
        for j in range(0,len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_ByDot = tokens_Bydot + temp_tokens
        print(tokens_ByDot)
        total_Tokens = total_Tokens + tokens + tokens_ByDot
    total_Tokens = list(set(total_Tokens))
    print(total_Tokens)
    if 'com' in total_Tokens :
        total_Tokens.remove('com')
    print("all-tokens:",total_Tokens)
    return total_Tokens

x = urls["url"]
print("urls:",x)
#print(list_url)
y = urls["label"]
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
# f_name = vectorizer.get_feature_names()
# dd = pd.DataFrame(x.toarray(),coumns  = f_name)
# print("name of features",dd)
print("vectorizer:",vectorizer)
X = vectorizer.fit_transform(x)
print("X normal:",X)
print("X:",X.toarray())

def countdots(input_datafile):
    num_dots = input_datafile.count('.')
    return num_dots
 
#def countdots(f):
#    return f.count('.')
#y = countdots(f)
#print(y)




   
        
        