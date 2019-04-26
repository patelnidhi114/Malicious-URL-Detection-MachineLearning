import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm


dataset = pd.read_csv("data.csv")

only_urls = str(dataset["url"])

def countdots(url):
    dots = 0
    char = '.'
    for i in url:
        if char==i:
            dots=dots+1
    return dots
#print("only_urls:",only_urls)

# def split_delim(url):
#        
#     tokens=url.split('\W+')
#     for a_url in range(len(tokens)):
#         dots = 0
#         for a_token in range(len(tokens)):
#             if tokens[a_token] == '.':
#                 dots = dots + 1
#                 return dots
#             hyphen = 0
#             if tokens[a_url] == '-':
#                 hyphen = hyphen + 1
#                 return hyphen
#     print(tokens)

# dotss = re.compile('-')
# if (dotss.match(only_urls)):
#     print("match found")

def dotscount(url):
    dots = 0
    char = '.'
    for i in url:
        if char==i:
            dots=dots+1
    return dotscount

for i in range(len(only_urls)):
#print(len(re.findall(".",only_urls[i])))

# dots = 0
# for i in range(len(urls)):
#     #print(only_urls[i])
#     if only_urls[i] == '.':
#         dots = dots + 1
#         #print(dots)     
# print(only_urls[1])    
#print(only_urls.split('.'))
#print(split_delim(only_urls))
#x = split_delim(urls)
#print(x)
# vectorizer = TfidfVectorizer(tokenizer= split_delim)
# x = vectorizer.fit_transform(urls)
# # y = dataset["label"]
# 
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# 
# lgs = LogisticRegression()
# lgs.fit(x_train,y_train)
# print("log reg:",end="")
# print(lgs.score(x_test,y_test)*100)




# #show_delim = split_delimetrs(urls)
# print("Show delimeters:",)