# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:37:11 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#Loading the dataset
email_data=pd.read_csv("D:/Data Science/6-Datasets/sms_raw_NB.csv.xls",encoding="ISO-8859-1")

#Cleaning the data
import re
def cleaning_txt(i):
    w=[]
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))
                    
#Testing above function with some test text
cleaning_txt("Hope your are having good week.just checking")
cleaning_txt("hope i can understand your feelings 123121.121.hi how are you")
cleaning_txt("hi how are you,i am sad")
email_data.text=email_data.text.apply(cleaning_txt)
email_data=email_data.loc[email_data.text!="",:]
from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(email_data, test_size=0.2)

#Creating matrix of token counts for entire text documents
def split_into_words(i):
    return[word for word in i.split(" ")]

emails_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.text)
all_mails_matrix=emails_bow.transform(email_data.text)

#For training messesges
train_emails_matrix=emails_bow.transform(email_train.text)

#For testing message 
test_emails_matrix=emails_bow.transform(email_test.text)

#Learning  term weightaging and normaliing on entire emails
tfidf_transformer=TfidfTransformer().fit(all_mails_matrix)


#Preparing TFIDF for train mails
train_tfidf=tfidf_transformer.transform(train_emails_matrix)
#Preparing TFIDF for test mails
test_tfidf=tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape

#Let us apply naive bayer to this
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
classifier_mb.fit(train_tfidf,email_train.type)

#Evalution on test data
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.type)
accuracy_test_m

