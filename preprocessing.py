#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries
import nltk
from nltk import FreqDist
nltk.download('stopwords') # run this one time

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#world Cloud Libraries 
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path, getcwd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#Getting a list of english stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

get_ipython().system('python -m spacy download en # one time run')



# In[2]:


def cleantext(df):

    #removing unwanted characters , symbols and numbers
    df['text'] = df['text'].str.lower()
    
    df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")

    df['text'] = df['text'].str.replace("https", " ")
    
    df['text'] = df['text'].str.replace("book", " ")
    
    df['text'] = df['text'].str.replace("museum" , " ")
    
    df['text'] = df['text'].str.replace("story" , " ")
    

    
    #Function that removes stopwords 
    def remove_stopwords(rev):
        rev_new = " ".join([i for i in rev if i not in stop_words])
        return rev_new 

    #Removing words that have a length less than 3 
    df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    #removing stopwords from text 
    reviews_df = [remove_stopwords(r.split()) for r in df['text']]

    #Making all text lower case 
    reviews_df = [r.lower() for r in reviews_df]
    
    return reviews_df


# In[3]:


nlp = spacy.load('en', disable = ['parser', 'ner'])

#Lemmatization to get rid of noisy words 
def lemmatization(texts , tags = ['NOUN','ADJ']): #filtering nouns and adjectives 
    
    output = [] 
    for sent in texts: 
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output


# In[ ]:




