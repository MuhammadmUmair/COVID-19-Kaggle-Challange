#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import re
import nltk
import gensim


# In[34]:


def format_name(author):#this function collect the author name from json file
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation): #this function collect the location and institudion of the author.
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):# combine all the text body part from the json file.
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):# this function use for topic of research paper
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


# In[35]:


def load_files(dirname): #there is use os library to collect the paper directry 
    filenames = os.listdir(dirname)
    raw_files = []
    
    for filename in tqdm(filenames): #one by one select the files name and load them into memory and append them as raw_files
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files
biorxiv_dir ='I:\\COVID\\comm_use_subset\\comm_use_subset\\pdf\\' #path of dir where data layes
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))

all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
    
cleaned_files = []

for file in all_files: # the loop send whole paper differnt towarde function use to collect all data from json file
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'],with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    cleaned_files.append(features) #list of data append into the variable(cleaned_files)


# In[36]:


col_names = [                #columns name use into pandas file
    'paper_id',        
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]
new_data = pd.DataFrame(cleaned_files, columns=col_names) #make a dataframe into pandas
new_data


# In[37]:


complete_df = new_data[new_data['text'].apply(lambda x: len(re.findall(r"(?i)\b[a-z]+\b", x))) > 1000] #clean the all text into the file one by one by using anonymace function


# In[38]:


from nltk.corpus import stopwords #load corpus of word
nltk.download('stopwords')


# In[39]:


import multiprocessing #import the multi-processing library for multi-processing due to huge ammount of data
context_size = 7
num_workers = multiprocessing.cpu_count() #count function count the sys. processor
min_word_count = 3
num_features = 300
seed = 1
word2vec_model = gensim.models.word2vec.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers, 
    size=num_features, 
    min_count=min_word_count, 
    window=context_size) #build model W2V


# In[40]:


word_tok=[]
stop_words = set(stopwords.words('english'))  #convert into word tokize
for doc in complete_df.iloc[:]['text'][:]:
    word_tok.append(nltk.word_tokenize(doc))
    


# In[41]:


filtered_sentence=[] #remove stopword
list1=[]
for doc_ in word_tok[:]:
    for w in doc_:
        if w not in stop_words: 
            filtered_sentence.append(w)
    list1.append(filtered_sentence) 


# In[42]:


word2vec_model.build_vocab(sentences=list1) #build model vocabulary
print("The vocabulary is built")
print("Word2Vec vocabulary length: ",len(word2vec_model.wv.vocab))


# In[43]:


count = 10000
word_vectors_matrix = np.ndarray(shape=(count, 300), dtype='float64') # A matrix use for load vacabulary
word_list = []
i = 0
for word in word2vec_model.wv.vocab:
    word_vectors_matrix[i] = word2vec_model[word]
    word_list.append(word)
    i = i+1
    if i == count:
        break
print("word_vectors_matrix shape is ", word_vectors_matrix.shape)


# In[44]:


import sklearn.manifold #manifold use to reduce the  dimention reduction
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
print("word_vectors_matrix_2d shape is ", word_vectors_matrix_2d.shape)


# In[45]:


#build Points DataFrame
points = pd.DataFrame(
    [(word, coords[0], coords[1]) for word, coords in [(word, word_vectors_matrix_2d[word_list.index(word)])for word in word_list]
    ],
    columns=["word", "x", "y"]
)
print("Points DataFrame built")


# In[3]:


points.head(10)


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("poster")


# In[48]:


points.plot.scatter("x", "y", s=10, figsize=(20, 12))

