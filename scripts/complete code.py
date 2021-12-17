#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES AND DATASETS

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading dataset

reviews_df = pd.read_csv('amazon_alexa.tsv', sep='\t')


# In[3]:


reviews_df


# In[4]:


reviews_df.info()


# In[5]:


reviews_df.describe()


# In[6]:


reviews_df['verified_reviews']


# ### EXPLORE DATASET

# In[7]:


sns.heatmap(reviews_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[8]:


reviews_df.hist(bins = 30, figsize = (13,5), color = 'r')


# In[9]:


# Let's get the length of the messages

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
reviews_df.head()


# In[10]:


reviews_df['length'].plot(bins=100, kind='hist') 


# In[11]:


reviews_df.length.describe()


# In[12]:


# Let's see the longest message 43952
reviews_df[reviews_df['length'] == 2851]['verified_reviews'].iloc[0]


# In[13]:


# Let's see the shortest message 
reviews_df[reviews_df['length'] == 1]['verified_reviews'].iloc[0]


# In[14]:


# Let's see the message with mean length 
reviews_df[reviews_df['length'] == 133]['verified_reviews'].iloc[0]


# In[15]:


positive = reviews_df[reviews_df['feedback']==1]


# In[16]:


negative = reviews_df[reviews_df['feedback']==0]


# In[17]:


negative


# In[18]:


positive


# In[19]:


sns.countplot(reviews_df['feedback'], label = "Count") 


# In[20]:


sns.countplot(x = 'rating', data = reviews_df)


# In[21]:


reviews_df['rating'].hist(bins = 5)


# In[22]:


plt.figure(figsize = (40,15))
sns.barplot(x = 'variation', y='rating', data = reviews_df, palette = 'deep')


# In[23]:


sentences = reviews_df['verified_reviews'].tolist()
len(sentences)


# In[24]:


print(sentences)


# In[25]:


sentences_as_one_string =" ".join(sentences)


# In[26]:


sentences_as_one_string


# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[27]:


negative_list = negative['verified_reviews'].tolist()

negative_list


# In[28]:


negative_sentences_as_one_string = " ".join(negative_list)


# In[ ]:


plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))


# ### PERFORMING DATA CLEANING

# In[29]:


# Let's drop the date
reviews_df = reviews_df.drop(['date', 'rating', 'length'],axis=1)


# In[30]:


reviews_df


# In[31]:


variation_dummies = pd.get_dummies(reviews_df['variation'], drop_first = True)
# Avoid Dummy Variable trap which occurs when one variable can be predicted from the other.


# In[32]:


variation_dummies


# In[33]:


# first let's drop the column
reviews_df.drop(['variation'], axis=1, inplace=True)


# In[34]:


# Now let's add the encoded column again
reviews_df = pd.concat([reviews_df, variation_dummies], axis=1)


# In[35]:


reviews_df


# ### LEARNING HOW TO REMOVE PUNCTUATION FROM TEXT

# In[36]:


import string
string.punctuation


# In[37]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[38]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[39]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# ### UNDERSTANDING HOW TO REMOVE STOPWORDS

# In[40]:


import nltk # Natural Language tool kit 

nltk.download('stopwords')


# In[41]:


# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[42]:


Test_punc_removed_join


# In[43]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[44]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# In[45]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'


# In[46]:


challege = [ char     for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 


# ### UNDERSTANDING HOW TO PERFORM COUNT VECTORIZATION (TOKENIZATION)

# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[48]:


print(vectorizer.get_feature_names())


# In[49]:


print(X.toarray())  


# In[50]:


mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())


# ### PERFORMING DATA CLEANING BY APPLYING EVERYTHING WE LEARNED SO FAR!

# In[51]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[52]:


# Let's test the newly added function
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)


# In[53]:


print(reviews_df_clean[3]) # show the cleaned up version


# In[54]:


print(reviews_df['verified_reviews'][3]) # show the original version


# In[55]:


reviews_df_clean


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])


# In[57]:


print(vectorizer.get_feature_names())


# In[58]:


print(reviews_countvectorizer.toarray())  


# In[59]:


reviews_countvectorizer.shape


# In[60]:


reviews_df


# In[61]:


# first let's drop the column
reviews_df.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(reviews_countvectorizer.toarray())


# In[62]:


# Now let's concatenate them together
reviews_df = pd.concat([reviews_df, reviews], axis=1)


# In[63]:


reviews_df


# In[64]:


# Let's drop the target label coloumns
X = reviews_df.drop(['feedback'],axis=1)


# In[65]:


X


# In[66]:


y = reviews_df['feedback']


# ### TRAINING A NAIVE BAYES CLASSIFIER MODEL

# In[67]:


X.shape


# In[68]:


y.shape


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[70]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# ### EVALUTING TRAINED MODEL PERFORMANCE  

# In[71]:


from sklearn.metrics import classification_report, confusion_matrix


# In[72]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[73]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.savefig('confustion_matrix_on_test_set.png')


# In[74]:


print(classification_report(y_test, y_predict_test))


# ### TRAINING AND EVALUATING A LOGISTIC REGRESSION CLASSIFIER

# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[76]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[77]:


y_pred = model.predict(X_test)


# In[78]:


# Testing Set Performance
y_pred


# In[79]:


from sklearn.metrics import confusion_matrix, classification_report

print('Accuracy {} %'.format( 100 * accuracy_score(y_pred, y_test)))


# In[82]:


cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)
plt.savefig('correlation_logistic_reg.png')


# In[83]:


print(classification_report(y_test, y_pred))

