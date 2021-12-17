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