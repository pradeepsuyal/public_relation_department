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