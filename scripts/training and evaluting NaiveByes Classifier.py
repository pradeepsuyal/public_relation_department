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