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

