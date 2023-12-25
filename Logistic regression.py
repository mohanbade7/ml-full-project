#!/usr/bin/env python
# coding: utf-8

# In[64]:


## logistic regression


# In[122]:


import seaborn as sns
import pandas as pd

df=sns.load_dataset('titanic')

df


# In[123]:


df.columns


# In[124]:


df.drop(['embarked','alive'],axis=1,inplace=True)


# In[125]:


df


# In[126]:


df.columns


# In[127]:


df.drop_duplicates(inplace=True)


# In[128]:


df


# In[ ]:





# In[129]:


df.fillna(method='ffill')


# In[130]:


df.isnull().sum()


# In[131]:


df.fillna(method='bfill')


# In[132]:


df.isnull().sum()


# In[133]:


df=df.fillna(method='bfill')
df=df.fillna(method='ffill')


# In[134]:


df.isnull().sum()


# In[135]:


df


# In[136]:


df.columns


# In[137]:


from sklearn.preprocessing import LabelEncoder





ordinal=['class','deck']
nominal=['sex','who','adult_male','embark_town','alone']

model=LabelEncoder()
for col in ordinal:
    df[col]=model.fit_transform(df[col])
    


# In[138]:


df


# In[139]:


from sklearn.preprocessing import LabelEncoder





ordinal=['class','deck']
nominal=['sex','who','adult_male','embark_town','alone']

model=LabelEncoder()
for col in ordinal:
    df[col]=model.fit_transform(df[col])
    
    pd.get_dummies(df[nominal])


# In[140]:


pd.get_dummies(df[nominal])


# In[141]:


dff=pd.get_dummies(df[nominal])


# In[142]:


dff


# In[143]:


dff


# In[144]:


df.columns


# In[145]:


dff[['survived','pclass','age','sibsp','parch','fare','class','deck']]=df[['survived','pclass','age','sibsp','parch','fare','class','deck']]
dff


# In[146]:


dff=dff.astype('int')


# In[147]:


dff


# In[148]:


import seaborn as sns
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

x=dff.drop(['survived'],axis=1)
y=dff['survived']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

model=LogisticRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("*********train******")
print("accuracy:",accuracy_score(y_train,train_pred))
print("precision:",precision_score(y_train,train_pred))
print("recall:",recall_score(y_train,train_pred))
print("f1score:",f1_score(y_train,train_pred))


print("*********test******")
print("accuracy:",accuracy_score(y_test,test_pred))
print("precision:",precision_score(y_test,test_pred))
print("recall:",recall_score(y_test,test_pred))
print("f1score:",f1_score(y_test,test_pred))


# In[ ]:




