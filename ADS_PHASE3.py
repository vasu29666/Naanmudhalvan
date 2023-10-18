#!/usr/bin/env python
# coding: utf-8

# In[13]:


#ADS-PHASE-3

#PROJECT TITLE-AIR QUALITY ANALYSIS AND PREDICTION

#TEAM NO-1

#---------PREPARED BY VASUDEVAN(TEAM MEMBER)


# In[14]:


#importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[15]:


#loading data set
file_path = r"D:\project\air1.csv"
encoding = "ISO-8859-1"
df = pd.read_csv(file_path, encoding=encoding)
df


# In[16]:



df.info()


# In[17]:


df.head()


# In[18]:


#to display null values
df.isnull()


# In[19]:


# Define the categorical column(s) you want to one-hot encode
categorical_columns = ['State']
# Perform one-hot encoding using get_dummies
df_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[20]:


# Display the one-hot encoded DataFrame
print(df_encoded)


# In[21]:


#scaling
scaler = StandardScaler()
df['RSPM/PM10'] = scaler.fit_transform(df['RSPM/PM10'].values.reshape(-1, 1))
df


# In[22]:


#train_test split
X = df.drop('RSPM/PM10', axis=1)
y = df['RSPM/PM10']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


print("\n X_test info")
print(X_test.info())


# In[ ]:




