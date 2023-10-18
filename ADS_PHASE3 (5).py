#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ADS-PHASE-3

#PROJECT TITLE-COVID-19 VACCINE ANALYSIS

#TEAM NO-1

#---------PREPARED BY VICKY(TEAM MEMBER)




#importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


#loading the dataset
file_path = r"D:\country_vaccinations.csv"
encoding = "ISO-8859-1"
df = pd.read_csv(file_path, encoding=encoding)
df


# In[3]:


#getting info of the dataset
df.info()


# In[4]:


#checking the missing values in the dataset
df.isnull()


# In[12]:


#filling and droping the missing values of the dataset
df.fillna(df.mean(), inplace=True) 
df.dropna(inplace=True)


# In[6]:


df.isnull()


# In[7]:


categorical_columns = ['total_vaccinations']

# Perform one-hot encoding using get_dummies
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Display the one-hot encoded DataFrame
print(df_encoded)


# In[9]:


#feature scaling is done here
scaler = StandardScaler()
df['total_vaccinations'] = scaler.fit_transform(df['total_vaccinations'].values.reshape(-1, 1))
df


# In[16]:


#splitting the dataset into test set and training set
X = df.drop('total_vaccinations', axis=1)
y = df['people_vaccinated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n X_test info")
print(X_test.info())
print(y_test.info())


# In[ ]:




