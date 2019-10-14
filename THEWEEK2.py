#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system(' pip install folium')


# In[9]:


get_ipython().system(' pip install geopy')


# In[10]:


from pandas.io.json import json_normalize
import folium
from geopy.geocoders import Nominatim 
import requests


# In[11]:


CLIENT_ID = 'XZBWAYLW2PIQ0AREL4XHGUD5OAHGVAXJI13FC1J45MDW4DKZ' # your Foursquare ID
CLIENT_SECRET = 'GWLM1YQKOGD2XDJQYXPDJ15TZ0QH5ZTM5LRG112DMFUUYVI5' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 200


# In[12]:


urltah='https://api.foursquare.com/v2/venues/search?categoryId=4bf58dd8d48988d14e941735&ll=21.60693,39.1270057&client_id=XZBWAYLW2PIQ0AREL4XHGUD5OAHGVAXJI13FC1J45MDW4DKZ&client_secret=GWLM1YQKOGD2XDJQYXPDJ15TZ0QH5ZTM5LRG112DMFUUYVI5&v=20180604'


# In[13]:


x = requests.get(urltah).json()


# In[16]:


import pandas as pd
from pandas.io.json import json_normalize
json_normalize(x)


# In[20]:


tah_data=x['response']
tah=tah_data['venues']
json_normalize(tah)
newtah=json_normalize(tah)
newtah.head()


# In[284]:


Master =newtah.to_csv('newtah1.csv')


# In[2]:


#Importing the datasetaftercleaning
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset = pd.read_csv('restaa.csv',encoding = "ISO-8859-1")
#dataset = pd.read_csv('ECON1.csv', encoding = 'utf-8')


# In[3]:


#Import first 5 row 
dataset.head()


# In[6]:


import matplotlib.pyplot as pl
Maste_sub=dataset.groupby(dataset["crossStreet"])['id'].count()
Maste_sub.plot.bar()
pl.title('Distribution of Tweet by month')
pl.xlabel('resturentName')
pl.ylabel('Count_of_rest')
pl.show()


# In[23]:


#We want also to make cluster so we need first to choose the coulmn 
x=dataset.iloc[:,[2,3]].values
x


# AS we can see the most American restaurants in prince SultanRD 

# In[25]:


#using the elbow methoed to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,30):
   kmeans = KMeans(n_clusters = i,init ='k-means++',random_state=42)
   kmeans.fit(x)
   wcss.append(kmeans.inertia_)
wcss
plt.plot(range(1,30),wcss)
plt.title('elbow method')
plt.xlabel('wcss')
plt.ylabel('numOfcluster')
plt.show()


# In[26]:


#As we can see above the best number of cluster is five
kmeans= KMeans(n_clusters = 5,init ='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(x)
#labels = y_kmeans.labels_
#y_kmeans=kmeans.fit(x)
kmeans
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
labels
dataset['cluster'] =labels
plt.scatter(x[y_kmeans ==0 , 0],x[y_kmeans ==0 ,1],  s=100 ,c='red'  ,label = 'cluster1')
plt.scatter(x[y_kmeans ==1 , 0],x[y_kmeans ==1 , 1], s=100 ,c='blue' ,label = 'cluster2')
plt.scatter(x[y_kmeans ==2 , 0],x[y_kmeans ==2 , 1], s=100 ,c='brown' ,label = 'cluster3')
plt.scatter(x[y_kmeans ==3 , 0],x[y_kmeans ==3 , 1], s=100 ,c='black' ,label = 'cluster3')
plt.scatter(x[y_kmeans ==4 , 0],x[y_kmeans ==4 , 1], s=100 ,c='green' ,label = 'cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] ,s=300 ,c='yellow',label= 'centroids')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()


# In[27]:


from pandas import Series
dataset['cluster']=Series(labels,index=dataset.index)
#dataset.groupby(['cluster']).count()
dataset.groupby(['cluster']).size()


# In[28]:


dataset


# In[29]:


Master =dataset.to_csv('dataset.csv')

