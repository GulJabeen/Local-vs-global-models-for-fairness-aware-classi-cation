
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib
import scipy
from sklearn import cluster
import numpy as np
import sklearn
from  sklearn import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import scatter_matrix
from itertools import cycle
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

url= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = [ "age","workclass", "fnlwgt","education" ,"education-num", "marital-status" ,"occupation","relationship","race","sex", "capital-gain","capital-loss","hours-per-week","native-country","class"]
data = pd.read_csv( url ,sep=',', names = names)


# In[2]:


# print(data.dtypes)
obj_df = data.copy()
# print(obj_df.head())
obj_df[obj_df.isnull().any(axis=1)]
obj_df["education"].value_counts()
obj_df = obj_df.fillna({"education": "HS-grad"})


# In[3]:


obj_df


# In[4]:


print(obj_df.dtypes.education)


# In[5]:


obj_df["sex"].value_counts()
obj_df = obj_df.fillna({"sex": "Female"})


# In[6]:


print(obj_df.dtypes.sex)


# In[7]:


print(obj_df)


# In[8]:


print(obj_df.dtypes.race)


# In[9]:


obj_df["race"].value_counts()


# In[10]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
obj_df["race_code"] = lb_make.fit_transform(obj_df["race"])
obj_df[["race", "race_code"]].head(10)


# In[11]:


obj_df["relationship"].value_counts()


# In[12]:


obj_df["relationship_code"] = lb_make.fit_transform(obj_df["relationship"])
obj_df[["relationship", "relationship_code"]].head(30)


# In[13]:


obj_df["education"].value_counts()


# In[14]:


obj_df["education_code"] = lb_make.fit_transform(obj_df["education"])
obj_df[["education", "education_code"]].head(5)


# In[15]:


obj_df["marital-status"].value_counts()


# In[16]:


obj_df["marital-status_code"] = lb_make.fit_transform(obj_df["marital-status"])
obj_df[["marital-status", "marital-status_code"]].head(5)


# In[17]:


obj_df["occupation"].value_counts()


# In[18]:


obj_df["occupation_code"] = lb_make.fit_transform(obj_df["occupation"])
obj_df[["occupation", "occupation_code"]].head(5)


# In[19]:


obj_df["native-country"].value_counts()


# In[20]:


obj_df["native-country_code"] = lb_make.fit_transform(obj_df["native-country"])
obj_df[["native-country", "native-country_code"]].head(5)


# In[21]:


obj_df["workclass"].value_counts()


# In[22]:


obj_df["workclass_code"] = lb_make.fit_transform(obj_df["workclass"])
obj_df[["workclass", "workclass_code"]].head(5)


# In[23]:


obj_df["sex_code"] = lb_make.fit_transform(obj_df["sex"])
obj_df[["sex", "sex_code"]].head(5)


# In[24]:


obj_df["class_code"] = lb_make.fit_transform(obj_df["class"])
obj_df[["class", "class_code"]].head(5)


# In[25]:


print(obj_df)


# In[59]:



obj_df.hist()


# In[28]:


import seaborn as sns
sns.distplot(obj_df.workclass_code.dropna(), kde=False, bins = 39);


# In[29]:


educationfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))

sns.distplot(obj_df.education_code.dropna(), kde=False, ax=axs[0])
second_plt = sns.distplot(obj_df.education_code.dropna()[obj_df.education_code > 2], kde=False, ax=axs[1])
sns.boxplot(obj_df.education_code, ax=axs[2], orient = 'v')
sns.boxplot(obj_df.education_code, ax=axs[3], orient = 'v', showfliers=False)

second_plt.set_yscale('log')


# In[31]:


obj_df["sex"].value_counts()


# In[32]:


obj_df.isnull()


# In[33]:


obj_df.isnull().sum()


# In[34]:


obj_df=obj_df.fillna(" ")


# In[35]:


obj_df.isnull().sum()


# In[36]:


#univariate analysis

obj_df.describe()


# In[37]:


sns.distplot(obj_df.age.dropna(), kde=False, bins = 10);


# In[38]:


sns.distplot(obj_df.education_code.dropna(), kde=False, bins = 10);


# In[39]:


sns.distplot(obj_df["education-num"].dropna(), kde=False, bins = 10);


# In[40]:


plt.figure(figsize=(18, 5))
#http://stackoverflow.com/questions/32891211/limit-the-number-of-groups-shown-in-seaborn-countplot for odering
sns.countplot(obj_df.education.dropna(), order = obj_df.education.value_counts().index);


# In[41]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["marital-status"].dropna(), order = obj_df["marital-status"].value_counts().index);


# In[42]:


plt.figure(figsize=(25, 5))
sns.countplot(obj_df["occupation"].dropna(), order = obj_df["occupation"].value_counts().index);


# In[43]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["relationship"].dropna(), order = obj_df["relationship"].value_counts().index);


# In[44]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["race"].dropna(), order = obj_df["race"].value_counts().index);


# In[45]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["sex"].dropna(), order = obj_df["sex"].value_counts().index);


# In[46]:


sns.distplot(obj_df["capital-gain"].dropna(), kde=False, bins = 10);


# In[47]:


sns.distplot(obj_df["capital-loss"].dropna(), kde=False, bins = 10);


# In[48]:


sns.distplot(obj_df["hours-per-week"].dropna(), kde=False, bins = 10);


# In[49]:


plt.figure(figsize=(13, 4))
sns.countplot(obj_df["native-country"].dropna(), order = obj_df["native-country"].value_counts().iloc[:40].index)
plt.xticks(rotation=90);


# In[50]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["workclass"].dropna(), order = obj_df["workclass"].value_counts().index);


# In[51]:


plt.figure(figsize=(18, 5))
sns.countplot(obj_df["class"].dropna(), order = obj_df["class"].value_counts().index);


# In[52]:


obj_df


# In[53]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from IPython.display import display, HTML
# Any results you write to the current directory are saved as output.

#For plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


#plot correlation matrix (Bi-variate analysis)

plt.figure(figsize=(12, 8))



correlations = obj_df.corr()
sns.heatmap(correlations, 
            xticklabels = correlations.columns.values,
            yticklabels = correlations.columns.values,
            annot = True);


# In[57]:


plt.figure(figsize=(14, 14))

sns.pairplot(obj_df, diag_kind='kde');


# In[65]:


print("-------------------------ALGORITHM 01: K-means clustering algorithm------------------")
centers = [[1, 1], [-1, -1], [1, -1]]
obj_df, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)
k_means = cluster.KMeans(n_clusters=3
                         , max_iter=1000)
k_means.fit(obj_df)
centroids=k_means.cluster_centers_
labels=k_means.labels_

n_cluster=len(set(labels))

print("Centroids:")
print(centroids)
print('Estimated number of clusters: %d' % n_cluster)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(obj_df, labels))

pca = PCA(n_components=2).fit(obj_df)
pca_2d = pca.transform(obj_df)


pl.figure('K-means with n clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=k_means.labels_)
pl.show()


# In[66]:


print("-------------------------ALGORITHM 02: DBSCAN clustering algorithm------------------------")
centers = [[1, 1], [-1, -1], [1, -1]]
obj_df, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(obj_df)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True

labels = dbsc.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels))

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(obj_df, labels))



unique_labels = set(labels)
colors = [plt2.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples]
    plt2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples]
    plt2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt2.title('Estimated number of clusters: %d' % n_clusters_)
plt2.show()


# In[67]:


print("-------------------------ALGORITHM 03: Affinity propagation clustering algorithm------------------------")


obj_df, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

af = AffinityPropagation(preference=-50).fit(obj_df)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(obj_df, labels, metric='sqeuclidean'))
plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = obj_df[cluster_centers_indices[k]]
    plt.plot(obj_df[class_members, 0], obj_df[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in obj_df[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[70]:


print("-------------------------ALGORITHM 04: Agglomerative Clustering Algorithm------------------")


centers = [[1, 1], [-1, -1], [1, -1]]
obj_df, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)
aglo_means = cluster.AgglomerativeClustering(n_clusters=3
                         , linkage="ward")
aglo_means.fit(obj_df)

labels=aglo_means.labels_

n_cluster=len(set(labels))


print('Estimated number of clusters: %d' % n_cluster)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(obj_df, labels))


pl.figure('Agglomerative Clustering with n clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=aglo_means.labels_)
pl.show()







# In[79]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(obj_df)
y_kmeans = kmeans.predict(obj_df)
plt.scatter(obj_df[:, 0], obj_df[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

