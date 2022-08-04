#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


x_test = np.load('/Users/johnpark/Desktop/Avellino/ml/X_test.npy')
 
print(x_test)


# In[72]:


x_test[0]
ba=list(reversed(x_test[0]))
print(ba)


# In[90]:


xy=[]
for i in range(3000):
    xx=list(reversed(x_test[i]))
    xy.append(xx)


# In[80]:


print(xy)


# In[40]:


a=list(reversed(wavenumbers))
print(a)


# In[4]:


import pandas as pd


# In[41]:


df=pd.DataFrame(x_test, columns=a)


# In[6]:


df.to_csv('second.csv')


# In[42]:


df


# In[67]:


df.loc[0]


# In[91]:


df2=pd.DataFrame(xy, columns=a)


# In[92]:


df2


# In[170]:


from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[43]:


df.loc[0].plot(legend=None)


# In[99]:


df2.loc[0].plot(legend=None)


# In[93]:


import pybaselines
from pybaselines import utils

def whittaker(n):
    return pybaselines.whittaker.arpls(df2.loc[n], lam=100000.0, diff_order=1, max_iter=1000, tol=0.001, weights=None)


# In[94]:


def baseline(n):
    
    b=[]
    for i in range(0,n+1):
        line=[]
        for j in range(1):
            a=df2.loc[i]-whittaker(i)[0]
            line.append(a)
        b.append(line)
    return b


# In[95]:


ma=baseline(2999)
base_line=pd.DataFrame(data=ma[0])

for i in range (1,3000):
    base_line=base_line.append(ma[i])
    
base_line


# In[96]:


base_line


# In[97]:


base_line.T.plot(legend=None)
plt.title('Baseline-corrected Spectra')


# In[100]:


df_center = base_line.T.apply(lambda x: x-x.mean())
df_centered=df_center.T
df_centered


# In[22]:


df_centered.loc[0].plot(legend=None)


# In[ ]:





# In[50]:


df_centered.T.plot(legend=None)
plt.title('Mean-centered Spectra')


# In[ ]:


pip install pybaselines


# In[ ]:


def baseline(n):
    
    b=[]
    for i in range(0,n+1):
        line=[]
        for j in range(1):
            a=df_centered.loc[i]-whittaker(i)[0]
            line.append(a)
        b.append(line)
    return b


# In[ ]:


ma=baseline(2999)
center_base=pd.DataFrame(data=ma[0])

for i in range (1,3000):
    center_base=center_base.append(ma[i])
    
center_base


# In[ ]:


center_base


# In[ ]:


pca=PCA()
pca.fit(df_centered)
pca_data=pca.transform(df_centered)


# In[103]:


per_var=np.round(pca.explained_variance_ratio_ *100, decimals=1)
labels=['PC'+str(x) for x in range(1,11)]

plt.bar(x=range(1,11), height=per_var[0:10], tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot (Eigendecomposition)')
plt.show()


# In[105]:


per_var=np.round(pca.explained_variance_ratio_ *100, decimals=1)
print(per_var[0:10])
print(len(per_var))


# In[106]:


labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
pca_df=pd.DataFrame(pca_data, columns=labels)
pca_df


# In[108]:


plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA with Eigendecomposition')
plt.xlabel('PC1-{0}%'.format(per_var[0]))
plt.ylabel('PC2-{0}%'.format(per_var[1]))

    
plt.show()


# In[178]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[184]:


x=pca_df.iloc[:,0:2]

kmeans = KMeans(20)
kmeans.fit(x)


# In[181]:


wcss=[]
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,11)
plt.plot(number_clusters,wcss)
plt.title('Elbow Curve (Eigendecomposition)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[185]:


identified_clusters = kmeans.fit_predict(x)

data_with_clusters = pca_df
plt.scatter(data_with_clusters['PC1'],data_with_clusters['PC2'],c=identified_clusters,cmap='rainbow')
plt.title('K-means Clustering (Eigendecomposition)')


# In[186]:


df_centered.loc[:,'cluster']=kmeans.labels_
df_centered


# In[190]:


X=df_centered.iloc[:,0:1000].values


# # Y = 3000 x 20 matrix

# In[210]:


z= np.zeros(shape=(df_centered.shape[0], 20))


# In[211]:


for i in range(df_centered.shape[0]):
    z[i,df_centered.loc[:,'cluster'][i]-1]=1


# In[214]:


z


# # PLS-DA

# In[217]:


from sklearn.cross_decomposition import PLSRegression


# In[218]:


X=df_centered.values

plsr = PLSRegression(n_components=2, scale=False)
plsr.fit(X, z)
scores = plsr.x_scores_


# In[219]:


scores


# In[225]:


dff=pd.DataFrame(scores)
dff.iloc[:,0]


# In[230]:


plt.scatter(dff.iloc[:,0], dff.iloc[:,1])


# In[231]:


for i in range(len(y_categorical)):
    plt.scatter(dff.iloc[:,0], dff.iloc[:,1], label = y_categorical[i])


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
norm = Normalize()
cmap = 'rainbow'
clrs = identified_clusters


fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(121, projection='3d')
ax.set_zlim((-3, 5)), ax.set_ylim((-2, 5)), ax.set_xlim((-3, 5))
ax.azim=-100
ax.scatter(pca_df.PC1, pca_df.PC2, pca_df.PC3, c=clrs, s=5**2)
plt.title('3 PCs (Eigendecomposition)')

xx, yy = np.meshgrid(np.arange(-15, 10, 1), np.arange(-50, 30, 1))
normal = np.array([0.96981815, -0.188338, -0.15485978])
z = (-normal[0] * xx - normal[1] * yy) * 1. / normal[2]
xx = xx + 5
yy = yy + 5
z = z + 5


#ax = fig.add_subplot(122, projection='3d')
#ax.azim=10
ax.elev=20

#ax.set_zlim((-0.7, 0.8)), ax.set_ylim((-0.7, 0.8)), ax.set_xlim((-0.7, 0.8))
#ax.plot_surface(xx, yy, z, alpha=.1)
plt.tight_layout()
#display(fig)


# In[ ]:


df_centered


# In[109]:


from scipy.signal import savgol_filter


# In[161]:


dfeat = savgol_filter(df_centered, 101, polyorder = 5, deriv=1)


# In[166]:


df_smooth=pd.DataFrame(dfeat)
df_smooth


# In[163]:


df_smooth.T.plot(legend=None)


# In[172]:


pca=PCA()
pca.fit(df_smooth)
pca_data2=pca.transform(df_smooth)


# In[173]:


labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
pca_df2=pd.DataFrame(pca_data2, columns=labels)
pca_df2


# In[177]:


plt.scatter(pca_df2.PC1, pca_df2.PC2)
plt.title('PCA with Eigendecomposition')
plt.xlabel('PC1-{0}%'.format(per_var[0]))
plt.ylabel('PC2-{0}%'.format(per_var[1]))


    
plt.show()


# In[176]:


per_var=np.round(pca.explained_variance_ratio_ *100, decimals=1)
labels=['PC'+str(x) for x in range(1,11)]

plt.bar(x=range(1,11), height=per_var[0:10], tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot (Eigendecomposition)')
plt.show()


# In[159]:


from statsmodels.multivariate.pca import PCA

pc2 = PCA(df_centered, ncomp=2, method='svd')

pc2.scores


# In[160]:


plt.scatter(pc2.factors.comp_0, pc2.factors.comp_1)
plt.title('PCA with NIPALS')

 
plt.show()


# In[152]:


df_centered[0:1000]

