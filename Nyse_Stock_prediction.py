#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
import os
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',200)
from typing import *
import json


# In[2]:


with open("Input.json") as reqirements:
    config=json.load(reqirements)


# In[3]:


file1=r'C:\Users\Administrator\Desktop\Project_files\Trade_path\Trades_Calculation_0m-3m.csv'
df1=pd.read_csv(file1)
df1.head(3)


# ### filtering data from json given requirements

# In[4]:


config["DataFrameFilters"]


# In[5]:


filterdata= df1[(df1["EntrySeconds"] <= int(config["DataFrameFilters"][0]["Maximum"])) &
               (df1["PositionType"] == config["DataFrameFilters"][1]["Equals"]) &
               (df1["Float"].between(int(config["DataFrameFilters"][2]["Minimum"]) ,int(config["DataFrameFilters"][2]["Maximum"]), inclusive = False)) &
               (df1["EntryPrice"].between(int(config["DataFrameFilters"][3]["Minimum"]) ,int(config["DataFrameFilters"][3]["Maximum"]), inclusive = False))].reset_index(drop=True)


# In[6]:


filterdata.head()


# ### replace nan value from Fund52WeekChange which is given as ('∞%') to 0% as change in a stock cannot be in infinity 

# In[7]:


#counting to numner nan value is present
print(f"the total no of nan value in Fund52WeekChange  is :  {filterdata['Fund52WeekChange'].str.count('∞%').sum()}.")


# In[8]:


# As every data point is important here we cant drop the rows so its been replaced by 0% as specified in requirement documnet
filterdata['Fund52WeekChange']=filterdata['Fund52WeekChange'].replace('∞%',"0%")
filterdata['Fund52WeekChange']=filterdata['Fund52WeekChange'].str[:-1].astype(float)


# ### filtering columns as specified in the requirement document

# In[9]:


filterdata=filterdata.loc[:,['EntrySeconds','Float','FundShortRatio','FundShortPercofFloat','Vol1MinRatioMaxPD123','Vol1MinRatioMaxPD123_TotalVolume','AllExchangesVolume','IntradayCurrentMarketGapPerc','Fund52WeekChange','PercentageProfit','WinTrade']]


# In[10]:


filterdata.head(2)


# ### Finding corelation Matrix 

# In[11]:


filterdata.corr()


# In[12]:


corrmat =filterdata.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmin=-1, vmax=1, square=True,cmap="YlGnBu",annot=True);


# #### we can clearly see the correlation between the two var i.e. AllExchange vs Vol1MinRatioMaPD123_TotalVolume. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these two variables, we can conclude that they give almost the same information so multicollinearity really occurs.

# In[13]:


sns.regplot(data=filterdata,x='AllExchangesVolume',y='Vol1MinRatioMaxPD123_TotalVolume')


# #### there is no difference between the two varibale ,One Variable would be suffice to explain other that makes other Variable redundant for our model

# In[14]:


#dropping a variable to avoid multicollinearity
filterdata.drop(['AllExchangesVolume'],1,inplace=True)


# ### looking at the collinearity between the independent variable VS two dependent variable we have i.e. [ WINTRADE , PERCENTPROFIT]

# In[15]:


filterdata[filterdata.columns[:]].corr()['WinTrade'][:].sort_values(ascending=False)


# In[16]:


filterdata[filterdata.columns[:]].corr()['PercentageProfit'][:].sort_values(ascending=False)


# In[17]:


filterdata.describe()


# #### from this Descriptive analysis we can clearly see that the scales of the varible are differing a long so it would be a problem non tree based model and for dimentionality reduction techniques 

# In[18]:


from sklearn.preprocessing import RobustScaler
rs=RobustScaler()


# In[19]:


filterdata.loc[:,:'Fund52WeekChange']=rs.fit_transform(filterdata.loc[:,:'Fund52WeekChange'])


# In[20]:


filterdata.tail()


# In[21]:


corrmat =filterdata.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmin=-1, vmax=1, square=True,cmap="YlGnBu",annot=True);


# In[22]:


corrmat =filterdata.iloc[:,:-2].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmin=-1, vmax=1, square=True,cmap="YlGnBu",annot=True);


# In[23]:


filterdata=filterdata.reset_index(drop=True)


# In[24]:


train=filterdata.iloc[:926,:]
test=filterdata.iloc[926:,:]


# In[25]:


train.head(5)


# In[26]:


test.tail(5)


# ## Checking for imbalance data

# In[27]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['WinTrade'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('WinTrade')
ax[0].set_ylabel('')
sns.countplot('WinTrade',data=train,ax=ax[1])
ax[1].set_title('WinTrade')
plt.show()


# #### by lookking at the stats our target var are somewhat fairly distributed 

# In[28]:


sns.pairplot(data=filterdata,y_vars=['PercentageProfit'],x_vars=['EntrySeconds','Float','FundShortRatio','FundShortPercofFloat','Vol1MinRatioMaxPD123','Vol1MinRatioMaxPD123_TotalVolume','IntradayCurrentMarketGapPerc','Fund52WeekChange'])


# In[31]:


sns.set(rc={"figure.figsize": (20, 8)})
filterdata.drop(['PercentageProfit','WinTrade'],1).plot.kde()


# ### now we have to Find feature importance using various tree based and non tree based models

# In[32]:


#finding feature importance using Rf
train_copy=train.drop(['PercentageProfit'],1)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=150, max_depth=3,min_samples_leaf=6,max_features=0.3,n_jobs=-1,random_state=2)
rf.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])
feature=train_copy.drop(['WinTrade'],1).columns.values
print('--traing done___')


# In[34]:


x,y=(list(x) for x in zip(*sorted(zip(rf.feature_importances_,feature)
                                   ,reverse=False)))

trace2 = go.Bar(
    x=x,
    y=y,
    marker=dict(
          color=x,
           colorscale='Viridis',
          reversescale= True),
    name='RF feature imp',
    orientation='h'
)

layout=dict(
      title='Barplot of fI',
    width=900, height=600,
     yaxis=dict(
          showgrid=False,
           showline=False,
          showticklabels=True))
data=[trace2]

fig1=go.Figure(data=data)
fig1['layout'].update(layout)
py.iplot(fig1,filename='bars')


# In[35]:


#using xgboost


# In[38]:


import xgboost as xgbp
xgb=xgbp.XGBClassifier(objective='binary:logistic')
xgb.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])
feature=train_copy.drop(['WinTrade'],1).columns.values
print('--traing done___')


# In[39]:


x,y=(list(x) for x in zip(*sorted(zip(xgb.feature_importances_,feature)
                                   ,reverse=False)))

trace2 = go.Bar(
    x=x,
    y=y,
    marker=dict(
          color=x,
           colorscale='Viridis',
          reversescale= True),
    name='XGBOOST feature imp',
    orientation='h'
)

layout=dict(
      title='Barplot of fI',
    width=900, height=600,
     yaxis=dict(
          showgrid=False,
           showline=False,
          showticklabels=True))
data=[trace2]

fig1=go.Figure(data=data)
fig1['layout'].update(layout)
py.iplot(fig1,filename='bars')


# In[40]:


#using GbM
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(n_estimators=150,max_depth=3,min_samples_leaf=3,max_features=0.3,learning_rate=0.05,subsample=0.4,random_state=2)
gb.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])
feature=train_copy.drop(['WinTrade'],1).columns.values
print('--traing done___')


# In[41]:


x,y=(list(x) for x in zip(*sorted(zip(gb.feature_importances_,feature)
                                   ,reverse=False)))

trace2 = go.Bar(
    x=x,
    y=y,
    marker=dict(
          color=x,
           colorscale='Viridis',
          reversescale= True),
    name='GB feature imp',
    orientation='h'
)

layout=dict(
      title='Barplot of GB',
    width=900, height=600,
     yaxis=dict(
          showgrid=False,
           showline=False,
          showticklabels=True))
data=[trace2]

fig1=go.Figure(data=data)
fig1['layout'].update(layout)
py.iplot(fig1,filename='bars')


# In[42]:


#using gini


# In[43]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[44]:


params={ 'class_weight':['balanced'], 
        'criterion':['gini'],
        'max_depth':[None,2,3,5,],
            'min_samples_leaf':[1,2,5,], 
            'min_samples_split':[2,5,]
       }


# In[45]:


clf=DecisionTreeClassifier()
random_search=RandomizedSearchCV(clf,cv=10,
                                 param_distributions=params,
                                 scoring='roc_auc',
                                 n_iter=10
                                    )
random_search.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])


# In[46]:


dtree=random_search.best_estimator_
dtree.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])


# In[47]:


from sklearn import tree

dotfile = open("mytree.dot", 'w')

tree.export_graphviz(dtree,out_file=dotfile,
                     feature_names=train_copy.drop(['WinTrade'],1).columns,
                    class_names=["0","1"],
                     proportion=True)
dotfile.close()


# ##### top 3 for dt is Vol1MinRatioMaxPD123 ,Fund52WeekChange ,Float

# #####  mytree.dot file might have been generated in to working directory ( copy and paste in the link to see the tree diagram)  dt can be viewed on : http://webgraphviz.com

# In[48]:


#by using pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[49]:


X=train_copy.iloc[:,:-1].copy()


# In[50]:


X=scale(X)


# In[51]:


pca = PCA(n_components=8)


# In[52]:


pca.fit(X)


# In[53]:


pca.components_.shape


# In[54]:


var= pca.explained_variance_ratio_

print(var)


# In[55]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)


# In[56]:


#showing imp feature of  pca1
pca.components_[0]


# In[57]:


print(abs( pca.components_ ))


# In[58]:


loadings=pca.components_[0]
loadings


# In[59]:


list(zip(train_copy.iloc[:,:-1].columns,loadings))


# #### by observing all the following classifiers manually  'Vol1MinRatioMaxPD123','FundshortRatio' and a tie between 'float' and 'FundshortPercofFloat'  came out to be the top3 varible

# ### trainig data using ensemble method
# 

# In[65]:


from sklearn.svm import SVC


# In[66]:


classifier={'RandomForest':RandomForestClassifier(),
            'GradientBoosting':GradientBoostingClassifier(),
            'XGBClassifier':xgbp.XGBClassifier(objective='binary:logistic'),
            'Decisiontree':DecisionTreeClassifier(),
            
    }


# In[67]:


from sklearn.model_selection import cross_val_score


# In[68]:


for key, classifier in classifier.items():
    classifier.fit(train_copy.drop(['WinTrade'],1),train_copy['WinTrade'])
    training_score = cross_val_score(classifier,train_copy.drop(['WinTrade'],1),train_copy['WinTrade'], cv=5)
    feature=train_copy.drop(['WinTrade'],1).columns.values
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of",round(training_score.mean(), 2) * 100, "% accuracy score")
    print('--traing done___')


# In[69]:


x,y=(list(x) for x in zip(*sorted(zip(classifier.feature_importances_,feature)
                                   ,reverse=False)))

trace2 = go.Bar(
    x=x,
    y=y,
    marker=dict(
          color=x,
           colorscale='Viridis',
          reversescale= True),
    name='GB feature imp',
    orientation='h'
)

layout=dict(
      title='Barplot of FI using Ensemble',
    width=900, height=600,
     yaxis=dict(
          showgrid=False,
           showline=False,
          showticklabels=True))
data=[trace2]

fig1=go.Figure(data=data)
fig1['layout'].update(layout)
py.iplot(fig1,filename='bars')


# #### the top 3 factors using ensemble are 'Vol1MinRatioMaxPD123','EntrySeconds',''Vol1MinRatioMaxPD123_totalvolume'

# In[70]:


from sklearn.manifold import TSNE
import time


# In[71]:


X=train_copy.drop(['WinTrade'],1)
y=train_copy['WinTrade']

t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))


# #### To check if we have a good cluster of win and loss trade or not?

# In[72]:


import matplotlib.patches as mpatches
f, (ax1) = plt.subplots(ncols=1, figsize=(24,6))
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='Loss trade')
red_patch = mpatches.Patch(color='#AF0000', label='Win Trade')
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Loss trade', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Win Trade', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# ### as we got top 3 variables we will train our model

# In[73]:


top3=filterdata.loc[:,['Vol1MinRatioMaxPD123','EntrySeconds','Vol1MinRatioMaxPD123_TotalVolume','PercentageProfit','WinTrade']]


# In[74]:


top3.head()


# In[75]:


train=top3.iloc[:926,:]
test=top3.iloc[926:,:]


# In[76]:


test.shape


# In[77]:


train_r=train.drop(['WinTrade'],axis=1)
test_r=test.drop(['WinTrade'],axis=1)


# In[78]:


train_l=train.drop(['PercentageProfit'],1)
test_l=test.drop(['PercentageProfit'],1)


# In[79]:


test_l.shape


# In[80]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
params={ 'class_weight':[None,'balanced'], 
        'criterion':['entropy','gini'],
        'max_depth':[None,3,5,7,10,15,20],
            'min_samples_leaf':[1,2,5,10,15], 
            'min_samples_split':[1.0,2,5,10,15]
       }


# In[81]:


from sklearn.model_selection import GridSearchCV
clf=DecisionTreeClassifier()
grid_search=GridSearchCV(clf,cv=10,
                        param_grid=params,
                        scoring='roc_auc',
                         n_jobs=-1
                        )
                                    
grid_search.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])


# In[82]:


grid_search.best_estimator_


# In[83]:


dt_bst=grid_search.best_estimator_


# In[84]:


dtpred=dt_bst.predict(test_l.drop(['WinTrade'],1))


# In[85]:


accuracy_score(dtpred,test_l['WinTrade'])


# In[86]:


dt_mat=pd.DataFrame(list(zip(test_l['WinTrade'],dtpred)),columns=['real','predicted'])


# In[87]:


pd.crosstab(dt_mat['real'],dt_mat['predicted'])


# In[88]:


## GRID SEARCH TALKING ALOT OF TIME  IN MY SYSTEM SO I am  USING RANDOM SEARCH


# In[89]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()
param_dist = {"n_estimators":[100,200,300,500,700,1000],
              "bootstrap": [True, False],
              'class_weight':[None,'balanced'], 
                'criterion':['entropy','gini'],
                'max_depth':[None,2,4,5,10,15,20,30],
                'min_samples_leaf':[1,2,5,10,15,20], 
                'min_samples_split':[2,5,10,15,20]
                  }


# In[90]:


random=RandomizedSearchCV(rf,cv=10,param_distributions=param_dist,
                        scoring='roc_auc',n_iter=10
                        )
                                    
random.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])


# In[91]:


random.best_estimator_


# In[92]:


rf_bst=random.best_estimator_


# In[93]:


rf_pred=rf_bst.predict(test_l.drop(['WinTrade'],1))


# In[94]:


accuracy_score(rf_pred,test_l['WinTrade'])


# In[95]:


rf_mat=pd.DataFrame(list(zip(test_l['WinTrade'],rf_pred)),columns=['real','predicted'])


# In[96]:


pd.crosstab(rf_mat['real'],rf_mat['predicted'])


# In[ ]:





# In[97]:


from sklearn.ensemble import GradientBoostingClassifier
gbm_params={'n_estimators':[40,50,100,200,300,500,700],            
           'learning_rate': [0.005,0.01,.05,0.1,0.4,0.8,1],
            'max_depth':[1,2,3,4,5,6],
            'subsample':[0.2,0.4,0.5,0.8,1],
            'max_features':[1,2,3]
           }
gbm=GradientBoostingClassifier()


# In[98]:


random_gbm=RandomizedSearchCV(gbm,scoring='roc_auc',param_distributions=gbm_params,
                                 cv=10,n_iter=15,
                                 n_jobs=-1)


# In[99]:


random_gbm.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])


# In[100]:


random_gbm.best_estimator_


# In[101]:


rn_gbbst=random_gbm.best_estimator_


# In[102]:


gb_pred=rn_gbbst.predict(test_l.drop(['WinTrade'],1))


# In[103]:


accuracy_score(gb_pred,test_l['WinTrade'])


# In[104]:


gb_mat=pd.DataFrame(list(zip(test_l['WinTrade'],gb_pred)),columns=['real','predicted'])


# In[105]:


pd.crosstab(gb_mat['real'],gb_mat['predicted'])


# In[106]:


import xgboost as xgbp
xgb=xgbp.XGBClassifier(objective='binary:logistic')
xgb_params = {  
                "learning_rate":[0.005,0.01,0.05,0.1,0.3,0.5,0.7],
                'max_delta_step':[0,1,3,6,10],
                "max_depth": [2,3,4,5,6,7,8],
                "min_child_weight":[1,2,5,7,10,12],
                "max_delta_step":[0,1,2,5,7,10],
                "subsample":[i/10.0 for i in range(5,10)],
                "colsample_bytree":[i/10.0 for i in range(5,10)],
                "reg_lambda":[1e-5, 1e-2, 0.1, 1, 100], 
                'reg_alpha':[i/10 for i in range(0,50)],
                "scale_pos_weight":[1,2,3,4,5,6,7,8,9],
                "n_estimators":[100,300,500,700,1000]
             }
random_search=RandomizedSearchCV(xgb,n_jobs=-1,cv=10,n_iter=10,scoring='roc_auc',
                                 param_distributions=xgb_params)


# In[107]:


random_search.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])


# In[108]:


random_search.best_estimator_


# In[109]:


xg_bst=random_search.best_estimator_


# In[110]:


xg_pred=xg_bst.predict(test_l.drop(['WinTrade'],1))


# In[111]:


accuracy_score(xg_pred,test_l['WinTrade'])


# In[112]:


xg_mat=pd.DataFrame(list(zip(test_l['WinTrade'],xg_pred)),columns=['real','predicted'])


# In[113]:


pd.crosstab(xg_mat['real'],xg_mat['predicted'])


# In[114]:


#combination of all


# In[115]:


classifiers={'Decisiontree':DecisionTreeClassifier(),
            'RandomForest':RandomForestClassifier(),
            'GradientBoosting':GradientBoostingClassifier(),
            'XGBClassifier':xgbp.XGBClassifier(objective='binary:logistic')
            }


# In[116]:


for key, classifier in classifiers.items():
    classifier.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])
    training_score = cross_val_score(classifier,train_l.drop(['WinTrade'],1),train_l['WinTrade'], cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of",round(training_score.mean(), 2) * 100, "% accuracy score")
    


# In[117]:


from sklearn.ensemble import VotingClassifier
model1=DecisionTreeClassifier(random_state=1)
model2=RandomForestClassifier(random_state=1)
model3=GradientBoostingClassifier(random_state=1)
model4=xgbp.XGBClassifier(objective='binary:logistic',random_state=1)
model = VotingClassifier(estimators=[('dt', model1), ('rf', model2),('gbm',model3),('xg',model4)], voting='hard')
model.fit(train_l.drop(['WinTrade'],1),train_l['WinTrade'])
cb=model.predict(test_l.drop(['WinTrade'],1))


# In[118]:


accuracy_score(cb,test_l['WinTrade'])


# In[119]:


xc_mat=pd.DataFrame(list(zip(test_l['WinTrade'],cb)),columns=['real','predicted'])


# In[120]:


pd.crosstab(xc_mat['real'],xc_mat['predicted'])


# In[ ]:




