# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:12:08 2022

@author: Michael Kraus
"""

import os
import sys
import pandas as pd
import seaborn as sns

# from supervised import AutoML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.offsetbox import AnchoredText


plt.style.use('seaborn-paper')
plot_bg_line_color = '#A3A3A3'
plot_bg_line_linewidth = 4
msize = 66 # markersize
LiWi=3
plt.rcParams.update({'font.size': 14}) # font size


dataDir = r"C:\Users\Michael Kraus\polybox\Shared\21_FS_BA_MA\MA_Max_Lorenzo\01_Datasets\generator"
os.chdir(dataDir)


#%% Read in Data
df = pd.read_csv("220427_KI-Futter.csv", delimiter=',')

# Targets Y: das soll der Nutzer eingeben können (wurden in der Umfrage von Befragten eingefüllt)
Y = df[['F-PLE','F-STI','F-COM','F-DYN','F-PUB']]

#: Features X: wurden benutzt, um die Bilder der Räume zu erzeugen
X = df[['LY-A','LY-B','SC-A','SC-B','OP']]

#: abgeleitete Features X: wurden aus den erzeugten Bilder errechnet
X_comp = df.drop(labels=['F-PLE','F-STI','F-COM','F-DYN','F-PUB','LY-A','LY-B','SC-A','SC-B','OP'], axis=1)



#%% Plot Data Stats

# default Hist Plot
df.hist(figsize=(12,8), bins=20, xlabelsize=12, ylabelsize=12)
sns.histplot(data=df)
plt.tight_layout()
plt.savefig('EDA_Histogramme_Aesthetics.png')

# # %%
sns.set(font_scale=1.0)

plt.figure(figsize=(12,12))
g = sns.heatmap(df.corr(),cmap=plt.cm.Reds,annot=True)
# plt.title('Heatmap zur Korrelation der Features und Zielgrößen', fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig('Heatmap_Aesthetics.png')           

# %%

sns.set_style('dark') #set theme
#Create subplot
fig,ax = plt.subplots(figsize=(16,10))
#Plot the swarm
chart5 = sns.swarmplot(x="LY-A", y="F-PLE", data=df)
#Label
chart5.set_ylabel('Pleasant',weight='bold',fontsize=13)
chart5.set_xlabel('LY-A', weight='bold',fontsize=13)

ax = sns.stripplot(x="LY-A", y="F-PLE", hue="F-STI", data=df)




#%% ################################################################################
############################# Kollinearität Check ##################################
####################################################################################
# https://towardsdatascience.com/multi-collinearity-in-regression-fe7a2c1467ea

#Compute VIF data for each independent variable
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["features"] = df.columns
vif["vif_Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
vif

# If the VIF value is higher than 10, it is usually considered to have a high correlation with other independent 
# variables. However, the acceptance range is subject to requirements and constraints. From the results, we can 
# see that most features are highly correlated with other independent variables and only two features can pass 
# the below 10 threshold.

#plot color scaled correlation matrix
fig,ax = plt.subplots(figsize=(7,7))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
plt.tight_layout()
plt.show()



