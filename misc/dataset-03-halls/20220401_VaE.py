# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:57:23 2022

@author: Michael Kraus
"""

# https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9
# https://medium.com/mlearning-ai/generating-artificial-tabular-data-with-an-encoder-decoder-5e4de9b4d02e
# https://colab.research.google.com/github/lucmos/DLAI-s2-2020-tutorials/blob/master/08/8_Variational_Autoencoders_(VAEs).ipynb

import os
import sys
import pandas as pd
import seaborn as sns

# from supervised import AutoML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import statsmodels.api as sm


plot_bg_line_color = '#A3A3A3'
plot_bg_line_linewidth = 4
msize = 66 # markersize
LiWi=3
plt.rcParams.update({'font.size': 14}) # font size


#dataDir = r"C:\Users\Michael Kraus\polybox\Shared\22_FS_BA_MA\03_Enrico\02_Datasets"
#computationDir = r"C:\Users\Michael Kraus\polybox\Shared\22_FS_BA_MA\03_Enrico\03_Computations\02_Regression"

#%% Read in Data
#os.chdir(dataDir)

# import raw data set
# df = pd.read_csv("Hallenrahmen_Datenbank.csv", delimiter=',')

# # drop irrelevant column
# df.drop(labels='Projektnummer', axis=1, inplace=True)


# # extract features of interest
# cols=['Spannweite','Stuetzenhoehe','Dachneigung','Rahmenabstand',
# 	  'Design_Last', 'Stuetzenprofil', 'Riegelprofil', 'Voutenprofil', 'Voutenlaenge',
# 	  'Voutenprofilfaktor', 'Voutenlaengenfaktor', 'Stuetzenausnutzung', 'Riegelausnutzung']
# # Ohne Schneelastzone
# df_red = pd.DataFrame(df[cols])

# os.chdir(computationDir)


# df_red.to_pickle("Hallen_Rohdaten.pkl")

df = pd.read_pickle(r'Hallen_Rohdaten.pkl')

#%%
X = df[["Spannweite", "Stuetzenhoehe", "Dachneigung", "Rahmenabstand", "Design_Last", "Stuetzenprofil", "Riegelprofil", "Voutenprofil", "Voutenprofilfaktor", "Voutenlaengenfaktor"]]

y = df[["Stuetzenausnutzung", "Riegelausnutzung"]]

df_red = X


#%% ########################################################################################
############################################ VAE ###########################################
############################################################################################
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import backend as K
from keras import Model
from tensorflow.random import set_seed

latent_dimension = 2
batch_size = 20
hidden_nodes = 16

input_encoder = Input(shape=(df_red.shape[1],), name="Input_Encoder")
batch_normalize1 = BatchNormalization()(input_encoder)
hidden_layer = Dense(hidden_nodes, activation="relu", name="Hidden_Encoding")(batch_normalize1)
batch_normalize2 = BatchNormalization()(hidden_layer)
z = Dense(latent_dimension, name="Mean")(batch_normalize2)
encoder = Model(input_encoder, z)


input_decoder = Input(shape=(latent_dimension,), name="Input_Decoder")
batch_normalize1 = BatchNormalization()(input_decoder)
decoder_hidden_layer = Dense(hidden_nodes, activation="relu", name="Hidden_Decoding")(batch_normalize1)
batch_normalize2 = BatchNormalization()(decoder_hidden_layer)
decoded = Dense(df_red.shape[1], activation="linear", name="Decoded")(batch_normalize2)
decoder = Model(input_decoder, decoded, name="Decoder")

encoder_decoder = decoder(encoder(input_encoder))
ae = Model(input_encoder, encoder_decoder)
ae.summary()

set_seed(2021)
ae.compile(loss="mean_squared_error", optimizer="adam")

#history = ae.fit(df_red, df_red, shuffle=True, epochs=5000, batch_size=20, validation_split=0.2, verbose=1).history
history = ae.fit(df_red, df_red, shuffle=True, epochs=2, batch_size=20, validation_split=0.2, verbose=1).history


sns.set(font_scale=2)
sns.set_style("white")

def model_analysis(history):
	train_loss = history["loss"]
	val_loss = history["val_loss"]
	t = np.linspace(1, len(train_loss), len(train_loss))

	plt.figure(figsize=(16, 12))
	plt.title("Mean squared error")
	sns.lineplot(x=t, y=train_loss, label="Train", linewidth=3)
	sns.lineplot(x=t, y=val_loss, label="Validation", linewidth=3)
	plt.xlabel("Epochs")

	plt.legend()
	plt.savefig("FirstNet.png", dpi=400)
	plt.show()
	print(f"Training MSE = {np.sqrt(train_loss[-1])}")
	print(f"Validation MSE = {np.sqrt(val_loss[-1])}")


model_analysis(history)


fig = plt.figure(figsize=(16, 12))
ax = plt.axes(projection='3d')
plt.title("Empirical distribution function z")
plt.xticks((-5, -4, -3, -2, -1, 0, 1, 2, 3, 4))
plt.yticks((-5, -4, -3, -2, -1, 0, 1, 2, 3, 4))
plt.hist2d(encoder.predict(df_red)[:,0],encoder.predict(df_red)[:,1], bins=30, density=True)
fig.tight_layout()
plt.savefig("DistInternal.png", dpi=400)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(encoder.predict(df_red)[:,0],encoder.predict(df_red)[:,1],
									  bins=30, range=[[-5, 5], [-5, 5]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

#%% What is the prediction on the first row of the training set?
ae.predict(df_red)[0,:]

df_red.iloc[0, :]       #The actual first observation is:

#%% Generating data

from statsmodels.distributions.empirical_distribution import ECDF
from numpy.random import uniform
from numpy.random import seed

ecdf = ECDF(encoder.predict(df)[:, 0])
plt.figure(figsize=(16, 12))
plt.title("Empirical distribution function z")
x = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
plt.yticks(np.linspace(0, 1, 11))
plt.xticks(x)
plt.grid()
plt.plot(x, ecdf(x), linewidth=3)
plt.savefig("EmpiricalDF.png", dpi=400)


from scipy.interpolate import interp1d


x = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
sample_edf_values_at_slope_changes = [ecdf(i) for i in x]
inverted_edf = interp1d(sample_edf_values_at_slope_changes, x)

# Using the decoder, I can generate 10000 new observations that resemble the original Iris data set:
N = 10000

output = pd.DataFrame(decoder(inverted_edf(uniform(0.02, 0.95, N))).numpy())
output.columns = df.columns
output["SepalLength"] = [round(x, 1) for x in output["SepalLength"]]
output["SepalWidth"] = [round(x, 1) for x in output["SepalWidth"]]
output["PetalLength"] = [round(x, 1) for x in output["PetalLength"]]
output["PetalWidth"] = [round(x, 1) for x in output["PetalWidth"]]
output["Species"] = output.iloc[:, 4:8].apply(lambda x: dict[np.argmax(x)], axis=1)
output.drop(["Species_Setosa", "Species_Versicolor", "Species_Virginica"], axis=1, inplace=True)
print(output.head().to_markdown())



