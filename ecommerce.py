#%% 1. Import libraries
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks, metrics, losses, activations, regularizers, initializers, Sequential
import tensorboard
#%% 2. Defining Constant

print(os.getcwd())
CSV_PATH = os.path.join(os.getcwd(), "DATASET", "ecommerceDataset.csv")
SEED = 42
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 50 # for padding or truncating
EMBEDDING_DIM = 32
#%% Data Loading

df = pd.read_csv(CSV_PATH, names=["categories", 'product_and_description'])
df.head()

#%% EDA
print("################### BEFORE DATA CLEANING: #################")
print(f"There are {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("-----------------------------------------------\n")
print(f"There are {df['categories'].nunique()} categories: {df['categories'].unique()}\n")
print("-----------------------------------------------\n")
print("Distribution of categories:\n")
print(df['categories'].value_counts())
print("-----------------------------------------------\n")
print("Missing value:\n") 
print(df.isna().sum())
print("-----------------------------------------------\n")
print("duplicates:", df.duplicated().sum())
print("-----------------------------------------------\n")

##################### COMMENTS ####################
# There seem to be a lot of duplicates in the dataset.
###################################################
print("Distribution of duplicates:\n")
print(df[df.duplicated()].categories.value_counts())


#%% Data Visualisation

plt.figure(figsize=(10, 10))
sns.countplot(df['categories'])
plt.show()
# %% Data Preprocessing [Part A]

# remove duplicates
df.drop_duplicates(inplace=True)

# remove missing values
df.dropna(inplace=True)

# reinvestigate the dataset
print("################### AFTER DATA CLEANING: ###################")
print(f"There are {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("-----------------------------------------------\n")
print(f"There are {df['categories'].nunique()} categories: {df['categories'].unique()}\n")
print("-----------------------------------------------\n")
print("Distribution of categories:\n")
print(df['categories'].value_counts())
print("-----------------------------------------------\n")
print("Missing value:\n") 
print(df.isna().sum())
print("-----------------------------------------------\n")
print("duplicates:", df.duplicated().sum())
print("-----------------------------------------------\n")

#%% Data preprocessing [Part B]

# encoding the labels
le = LabelEncoder()
df['categories'] = le.fit_transform(df['categories'])

# verify the encoding is succesful
print(df["categories"].value_counts())

dict = {
    "0" : "Books",
    "1" : "Electronics",
    "2" : "Clothing & Accessories",
    "3" : "Household"
}

# %% Feature Selection
df_copy = df.copy()
label = df_copy.pop("categories")
features = df_copy

print(label.head())
print(features.head())
# %% Data Splitting
X_train, X_split, y_train, y_split = train_test_split(features, label, test_size=0.25, random_state=SEED, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, test_size=0.5, random_state=SEED, shuffle=True)

print("train: ", X_train.shape, y_train.shape)
print("val: ", X_val.shape, y_val.shape)
print("test: ", X_test.shape, y_test.shape)

# %% Data Preprocessing [Part C]

tokenizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=SEQUENCE_LENGTH)

#fit the state of the preprocessing layer to the dataset. This will cause the model to build an index of strings to integers
tokenizer.adapt(X_train )

# testing the tokenizer
print("Vocabulary Size: ", len(tokenizer.get_vocabulary()))
print("----------------------------------------------------\n")
sample_text = X_train.iloc[11:14]
sample_token = tokenizer(sample_text)
print("Sample text:")
print(sample_text)
print("\n----------------------------------------------------\n")
print("Sample token:")
print(sample_token)
print("\n----------------------------------------------------\n")
# this will print example of some words in the vocabulary represented by the tokens
print("Some representation of the vocabulary by tokens:")
print("1287 ---> ",tokenizer.get_vocabulary()[1287])
print(" 313 ---> ",tokenizer.get_vocabulary()[313])

# Apply the tokenizer to all the datasets
X_train = tokenizer(X_train)
X_val = tokenizer(X_val)
X_test = tokenizer(X_test)

print("train: ", X_train.shape, y_train.shape)
print("val: ", X_val.shape, y_val.shape)
print("test: ", X_test.shape, y_test.shape)
# %% Data Preprocessing [Part D]

# Embedding
embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

# callbacks using Tensorboard
# from tensorflow documentation:
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#%% Model Development

model = Sequential()
model.add(embedding)

# LSTM
model.add(layers.LSTM(64))

# dense layers
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(16, activation="relu"))

# output layer
model.add(layers.Dense(4, activation="softmax"))
model.summary()

#  compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x=X_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(X_val, y_val), 
          callbacks=[tensorboard_callback])