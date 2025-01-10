#%% 1. Import libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import tensorboard
#%% 2. Defining Constant
print(os.getcwd())
CSV_PATH = os.path.join(os.getcwd(), "DATASET", "ecommerceDataset.csv")
SEED = 42
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

# %% Data Preprocessing

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

#%% Data preprocessing

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

# %%
# %%
