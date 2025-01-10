#%% 1. Import libraries
import os
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tokenizer_to_json import save_tokenizer, load_tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks, metrics, losses, activations, regularizers, initializers, Sequential
import tensorboard
#%% 2. Defining Constant

print(os.getcwd())
CSV_PATH = os.path.join(os.getcwd(), "DATASET", "ecommerceDataset.csv")
PATH_TO_SAVE_TOKENIZER = os.path.join(os.getcwd(), "saved_models", "tokenizer.json")
PATH_TO_SAVE_MODEL = os.path.join(os.getcwd(), "saved_models", "model.h5")
SEED = 42
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 50 # for padding or truncating
EMBEDDING_DIM = 32
BATCH_SIZE = 64
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

plt.figure(figsize=(10, 10))
sns.countplot(df['categories'])
plt.show()

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
X_train, X_split, y_train, y_split = train_test_split(features, label, test_size=0.2, random_state=SEED, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, test_size=0.5, random_state=SEED, shuffle=True)

print("train: ", X_train.shape, y_train.shape)
print("val: ", X_val.shape, y_val.shape)
print("test: ", X_test.shape, y_test.shape)

# %% Data Preprocessing [Part C]

############# COMMENTS ##############
# TextVectorizer already does the preprocessing for us
# in term of encoding, stripping punctuation, lowercase
################################################

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

# saving the tokenizer
# %% Data Preprocessing [Part D]

# Embedding
embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

# callbacks using Tensorboard
# from tensorflow documentation:
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# early_stopping
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
# regularizer [Fine Tuning]
l1 = regularizers.L1(l1=0.002)
l2 = regularizers.L2(l2=0.002)
l1l2 = regularizers.L1L2(l1=0.002, l2=0.002)
#%% Model Development

model = Sequential()

# SEQUENCE_LENGTH instead of None, because we know the length of the input
model.add(layers.Input(shape=(SEQUENCE_LENGTH,)))
model.add(embedding) # at ths point this is in 3d tensor

########## COMMENTS ######################
# This layer performs the same function as Dropout, however, it drops entire 1D feature maps instead of individual elements. 
# If adjacent frames within feature maps are strongly correlated (as is normally the case in early convolution layers) 
# then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. 
# In this case, SpatialDropout1D will help promote independence between feature maps and should be used instead.
# (From keras documentation)
##################################################

model.add(layers.SpatialDropout1D(0.2))

########## COMMENTS ######################
# This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.
# e.g. Initial shape : (4, 10, 128)
# Conv1D layer : keras.layers.Conv1D(32, 3, activation='relu') where 32 is number of filters, 3 is filter size.
# After Conv1D : (4, 8, 32)
###########################################
model.add(layers.Conv1D(64, kernel_size=3, activation="relu"))
# LSTM
model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(layers.Dense(64, activation="relu", kernel_regularizer=l2))
# output layer
model.add(layers.Dense(4, activation="softmax"))
model.summary()

#  compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x=X_train, 
          y=y_train, 
          epochs=15, 
          validation_data=(X_val, y_val), 
          batch_size=BATCH_SIZE,
          callbacks=[tensorboard_callback, early_stopping])

# %% Model Evaluation

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: ", accuracy)

# using f1 score to evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
f1 = f1_score(y_test, y_pred, average='macro') # use macro because all class are treated equally in term of importance
print("F1 score: ", f1)

print("Classification Report:\n", classification_report(y_test, y_pred))

# %% ################# Saving the model and tokenizer #################

# Here i already created a helper module for saving and loading the tokenizer

# ----------------------------------------------------------------
# [A] Tokenizer
save_tokenizer(tokenizer, PATH_TO_SAVE_TOKENIZER)

# let's try loading the tokenizer
loaded_tokenizer = load_tokenizer(PATH_TO_SAVE_TOKENIZER)

# lets just use the existing sample_text, 
# and then compare sample_token with sample_token_2 to ensure that they are the same
sample_token_2 = loaded_tokenizer(sample_text) 
if (sample_token.numpy() == sample_token_2.numpy()).all():
    print("The tokenizers are the same")
else:
    print("The tokenizers are not the same")

# ----------------------------------------------------------------
# [B] Model

model.save(PATH_TO_SAVE_MODEL)

# %% ################ SUMMARY #####################
# the model is overfitting no matter how i change the architecture. 
# However, this is the best val_loss that i can produce, maintaning a high test_accuracy and val_accuracu
##################################################