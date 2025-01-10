#%%
import json
import tensorflow as tf

def save_tokenizer(tokenizer):
    # get the configuration
    config = tokenizer.get_config()
    
    # get the vocab
    vocab = tokenizer.get_vocabulary()
    
    vectorizer_data = {
        "config": config,
        "vocab": vocab
    }
    
    # save to json
    with open("tokenizer.json", "w") as f:
        json.dump(vectorizer_data, f)
        
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        vectorizer_data = json.load(f)
        
    config = vectorizer_data["config"]
    vocab = vectorizer_data["vocab"]
    
    # rrecreate the tokenizer layer from the config
    tokenizer = tf.keras.layers.TextVectorization.from_config(config)
    tokenizer.set_vocabulary(vocab)
    
    return tokenizer
# %%
