#%%
import json
import tensorflow as tf

def save_tokenizer(tokenizer, save_path):
    # get the configuration
    config = tokenizer.get_config()
    
    # get the vocab
    vocab = tokenizer.get_vocabulary()
    
    vectorizer_data = {
        "config": config,
        "vocab": vocab
    }
    
    # save to json
    with open(save_path, "w") as f:
        json.dump(vectorizer_data, f)
        
def load_tokenizer(load_path):
    with open(load_path, "r") as f:
        vectorizer_data = json.load(f)
        
    config = vectorizer_data["config"]
    vocab = vectorizer_data["vocab"]
    
    # rrecreate the tokenizer layer from the config
    tokenizer = tf.keras.layers.TextVectorization.from_config(config)
    tokenizer.set_vocabulary(vocab)
    
    return tokenizer
# %% ################## COMMENTS ##################

# the reason why we do it this way is because this layer doesnt have a to_json method
# so need to get the configuration and vocab manually
# and then recreate the layer when the load_tokenizer is call 

####################################################
