import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

class RelationExtractModel(object):
    def __init__(self, transformer_dir:str, pickle_path:str):
        self.model = self.load_model(transformer_dir, pickle_path, 512)

    @staticmethod
    def build_model(transformer:TFAutoModel, max_len:int=256):
        """
        Constructs a BERT model  
        https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
        """
        input_ids = Input(shape=(max_len, ), dtype=tf.int32)

        x = transformer(input_ids)[0]
        x = x[:, 0, :]
        x = Dense(1, activation='sigmoid', name='sigmoid')(x)

        # BUILD AND COMPILE MODEL
        model = Model(inputs=input_ids, outputs=x)
        model.compile(
            loss='binary_crossentropy', 
            metrics=['accuracy'], 
            optimizer=Adam(lr=1e-5)
        )
        return model

    def save_model(self, model, pickle_dir:str, transformer_dir:str):
        """
        Special function to save a keras model that uses a transformer layer
        """
        transformer = self.model.layers[1]
        transformer.save_pretrained(transformer_dir)
        sigmoid = self.model.get_layer('sigmoid').get_weights()
        pickle.dump(sigmoid, open(pickle_dir, 'wb'))

    def load_model(self, pickle_path:str, transformer_dir:str='transformer', max_len=512):
        """
        Special function to load a keras model that uses a transformer layer
        """
        transformer = TFAutoModel.from_pretrained(transformer_dir)
        model = self.build_model(transformer, max_len=max_len)
        sigmoid = pickle.load(open(pickle_path, 'rb'))
        model.get_layer('sigmoid').set_weights(sigmoid)
        
        return model