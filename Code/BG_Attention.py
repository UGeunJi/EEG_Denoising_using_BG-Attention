import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential
import os
from keras_self_attention import SeqSelfAttention



'''
def yhy_BG_Attention(datanum, embedding):
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, 1)))

    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(SeqSelfAttention())
    
    # decoding
    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(SeqSelfAttention())
    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(layers.Dense(1))

    model.summary()
    return model
'''

def BG_Attention(datanum, embedding=1):
    input_layer = layers.Input(shape=(datanum,1))

    # BiGRU layer
    gru_layer_encoding = layers.Bidirectional(layers.GRU(embedding, return_sequences=True))(input_layer)

    # Multi-Head Self-Attention layer (encoding)
    attention_layer_encoding = layers.MultiHeadAttention(2,512)(gru_layer_encoding,gru_layer_encoding)
    #attention_layer_encoding = MultiHeadSelfAttention()(gru_layer_encoding)
    #print(gru_layer_encoding.shape)
    #print(attention_layer_encoding.shape)
    adding1_encoding = layers.Add()([gru_layer_encoding,attention_layer_encoding])
    layernorm1_encoding = layers.LayerNormalization()(adding1_encoding)
    dense1_encoding = layers.Dense(embedding*2)(layernorm1_encoding)
    adding2_encoding = layers.Add()([layernorm1_encoding,dense1_encoding])
    layernorm2_encoding = layers.LayerNormalization()(adding2_encoding)


    #decoding

    gru_layer1_decoding = layers.Bidirectional(layers.GRU(embedding*2, return_sequences=True))(layernorm2_encoding) 

    # Multi-Head Self-Attention layer (decoding)
    attention_layer_decoding = layers.MultiHeadAttention(2,512)(gru_layer1_decoding,gru_layer1_decoding)
    
    adding1_decoding = layers.Add()([gru_layer1_decoding,attention_layer_decoding])
    layernorm1_decoding = layers.LayerNormalization()(adding1_decoding)
    dense1_decoding = layers.Dense(embedding*4)(layernorm1_decoding)
    adding2_decoding = layers.Add()([layernorm1_decoding,dense1_decoding])
    layernorm2_decoding = layers.LayerNormalization()(adding2_decoding)

    gru_layer2_decoding = layers.Bidirectional(layers.GRU(embedding*2, return_sequences=True))(layernorm2_decoding)

    flatten = layers.Flatten()(gru_layer2_decoding)
    output_layer = layers.Dense(datanum, activation=None)(flatten)


    # Output layer (e.g., classification or regression layer)
    #output_layer = layers.Dense(num_classes, activation='softmax')(attention_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

