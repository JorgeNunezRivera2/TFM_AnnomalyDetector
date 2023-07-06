from keras import layers
from keras.layers import Input,Dense,Dropout,Embedding
import tensorflow as tf
from tensorflow.keras.models import load_model,Model

#MODEL_1
def get_model():
  input_layer = Input(shape=(1024))
  dense = Dense(256, activation='relu')(input_layer)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(64, activation='sigmoid')(dense)
  dense = Dense(32, activation='sigmoid')(dense)
  dense = Dense(54, activation='sigmoid')(dense)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(256, activation='sigmoid')(dense)
  output_layer = Dense(1024, activation='sigmoid')(dense)
  mal_model = Model(inputs=input_layer, outputs=output_layer)
  return mal_model

#MODEL 2
def get_model():
  input_layer = Input(shape=(1024))
  dense = Dense(512, activation='relu')(input_layer)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(32, activation='sigmoid')(dense)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(512, activation='sigmoid')(dense)
  output_layer = Dense(1024, activation='sigmoid')(dense)
  mal_model = Model(inputs=input_layer, outputs=output_layer)
  return mal_model

#MODEL 3
def get_model():
  input_layer = Input(shape=(1024))
  dense = Dense(512, activation='relu')(input_layer)
  drop = Dropout(0.2)(dense)
  dense = Dense(256, activation='sigmoid')(drop)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(64, activation='sigmoid')(dense)
  dense = Dense(128, activation='sigmoid')(dense)
  dense = Dense(256, activation='sigmoid')(dense)
  dense = Dense(512, activation='sigmoid')(dense)
  output_layer = Dense(1024, activation='sigmoid')(dense)
  mal_model = Model(inputs=input_layer, outputs=output_layer)
  return mal_model

#RECOMMENDER 1
def get_model():
  num_users=4000 #?
  embedding_size=64
  input_users = Input(shape=num_users)
  input_features=Input(shape=1024)
  embedding_users=Embedding(num_users,embedding_size)(input_users)
  dense_features = Dense(embedding_size, activation='relu')(input_features)
  dot_user_restaurant = tf.reduce_sum(embedding_users * dense_features, axis=1, keepdims=True)

  output_layer = Activation(activation='sigmoid')(dot_user_restaurant)
  mal_model = Model(inputs=(input_users,input_features), outputs=output_layer)
  return mal_model
