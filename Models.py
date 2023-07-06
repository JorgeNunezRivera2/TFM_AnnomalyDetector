import numpy as np

import tensorflow as tf
#from tensorflow import keras


from keras import layers, models,regularizers
from keras.models import Model
from keras.layers import Layer, Activation, Lambda
from Utils import clip_eps
from Utils import recommender_weights_with_augmentation_path,recommender_weights_path




def get_deep_features(train_generator, deep_features_path):
    # Pretrained model
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")  # , input_shape=(150, 150, 3))
    output = base_model.output
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    model = tf.keras.models.Model(base_model.inputs, output)
    deep_features = model.predict(train_generator)
    with open(deep_features_path, "wb") as f:
        np.save(f, deep_features)
    return deep_features

class RecommenderNet(tf.keras.Model):

    def __init__(self, num_users, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
        )
        self.restaurant_dense = tf.keras.layers.Dense(
            units=embedding_size
        )

    def call(self, inputs):
        user_vector = tf.squeeze(self.user_embedding(inputs[0]))
        restaurant_features = self.restaurant_dense(inputs[1])
        dot_user_restaurant = tf.reduce_sum(user_vector * restaurant_features, axis=1, keepdims=True)
        x = dot_user_restaurant
        return tf.nn.sigmoid(x)

class RecommenderNet2(tf.keras.Model):

    def __init__(self, num_users, embedding_size, **kwargs):
        super(RecommenderNet2, self).__init__(**kwargs)
        self.num_users = num_users
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1) #Estaba comentado
        self.restaurant_dense = tf.keras.layers.Dense(
            units=embedding_size
        )

    def call(self, inputs):
        user_vector = tf.squeeze(self.user_embedding(inputs[0]))
        user_bias=self.user_bias(inputs[0])
        if user_bias.shape.rank>1:
            user_bias = tf.squeeze(user_bias, axis=1)
        restaurant_features = self.restaurant_dense(inputs[1])
        dot_user_restaurant = tf.reduce_sum(user_vector * restaurant_features, axis=1, keepdims=True)
        x = dot_user_restaurant + user_bias
        return tf.nn.sigmoid(x)

class RecommenderNet3(tf.keras.Model):

    def __init__(self, num_users, embedding_size, **kwargs):
        super(RecommenderNet3, self).__init__(**kwargs)
        self.num_users = num_users
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1) #Estaba comentado
        self.restaurant_dense1 = tf.keras.layers.Dense(
            units=512
        )
        self.restaurant_dense2 = tf.keras.layers.Dense(
            units=256
        )
        self.restaurant_dense3 = tf.keras.layers.Dense(
            units=embedding_size
        )

    def call(self, inputs):  #Model 3
        user_vector = tf.squeeze(self.user_embedding(inputs[0]))
        user_bias=self.user_bias(inputs[0])
        if user_bias.shape.rank > 1:
            user_bias = tf.squeeze(user_bias, axis=1)
        restaurant_features = self.restaurant_dense1(inputs[1])
        restaurant_features = self.restaurant_dense2(restaurant_features)
        restaurant_features = self.restaurant_dense3(restaurant_features)
        dot_user_restaurant = tf.reduce_sum(user_vector * restaurant_features, axis=1, keepdims=True)
        x = dot_user_restaurant + user_bias
        return tf.nn.sigmoid(x)



'''
class AddedWeights(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddedWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer='zeros',
                                      trainable=True)
        super(AddedWeights, self).build(input_shape)

    def call(self, x, **kwargs):
        return x + self.kernel  # *(255./2.)

    def compute_output_shape(self, input_shape):
        return input_shape

'''
class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(1024, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class AnomalyDetector2(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector2, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(32, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1024, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector3(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector3, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation="relu",activity_regularizer=tf.keras.regularizers.l2(l2=0.001)),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1024, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ImageAnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(ImageAnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(1024, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

recommender_model=[None,RecommenderNet,RecommenderNet2,RecommenderNet3]
class RecommenderFromImageNet(tf.keras.Model):
    def __init__(self, num_users, embedding_size,augmentation,model_n, **kwargs):
        super(RecommenderFromImageNet, self).__init__()


        # clip entre [0, 255]
        self.clip = Lambda(lambda x: clip_eps(x, 0, 255))

        # Preprocess_input
        self.preprocess = Activation(tf.keras.applications.densenet.preprocess_input)
        self.preprocess.trainable = False

        # Pretrained model: get_deep_features()
        densenet_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet",input_shape=(224, 224, 3))
        self.densenet = densenet_model
        self.densenet.trainable = False
        self.globavepool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        # Call the recommender net
        recommender_net = recommender_model[model_n](num_users, embedding_size)
        if augmentation:
            recommender_net.load_weights(recommender_weights_with_augmentation_path[model_n])#.expect_partial()
        else:
            recommender_net.load_weights(recommender_weights_path[model_n])#.expect_partial()

        self.recommender = recommender_net
        self.recommender.trainable = False

    def call(self, inputs):
        # users = inputs[0]
        users = tf.convert_to_tensor(inputs[0])
        if users.shape.rank==1:
            users=tf.expand_dims(users,axis=1)
        images = inputs[1]

        # clip entre [0, 255]
        # x = clip_eps(x, 0, 255)
        x = self.clip(images) #prueba

        #x = self.preprocess(x) #prueba

        x = self.densenet(x) #x)
        x = self.globavepool(x)
        x = self.flatten(x)

        # x = self.recommender((K.constant(users), x))
        x = self.recommender([users, x])

        return x

class RecommenderFromMobileNet(tf.keras.Model):
    def __init__(self, num_users, embedding_size,augmentation,model_n, **kwargs):
        super(RecommenderFromMobileNet, self).__init__()


        # clip entre [0, 255]
        self.clip = Lambda(lambda x: clip_eps(x, 0, 255))

        # Preprocess_input
        self.preprocess = Activation(tf.keras.applications.mobilenet_v2.preprocess_input)
        self.preprocess.trainable = False

        # Pretrained model: get_deep_features()
        mobilenet_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",input_shape=(224, 224, 3))
        self.mobilenet = mobilenet_model
        self.densenet.trainable = False
        self.globavepool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        # Call the recommender net
        recommender_net = recommender_model[model_n](num_users, embedding_size)
        if augmentation:
            recommender_net.load_weights(recommender_weights_with_augmentation_path[model_n])#.expect_partial()
        else:
            recommender_net.load_weights(recommender_weights_path[model_n])#.expect_partial()

        self.recommender = recommender_net
        self.recommender.trainable = False

    def call(self, inputs):
        # users = inputs[0]
        users = tf.convert_to_tensor(inputs[0])
        if users.shape.rank==1:
            users=tf.expand_dims(users,axis=1)
        images = inputs[1]

        # clip entre [0, 255]
        # x = clip_eps(x, 0, 255)
        x = self.clip(images) #prueba

        #x = self.preprocess(x) #prueba

        x = self.mobilenet(x) #x)
        x = self.globavepool(x)
        x = self.flatten(x)

        # x = self.recommender((K.constant(users), x))
        x = self.recommender([users, x])

        return x

