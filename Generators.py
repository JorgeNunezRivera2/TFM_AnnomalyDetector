import time

from keras.applications.densenet import DenseNet121
#from keras.utils.data_utils import Sequence
from tensorflow import keras
import tensorflow as tf
#from tensorflow.keras.utils import Sequence
import numpy as np
import keras.layers as layers
import keras.models as models



class MyTrainingGenerator(tf.keras.utils.Sequence):

    def __init__(self, users, deep_features, ratings,
                 batch_size,
                 input_size=(224, 224, 3),
                 with_data_augmentation=False,
                 shuffle=True):
        self.indexes = np.arange(len(users))
        self.users = users
        self.deep_features = deep_features
        self.ratings = ratings
        self.batch_size = batch_size
        self.input_size = input_size
        self.with_data_augmentation = with_data_augmentation
        self.shuffle = shuffle
        # Data augmentation properties
        self.unique_users = np.unique(self.users)
        self.users_augmented = users
        self.deep_features_augmented = deep_features
        self.ratings_augmented = ratings
        self.positive_images_idx = np.where(ratings >= 0.5)
        self.negative_images_idx = np.where(ratings < 0.5)
        # Initialize data augmentation
        #self.initialize_data()
        self.on_epoch_end()


    def initialize_data(self):
        # Initialize original data
        self.users_augmented = self.users.copy()  # prueba
        self.deep_features_augmented = self.deep_features.copy()  # prueba
        self.ratings_augmented = self.ratings.copy()  # prueba
        # Data augmentation for each user
        if self.with_data_augmentation:
            #self.data_augmentation()
            #self.data_augmentation_indifferent_user()
            self.data_augmentation_equal_distribution()
        # Add a user for users not present in test set
        self.add_unknown_user()
        # Update indexes with the augmented number
        self.indexes = np.arange(len(self.users_augmented))

    def on_epoch_end(self):
        self.initialize_data()
        if self.shuffle:
            np.random.shuffle(self.indexes)
            #self.df.sample(frac=1).reset_index(drop=True)

    def add_unknown_user(self):
        n_unkown_images = 5
        rand_positives = np.random.choice(len(self.positive_images_idx[0]), size=n_unkown_images, replace=False)
        rand_negatives = np.random.choice(len(self.negative_images_idx[0]), size=n_unkown_images, replace=False)
        # Append positive images
        self.users_augmented = np.append(self.users_augmented, [[max(self.unique_users)+1]] * n_unkown_images, axis=0)
        self.deep_features_augmented = np.concatenate((self.deep_features_augmented, self.deep_features[rand_positives]))
        self.ratings_augmented = np.append(self.ratings_augmented, self.ratings[rand_positives], axis=0)
        # Append negative images
        self.users_augmented = np.append(self.users_augmented, [[max(self.unique_users)+1]] * n_unkown_images, axis=0)
        self.deep_features_augmented = np.concatenate((self.deep_features_augmented, self.deep_features[rand_negatives]))
        self.ratings_augmented = np.append(self.ratings_augmented, self.ratings[rand_negatives], axis=0)

    def data_augmentation_indifferent_user(self):
        positive_reviews = len(self.positive_images_idx[0])
        negative_reviews = len(self.negative_images_idx[0])
        if positive_reviews>negative_reviews:
            diff = positive_reviews - negative_reviews
            # Get random positions
            random_selection = np.random.choice(negative_reviews, size=diff, replace=True)#replace=False
            # Get the idx of the random positions in negative images
            random_idxs = self.negative_images_idx[0][random_selection]
        else:
            diff = negative_reviews - positive_reviews
            # Get random positions
            random_selection = np.random.choice(positive_reviews, size=diff, replace=True)#replace=False
            # Get the idx of the random positions in positive images
            random_idxs = self.positive_images_idx[0][random_selection]
        # Append new values
        if diff != 0:
            self.users_augmented = np.append(self.users_augmented, self.users[random_idxs], axis=0)
            self.deep_features_augmented = np.concatenate((self.deep_features_augmented, self.deep_features[random_idxs]))
            self.ratings_augmented = np.append(self.ratings_augmented, self.ratings[random_idxs], axis=0)


    def data_augmentation_equal_distribution(self):
        total_reviews=10000
        for i in range(0,4):
            reviews=np.where(self.ratings == 0.25*i)[0]
            diff = total_reviews - len(reviews)
            # Get random positions
            random_selection = np.random.choice(len(reviews), size=diff, replace=True)#replace=False
            # Get the idx of the random positions
            random_idxs = reviews[random_selection]
            # Append new values
            if diff != 0:
                self.users_augmented = np.append(self.users_augmented, self.users[random_idxs], axis=0)
                self.deep_features_augmented = np.concatenate((self.deep_features_augmented, self.deep_features[random_idxs]))
                self.ratings_augmented = np.append(self.ratings_augmented, self.ratings[random_idxs], axis=0)

    def data_augmentation(self):
        start_time=time.time()
        for user in self.unique_users:
            user_idxs = np.where(self.users == user)
            positive_reviews = 0
            negative_reviews = 0
            for idx in user_idxs[0]:
                if (self.ratings[idx][0] >= 0.5):
                    positive_reviews += 1
                else:
                    negative_reviews += 1
            if positive_reviews > negative_reviews:
                diff = positive_reviews - negative_reviews
                # Get random positions
                random_selection = np.random.choice(len(self.negative_images_idx[0]) - negative_reviews, size=diff, replace=True)#replace=False
                # Get the idx of the random positions in the images without the users ones
                random_idxs = np.setdiff1d(self.negative_images_idx[0], user_idxs[0])[random_selection]
            else:
                diff = negative_reviews - positive_reviews
                # Get random positions
                random_selection = np.random.choice(len(self.positive_images_idx[0]) - positive_reviews, size=diff, replace=True)#replace=False
                # Get the idx of the random positions in the images without the users ones
                random_idxs = np.setdiff1d(self.positive_images_idx[0], user_idxs[0])[random_selection]
            # Append new values
            if diff != 0:
                self.users_augmented = np.append(self.users_augmented, [[user]] * diff, axis=0)
                self.deep_features_augmented = np.concatenate((self.deep_features_augmented, self.deep_features[random_idxs]))
                self.ratings_augmented = np.append(self.ratings_augmented, self.ratings[random_idxs], axis=0)
        end_time = time.time()
        print(f"elapsed time{end_time-start_time}")
        print(f"data size: {len(self.ratings_augmented)}")

    def __getitem__(self, index):
        batches = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        users_batch = self.users_augmented[batches]
        deep_features_batch = self.deep_features_augmented[batches]
        ratings_batch = self.ratings_augmented[batches]
        return [users_batch, deep_features_batch], ratings_batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

class MyTestingGenerator(tf.keras.utils.Sequence):

    def __init__(self, users, deep_features, ratings,
                 training_users,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True):
        self.indexes = np.arange(len(users))
        self.users = users
        self.deep_features = deep_features
        self.ratings = ratings
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        # Data augmentation properties
        self.unique_users = np.unique(self.users)
        self.training_users = training_users
        # self.users_augmented = [user if user in training_users else [max(training_users)+1] for user in users]
        self.users_augmented = np.apply_along_axis(self.transform_unknown_user, 1, users)
        self.deep_features_augmented = deep_features
        self.ratings_augmented = ratings
        self.positive_images_idx = np.where(ratings > 0.5)
        self.negative_images_idx = np.where(ratings <= 0.5)
        # Initialize data augmentation
        self.on_epoch_end()

    def transform_unknown_user(self, users):
        '''unknown_user = []
        for user in users:
            if user in self.training_users:
                unknown_user.append(user)
            else:
                unknown_user.append([max(self.training_users) + 1])
        return unknown_user'''
        if users in self.training_users:
            return users
        else:
            return [max(self.training_users) + 1]

    def on_epoch_end(self):
        # Update indexes with the augmented number
        self.indexes = np.arange(len(self.users_augmented))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batches = self.indexes[index * self.batch_size: min(len(self.indexes),(index + 1) * self.batch_size)]
        users_batch = self.users_augmented[batches]
        deep_features_batch = self.deep_features_augmented[batches]
        ratings_batch = self.ratings_augmented[batches]
        return [users_batch, deep_features_batch], ratings_batch

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

class AdversarialAttackGenerator(tf.keras.utils.Sequence):

    def __init__(self, users, image, ratings,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=False):
        self.indexes = np.arange(len(users))

        self.users = users
        self.images = np.array([image] * len(users))
        self.ratings = np.array([1.0] * len(users))
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        # Initialize data augmentation
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batches = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        users_batch = self.users[batches]
        images_batch = self.images[batches]
        ratings_batch = self.ratings[batches]
        # ratings_batch = self.ratings.reindex(batches)
        return [users_batch, images_batch], ratings_batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

class UserImageSequence(tf.keras.utils.Sequence):
    def __init__(self, users, images,batch_size):
        self.users = tf.cast(users,tf.int32)
        self.images = images
        self.index = -1
        self.batch_size=batch_size
        self.total_length=len(self.users)

    def __getitem__(self, item):
        length = min(self.batch_size, len(self.users) - item * self.batch_size)
        return [self.users[self.batch_size*item:self.batch_size*item + length], self.images.__getitem__(item)[0]]

    def __len__(self):
        # SOLO 64 imagenes
        return int(len(self.users) /self.batch_size) + 1

    def get_images(self):
        images = np.array(self.images.__getitem__(0)[0])
        for i in range(1, self.__len__()):
            images = np.append(images, self.images.__getitem__(i)[0], 0)
        return images

    def get_single_item(self, item):

        return self.users[item], np.array([self.images.__getitem__(int(item/self.batch_size))[0][item%self.batch_size]])

    def next(self):
        self.index = self.index + 1
        if (self.index >= len(self.users)):
            return None
        return (self.get_single_item(self.index - 1))

class UserImageRatingSequence(tf.keras.utils.Sequence):
    def __init__(self, users, images,ratings,batch_size):
        self.users = users
        self.images = images
        self.ratings=ratings
        self.index = 0
        self.batch_size=batch_size
        self.total_length = len(self.users)

    def __getitem__(self, item):
        length = min(self.batch_size, len(self.users) - item * self.batch_size)
        return ((self.users[self.batch_size*item:self.batch_size*item + length], self.images.__getitem__(item)[0]),self.ratings[self.batch_size*item:self.batch_size*item + length])

    def __len__(self):
        # SOLO 64 imagenes
        return int(np.ceil(len(self.users) / self.batch_size))

    def get_images(self):
        images = np.array(self.images.__getitem__(0)[0])
        for i in range(1, self.__len__()):
            images = np.append(images, self.images.__getitem__(i)[0], 0)
        return images

    def get_single_item(self, item):
        return self.users[item], np.array([self.images.__getitem__(int(item /self.batch_size))[0][item % self.batch_size]]),self.ratings[item]

    def next(self):
        ret=self.get_single_item(self.index)
        self.index = self.index + 1
        if self.index >= len(self.users):
            self.index=0
        return ret
