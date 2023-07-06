from enum import Enum

import pandas as pd
import numpy as np
import os
import logging
import pickle
import requests
import shutil
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras_preprocessing.image import array_to_img, img_to_array, load_img, save_img

EMBEDDING_SIZE=64

def create_logger(module_name):
    # create logger
    logging.basicConfig(filename='output.log', format='%(asctime)s - %(name)s [%(levelname)s]: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s]: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


logger = create_logger(__name__)

class Test_set(Enum):
    FULL=1,
    WORST=2,
    BALANCED=3

#################################################################################
################################## PATHS ##########################################
#################################################################################
dirname = 'data'
weightsdir = 'weights'
featuresdir = 'features'
imagedir='images'
dfdir='dataframes'

#data_path = os.path.join(dirname, 'data')
####################################### IMAGES ###########################################

original_images_path = os.path.join(dirname,imagedir, 'images_shared')
scaled_images_path = os.path.join(dirname,imagedir, 'images_scaled')

cl_densenet_worst_images_path=os.path.join(dirname,imagedir,'cleverhans_densenet_worst_images')
cl_densenet_test_images_path=[
    "",
    os.path.join(dirname,imagedir,'cleverhans_densenet_test_images_model1'),
    os.path.join(dirname, imagedir, 'cleverhans_densenet_test_images_model2'),
    os.path.join(dirname, imagedir, 'cleverhans_densenet_test_images_model3')
]

    ###################################### FEATURES ############################################
deep_features_path = os.path.join(dirname, featuresdir,'clean_deep_features.npy')
test_features_path= os.path.join(dirname, featuresdir,'test_deep_features.npy')
worst_features_path= os.path.join(dirname, featuresdir,'worst_deep_features.npy')
balanced_test_features_path= os.path.join(dirname, featuresdir,'balanced_test_deep_features.npy')

cl_recommender_test_features_path=[
    "",
    os.path.join(dirname,featuresdir,'cl_recommender_test_features_model1.npy'),
    os.path.join(dirname, featuresdir, 'cl_recommender_test_features_model2.npy'),
    os.path.join(dirname, featuresdir, 'cl_recommender_test_features_model3.npy')
    ]

cl_recommender_worst_features_path=os.path.join(dirname,featuresdir,'cl_recommender_worst_features.npy')
cl_densenet_test_features_path=[
    "",
    os.path.join(dirname,featuresdir,'cl_densenet_test_features_model1.npy'),
    os.path.join(dirname, featuresdir, 'cl_densenet_test_features_model2.npy'),
    os.path.join(dirname, featuresdir, 'cl_densenet_test_features_model3.npy'),
    ]
cl_densenet_worst_features_path=os.path.join(dirname,featuresdir,'cl_densenet_worst_features.npy')


##################################### WEIGHTS #################################################

recommender_weights_path=[
    "",
    os.path.join(dirname, weightsdir,'recommender_weights1','recommender_weights1'),
    os.path.join(dirname, weightsdir,'recommender_weights2','recommender_weights2'),
    os.path.join(dirname, weightsdir,'recommender_weights3','recommender_weights3')
]
recommender_weights_with_augmentation_path=[
    "",
    os.path.join(dirname,weightsdir,'recommender_with_augmentation_weights1','recommender_weights1'),
    os.path.join(dirname,weightsdir,'recommender_with_augmentation_weights2','recommender_weights2'),
    os.path.join(dirname,weightsdir,'recommender_with_augmentation_weights3','recommender_weights3')
]
annomaly_detector_weights_path=[
    "",
    os.path.join(dirname,weightsdir,'annomaly_detector_weights1','annomaly_detector_weights'),
    os.path.join(dirname,weightsdir,'annomaly_detector_weights2','annomaly_detector_weights'),
    os.path.join(dirname,weightsdir,'annomaly_detector_weights3','annomaly_detector_weights'),
]


###################################### DATAFRAMES ##########################################

original_database_path= os.path.join(dirname,dfdir, 'RVW')
worst_images_df_path=os.path.join(dirname,dfdir,'worst_images_df')
train_images_df_path=os.path.join(dirname,dfdir,'train_images_df')
test_images_df_path=os.path.join(dirname,dfdir,'test_images_df')
balanced_images_df_path=os.path.join(dirname,dfdir,'balanced_test_images_df')

#test_images_noise_path=os.path.join(dirname,'test_images_with_noise')
#ch_densenet_worst_test_images_noise_path=os.path.join(dirname,'cleverhans_densenet_worst_images')
#ch_densenet_test_images_noise_path=os.path.join(dirname,'cleverhans_densenet_test_images_model3')
#ch_mynet_test_images_noise_path=os.path.join(dirname,'cleverhans_mynet_test_images')

#deep_features_train_path=os.path.join(dirname,'deep_features_train.npy')
#cleaverhans_mynet_deep_features_path=os.path.join(dirname,'cl_mynet_deep_features.npy')
#clean_test_deep_features_path=os.path.join(dirname,'clean_test_deep_features.npy')

#recommender_direct_weights_path=os.path.join(dirname,'direct_recommender_weights','direct_recommender_weights')




def get_pkl_data(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)
    return df

def pkl_save(data,file):
    with open(file, 'wb') as f:
        pickle.dump(data,f)


def clip_eps(tensor, min_eps, max_eps):
    # Clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor, clip_value_min=min_eps, clip_value_max=max_eps)

def custom_loss(y_true, y_pred):
    return K.relu(y_true - y_pred)

def aa_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return 0.15

def sum_noise(image,noise):
    return array_to_img(img_to_array(image)+noise)

def load_add_noise_and_save(path,noise):
    image = load_img(path)
    image=image.resize((150,150))
    masked_image=sum_noise(image,noise)
    save_img(path,masked_image)

class SecondValidationSet(tf.keras.callbacks.Callback):
    def __init__(self,validation_set):
        super(SecondValidationSet,self).__init__()
        self.validation_set=validation_set
        self.history=[]

    def on_epoch_end(self, epoch, logs=None):
        res_validation = self.model.evaluate(self.validation_set,self.validation_set, verbose = 0)
        self.history.append(res_validation)
        #print(f" adversarial val loss: {res_validation}")

class SecondValidationSequence(tf.keras.callbacks.Callback):
    def __init__(self,validation_sequence):
        super(SecondValidationSequence,self).__init__()
        self.validation_sequence=validation_sequence
        self.history=[]

    def on_epoch_end(self, epoch, logs=None):
        res_validation = self.model.evaluate(self.validation_sequence, verbose = 0)
        self.history.append(res_validation[0])
        print(f" adversarial val loss: {res_validation[0]}")

def download_all_images(file,path):
    with open(file, 'rb') as f:
        df = pickle.load(f)
    count = 0
    #CREATE LIST OF FILES AND IDS
    file_list=list()
    for index, review in df.iterrows():
        if len(review['images']) > 0:
            for image_url in review['images']:
                image_name = str(review['reviewId']) + '_' + image_url['image_url_lowres'].split("/")[-1]
                file_list.append({'reviewId': review['reviewId'],'url': image_url['image_url_lowres'],'filename':image_name })
    i=1
    while(len(file_list)>0):
        print("Intento n " + str(i) + "Quedan " + str(len(file_list))+ "imágenes")
        for item in list(file_list):
            rId, url, filename = item.values()

            #print("Requesting image " + url)
            try:
                r = requests.get(url, stream=True,timeout=i+1)
            except requests.exceptions.RequestException as error:
            #    print("Error downloading " + url + ": " + str(error))
                continue
            # Check if the image was retrieved successfully
            if r.status_code == 200:
                count = count + 1
            #    print("Downloading " + url)
                r.raw.decode_content = True
                with open(os.path.join(path, filename), 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                file_list.remove(item)
            if(count%50==0):
               print("Downloaded " + str(count) + "images")
        i=i+1


def unpreprocess(image):
    mean =  [0.485,
     0.456,
     0.406]
    std = [0.229,
   0.224,
   0.225]
    image = ((image *std)+ mean) *255.0

    return image

class SecondValidationSequenceManual(tf.keras.callbacks.Callback):
    def __init__(self,validation_sequence):
        super(SecondValidationSequenceManual,self).__init__()
        self.validation_sequence=validation_sequence
        self.history=[]

    def on_epoch_end(self, epoch, logs=None):
        pred_y = self.model((self.validation_sequence.users,self.validation_sequence.deep_features))
        y=self.validation_sequence.ratings
        mse=np.mean(pow(pred_y-y,2))
        self.history.append(mse)
        print(f" adversarial val loss: {mse}")