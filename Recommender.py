import os
import shutil
from enum import Enum

import keras.metrics
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from keras.utils import save_img
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from Utils import get_pkl_data, pkl_save, logger, recommender_weights_path, recommender_weights_with_augmentation_path, \
    test_images_df_path, train_images_df_path, worst_images_df_path, \
    SecondValidationSequence, test_features_path, worst_features_path, balanced_test_features_path, \
    balanced_images_df_path, Test_set, SecondValidationSequenceManual, original_images_path, scaled_images_path
# Paths
from Utils import deep_features_path, original_database_path  # Esto hay que ver donde va
from Models import get_deep_features, RecommenderNet, RecommenderNet2, RecommenderNet3
from Generators import MyTestingGenerator, MyTrainingGenerator
from Utils import download_all_images

BATCH_SIZE = 64
EMBEDDING_SIZE = 64
TARGET_SIZE = (224, 224)
LR = 0.00001
FINE_LR = 0.000001
FINE_LR_EPOCHS = 70


def scheduler(epoch, lr):
    if epoch < FINE_LR_EPOCHS:
        return lr
    else:
        return FINE_LR


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


def get_model(num_users, model_type=1, lr=0.00001):
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    auc = keras.metrics.AUC(curve='PR')

    if model_type == 2:
        model = RecommenderNet2(num_users, EMBEDDING_SIZE)
    elif model_type == 3:
        model = RecommenderNet3(num_users, EMBEDDING_SIZE)
    else:
        model = RecommenderNet(num_users, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['mse', 'mae', precision, recall, auc]
    )
    return model


def get_train_generator(augmentation=False):
    train_data = get_pkl_data(train_images_df_path)
    with open(deep_features_path, 'rb') as f:
        deep_features = np.load(f)
    deep_features_train = deep_features[train_data.index]

    return MyTrainingGenerator(
        train_data[['user']].values,
        deep_features_train,
        train_data[['rating']].values,
        batch_size=BATCH_SIZE,
        input_size=TARGET_SIZE,
        with_data_augmentation=augmentation)



def get_test_generator(t_set):
    with open(deep_features_path, 'rb') as f:
        deep_features = np.load(f)
    if t_set==Test_set.FULL:
        test_data = get_pkl_data(test_images_df_path)
    elif t_set == Test_set.WORST:
        test_data = get_pkl_data(worst_images_df_path)
    else:  #if t_set==Test_set.BALANCED:
        test_data = get_pkl_data(balanced_images_df_path)
    deep_features_test = deep_features[test_data.index]
    train_data = get_pkl_data(train_images_df_path)

    return MyTestingGenerator(
        test_data[['user']].values,
        deep_features_test,
        test_data[['rating']].values,
        np.unique(train_data[['user']].values),
        batch_size=BATCH_SIZE,
        input_size=TARGET_SIZE)


###############################################################################################################
########### DATA PREPARATION ##################################################################################
###############################################################################################################
def recommender_data_preparation(db_path,generate_deep_features=True,download_images=False,scale_images=True):
    # Load data
    data = get_pkl_data(db_path)
    if data.empty:
        logger.error('Error reading file')
        exit()
    data = data[['reviewId', 'restaurantId', 'userId', 'date', 'rating', 'images', 'url']]

    if download_images:
        if not os.path.exists(original_images_path):
            Path(original_images_path).mkdir(parents=True, exist_ok=True)
        download_all_images(original_database_path, original_images_path)

    restaurant_ids = data['restaurantId'].unique().tolist()
    restaurant2restaurant_encoded = {x: i for i, x in enumerate(restaurant_ids)}
    # restaurant_encoded2restaurant = {i: x for i, x in enumerate(restaurant_ids)}############
    data['restaurant'] = data['restaurantId'].map(restaurant2restaurant_encoded)
    data['user'] = data['userId']  # .map(user2user_encoded)

    # Change data type and get max and min to normalize
    # data['rating'] = data['rating'].values.astype(np.float32)
    min_rating = min(data['rating'])
    max_rating = max(data['rating'])
    data["rating"] = data["rating"].apply(
        lambda x: (x - min_rating) / (max_rating - min_rating)
    ).values

    # Prepare datasets
    images_df = pd.DataFrame()
    # Read the images in the folder
    images_df['image_name'] = os.listdir(original_images_path)
    images_df['review_id'] = images_df.apply(lambda row: int(row['image_name'].split('_')[0]), axis=1)

    # Join read images with the original dataset
    images_df = images_df.set_index('review_id').join(data.set_index('reviewId'))
    images_df = images_df[images_df['restaurantId'].notnull()]
    # Transform rates to float
    # images_df['rating'] = images_df['rating'].values.astype(np.float32)
    # images_df['rating'] = np.array(images_df['rating']).astype(np.float32)
    # Define datasets
    user_image_rate_df = images_df[['user', 'image_name', 'rating']]
    # image_restaurant_df = images_df[['image_name', 'restaurant']]


    #Recale de images and save them in scaled
    if(scale_images):
        print("Scaling images")
        for filename in os.listdir(original_images_path):
            f = os.path.join(original_images_path, filename)
            # checking if it is a file

            if os.path.isfile(f):
                im = Image.open(f)
                im= im.resize((224, 224))
                path = os.path.join(scaled_images_path, filename)
                save_img(path, im)
        print("Scaled images")
    # Create generator to extract deep features

    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    # Resize the images (640x480)
    img_width = 224
    img_height = 224
    x_col = 'image_name'
    y_col = 'rating'

    data_generator_df = image_data_generator.flow_from_dataframe(dataframe=user_image_rate_df, directory=original_images_path, #scaled
                                                                 x_col=x_col, y_col=y_col, class_mode="raw",
                                                                 target_size=(img_width, img_height),
                                                                 batch_size=BATCH_SIZE)

    # Calcular deep features
    print("Generating deep features")
    if (generate_deep_features):
        get_deep_features(data_generator_df, deep_features_path)  # Solo una vez

    user_image_rate_df = user_image_rate_df.reset_index(drop=True)

    # Prepare training and validation data

    seed = 42
    train_data, test_data = train_test_split(user_image_rate_df, test_size=0.2, random_state=seed)
    # Update indexes
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Preprocess the data
    user_ids = train_data['user'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    train_data['user'] = train_data['user'].map(user2user_encoded)
    user_image_rate_df['user'] = user_image_rate_df['user'].map(user2user_encoded)
    worst_data = (user_image_rate_df[user_image_rate_df['rating'] <= 0.25]).copy()

    rated_data=list()
    min_rate_data_count=99999
    for f in [0.0,0.25,0.5,0.75,1.0]:
        d=user_image_rate_df[user_image_rate_df['rating'] == f]
        min_rate_data_count=min(min_rate_data_count,d.shape[0])
        rated_data.append(d.copy())
    balanced_test_data=rated_data[0][0:min_rate_data_count].copy()
    for i in range(1,5):
        balanced_test_data=pd.concat([balanced_test_data,rated_data[i][0:min_rate_data_count]])

    if (generate_deep_features):
        test_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=test_data, directory=scaled_images_path,
                                                                          x_col=x_col, y_col=y_col, class_mode="raw",
                                                                          target_size=(img_width, img_height),
                                                                          batch_size=BATCH_SIZE,
                                                                          shuffle=False)
        get_deep_features(test_data_generator_df, test_features_path)
        worst_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=worst_data, directory=scaled_images_path,
                                                                      x_col=x_col, y_col=y_col, class_mode="raw",
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=BATCH_SIZE,
                                                                      shuffle=False)
        get_deep_features(worst_data_generator_df, worst_features_path)
        balanced_test_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=balanced_test_data, directory=scaled_images_path,
                                                                           x_col=x_col, y_col=y_col, class_mode="raw",
                                                                           target_size=(img_width, img_height),
                                                                           batch_size=BATCH_SIZE,
                                                                           shuffle=False)
        get_deep_features(balanced_test_data_generator_df, balanced_test_features_path)

    unknown_user = max(train_data['user']) + 1
    test_data['user'] = test_data['user'].map(user2user_encoded).fillna(unknown_user)
    test_data['user'] = test_data['user'].apply(lambda x: int(x))
    worst_data['user'] = worst_data['user'].fillna(unknown_user).apply(lambda x: int(x))
    balanced_test_data['user'] = balanced_test_data['user'].fillna(unknown_user).apply(lambda x: int(x))

    pkl_save(test_data, test_images_df_path)
    pkl_save(train_data, train_images_df_path)
    pkl_save(worst_data, worst_images_df_path)
    pkl_save(balanced_test_data, balanced_images_df_path)

##### TRAIN ##########################################################################################
def recommender_train(training_epochs=90, lr=0.00001, execute_augmentation=False, model_type=1, showplots=False):
    if not execute_augmentation:
        print("TRAIN WITHOUT AUGMENTATION")
    else:
        print("TRAIN WITH AUGMENTATION")
    train_generator = get_train_generator(execute_augmentation)
    test_generator = get_test_generator(t_set=Test_set.FULL)
    worst_generator = get_test_generator(t_set=Test_set.WORST)
    balanced_generator= get_test_generator(t_set=Test_set.BALANCED)
    second_validation_set = SecondValidationSequenceManual(worst_generator)
    third_validation_set = SecondValidationSequenceManual(balanced_generator)

    num_users = len(train_generator.unique_users) + 1
    model = get_model(num_users, model_type, lr)

    print("fitting model")
    history = model.fit(
        x=train_generator,
        batch_size=BATCH_SIZE,
        epochs=training_epochs,
        callbacks=[lr_callback, second_validation_set, third_validation_set],
        verbose=1,
        validation_data=test_generator,
    )
    # Loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.plot(second_validation_set.history, label="Worst validation Loss")
    plt.plot(third_validation_set.history, label="Balanced validation Loss")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test", "worst","balanced"], loc="upper left")
    plt.axis([0,40,0,0.18])
    if showplots:
        plt.show()
    plt.savefig(f"figures\\recommender\\train\\augmentation_{execute_augmentation}_model_{model_type}.png")
    plt.close()
    '''
    #PRECISSION RECALL AUC
    plt.plot(history.history["precision"])
    plt.plot(history.history["recall"])
    plt.plot(history.history["auc"])
    plt.title("precision recall AUC")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend(["precision", "recall", "AUC"], loc="lower right")
    plt.show()
    '''

    # Save the weights
    if execute_augmentation:
        model.save_weights(recommender_weights_with_augmentation_path[model_type])
    else:
        model.save_weights(recommender_weights_path[model_type])


################################################# TEST ##################################3
#######################################################################################################
def recommender_test(augmentation=False, test_set=Test_set.FULL, model_type=1):
    test_generator = get_test_generator(t_set=test_set)
    num_users = len(test_generator.training_users) + 1
    model = get_model(num_users, model_type)

    if not augmentation:
        print("NO AUGMENTATION")
        model.load_weights(recommender_weights_path[model_type])
    else:
        print("AUGMENTATION")
        model.load_weights(recommender_weights_with_augmentation_path[model_type])

    eval=model.evaluate(test_generator)
    print(eval)
    pred = model((test_generator.users,test_generator.deep_features)).numpy()
        #model.predict(test_generator)
    test_y = (test_generator.ratings * 4).astype(int)
    pred_y = (0.5 + pred * 4).astype(int)
    absolute_error = abs(test_generator.ratings - pred)
    squared_error = pow((test_generator.ratings - pred),2)
    accuracy = np.sum(test_y == pred_y) / len(test_generator.indexes)
    binary_test_y = (test_generator.ratings >= 0.5)
    binary_pred_y = ( model((test_generator.users,test_generator.deep_features)).numpy() >= 0.5)
    binary_accuracy = np.sum(binary_test_y == binary_pred_y) / len(test_generator.indexes)
    confusion = confusion_matrix(test_y, pred_y, normalize=None)
    confusion_plt = ConfusionMatrixDisplay(confusion)
    print(f"Real test mean: {np.mean(test_y)}")
    print(f"Prediction mean: {np.mean(pred_y)}")
    print(f"mae: {np.mean(absolute_error)}")
    print(f"mse: {np.mean(squared_error)}")
    print(f"accuracy: {accuracy}")
    print(f"binary accuracy: {binary_accuracy}")

    # print(confusion)

    #########Distribution histogram    

    plt.hist(pred_y, alpha=0.6)
    plt.hist(test_y, alpha=0.6)
    plt.title("Prediction distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Prediction")
    plt.legend(["Predicted", "Real"], loc="upper left")
    plt.show()

    confusion_plt.plot()
    plt.show()



