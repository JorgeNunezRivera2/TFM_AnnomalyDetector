import os

from keras_preprocessing.image import save_img

from Generators import UserImageRatingSequence
from Models import RecommenderFromImageNet, get_deep_features
from Utils import get_pkl_data, EMBEDDING_SIZE, test_images_df_path, \
    train_images_df_path, worst_images_df_path, cl_densenet_test_features_path, cl_densenet_test_images_path, \
    cl_densenet_worst_features_path, cl_densenet_worst_images_path, scaled_images_path, unpreprocess
import tensorflow as tf
import numpy as np
from PIL import Image

from recommender_fast_gradient_method import recommender_fast_gradient_method
from recommender_projected_gradient_descent import recommender_projected_gradient_descent

#CARGAR IMAGENES DE TEST
NUM_IMAGES = 64 #64
EPS = 0.2
model_type=1
def densenet_generate(augmentation=False, worst=False, fgm=False, pgd=True, eps=EPS, eps_iter=0.025, n_iter=60, num_images=NUM_IMAGES):
    if worst:
        test_data = get_pkl_data(worst_images_df_path)
    else:
        test_data = get_pkl_data(test_images_df_path)
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    # Resize the images (640x480)
    img_width = 224
    img_height = 224
    x_col = 'image_name'
    y_col = 'rating'
    batch_size = 64
    test_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=test_data, directory=scaled_images_path,
                                                                      x_col=x_col, y_col=y_col, class_mode="raw",
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size,
                                                                      shuffle=False,verbose=0)
    train_data=get_pkl_data(train_images_df_path)
    num_users=np.max(train_data.user)+2

    model = RecommenderFromImageNet(num_users=num_users, embedding_size=EMBEDDING_SIZE,augmentation=augmentation,model_n=model_type)
    test_generator = UserImageRatingSequence(test_data.user.iloc[0:num_images].to_numpy(), test_data_generator_df,test_data.rating.iloc[0:num_images].to_numpy(),batch_size)

    test_loss = tf.losses.MeanSquaredError()

    if fgm:
        x_adv=list()
        for i in range(num_images):
            nxt = test_generator.next()
            user=nxt[0]
            image=nxt[1]
            x_adv.append(recommender_fast_gradient_method(model, user, image, eps, np.inf,
                                                           loss_fn=test_loss, y=np.array([1] * num_images), targeted=True)[0])
            #if i%10==9:
            #    print(f"processed {i+1} images of {num_images}")

    if pgd:
        # GENERACION DE IMAGENES CON PROJECTED GRADIENT DESCENT
        x_adv=list()
        print(f"processing {num_images} images")
        for i in range(num_images):
            nxt = test_generator.next()
            user=nxt[0]
            image=nxt[1]
            x_adv.append(recommender_projected_gradient_descent(model_fn=model, users=user, deep_features=image,
                                                           eps=eps, eps_iter=eps_iter,
                                                           nb_iter=n_iter, norm=np.inf,
                                                           loss_fn=test_loss, y=np.array([1] * num_images), targeted=True)[0])
            if (i) % 10 == 9:
                print(f"processed {i+1} images of {num_images}")
    if worst:
        cl_images_path = cl_densenet_worst_images_path
    else:
        cl_images_path = cl_densenet_test_images_path[model_type]
    #Guardado de imagenes
    if(fgm or pgd):


        #GUARDADO IMÁGENES
        #print("Saving images")
        for i,im in enumerate(x_adv):
            #REESCALAR
            im=unpreprocess(im)
            im=np.clip(im,0,255)
            im= im.astype(np.uint8)
            #orig_im=Image.fromarray(im)
            #final_im=orig_im.resize((150,150))
            final_im=im
            #GUARDAR

            im_name = test_data['image_name'].iloc[i]
            path = os.path.join(cl_images_path,str(eps_iter),im_name)
            save_img(path,final_im)

    #print("Generating deep features")
    print(f"epsilon = {eps}")
    adversarial_test_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=test_data[0:num_images], directory=os.path.join(cl_images_path,str(eps_iter)),
                                                                      x_col=x_col, y_col=y_col, class_mode="raw",
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size)
    if worst:
        get_deep_features(adversarial_test_data_generator_df, cl_densenet_worst_features_path + str(eps_iter)+".npy")  # Solo una vez
    else:
        get_deep_features(adversarial_test_data_generator_df, cl_densenet_test_features_path[model_type]+str(eps_iter)+".npy") #Solo una vez

def densenet_adversarial_test(augmentation = False, worst=False, num_images=NUM_IMAGES, eps_iter=0.1):
    train_data = get_pkl_data(train_images_df_path)
    num_users = np.max(train_data.user) + 2
    if worst:
        test_data = get_pkl_data(worst_images_df_path)
        cl_images_path = cl_densenet_worst_images_path
    else:
        test_data = get_pkl_data(test_images_df_path)
        cl_images_path = cl_densenet_test_images_path[model_type]
    cl_images_path=os.path.join(cl_images_path,str(eps_iter))
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    model = RecommenderFromImageNet(num_users=num_users, embedding_size=EMBEDDING_SIZE, augmentation=augmentation,
                                    model_n=model_type)
    img_width = 224
    img_height = 224
    x_col = 'image_name'
    y_col = 'rating'
    batch_size = 64
    test_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=test_data, directory=scaled_images_path,
                                                                      x_col=x_col, y_col=y_col, class_mode="raw",
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size,
                                                                      shuffle=False)
    test_generator = UserImageRatingSequence(test_data.user.iloc[0:num_images].to_numpy(), test_data_generator_df,
                                             test_data.rating.iloc[0:num_images].to_numpy(), batch_size)
    adversarial_data_generator_df = image_data_generator.flow_from_dataframe(dataframe=test_data, directory=cl_images_path,
                                                                      x_col=x_col, y_col=y_col, class_mode="raw",
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size,
                                                                      shuffle=False)
    adversarial_generator = UserImageRatingSequence(test_data.user.iloc[0:num_images].to_numpy(), adversarial_data_generator_df,
                                             test_data.rating.iloc[0:num_images].to_numpy(), batch_size)
    x_adv = list()
    for i in range(num_images):
        nxt = adversarial_generator.next()[1][0]
        x_adv.append(nxt)
    mean_x = np.mean(test_generator.ratings)    
    y_pred = model.predict(test_generator)
    mean_pred = np.mean(y_pred)
    
    # TEST (CUANTOS ENGAÑOS)
    y_pred_adv = model([adversarial_generator.users[0:num_images], x_adv])
    mean_pred_adv = np.mean(y_pred_adv)
    fooled = 0
    unfooled = 0
    bad = 0
    improved = 0
    worsened = 0
    for i in range(num_images):
        if y_pred[i] <= 0.5:
            bad = bad + 1
            if y_pred_adv[i] >= 0.5:
                fooled = fooled + 1
        if y_pred[i] >= 0.5 and y_pred_adv[i] < 0.5:
            unfooled = unfooled + 1
        if y_pred[i] < y_pred_adv[i]:
            improved = improved + 1
        elif y_pred[i] > y_pred_adv[i]:
            worsened = worsened + 1
    print(
        f"Real mean: {mean_x}, prediction mean: {mean_pred}, adversarial prediction mean: {mean_pred_adv}, diff: {mean_pred_adv-mean_pred}"
    )
    print(f"{bad}bad recomendations, {fooled} fooled, {unfooled} unfooled {improved} improved, {worsened} worsened")
    # \TEST