import os

from Cleverhans_densenet import densenet_generate, densenet_adversarial_test
from Cleverhans_recommender import recomender_generate, recomender_adversarial_test
from Recommender import recommender_data_preparation, recommender_train, recommender_test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from AnommalyDetector import anomaly_detector_train, anomaly_detector_test
from Utils import cl_densenet_test_features_path, cl_recommender_test_features_path, Test_set, \
    cl_densenet_worst_features_path, original_database_path
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.FATAL)
tf.autograph.set_verbosity(0)


############################# DATA PREPARATION ###################################################

recommender_data_preparation(original_database_path,generate_deep_features=True,download_images=True,scale_images=True)
'''
#Genera los dfs (data/dataframes/test_images_df,train_images_df,worst_images_df)
#Descarga las imagenes (images_shared)
#Genera los deep features(data/features/clean_deep_features,test_deep_features,worst_deep_features)
'''
#recommender_data_preparation(original_database_path,generate_deep_features=True,download_images=False,scale_images=False)

########################### RECOMMENDER TRAINING ###################################################
'''
Entrena los modelos de recomendador +
Guarda los pesos (data/weights/recommender_weights[1-3] , weights/recommender_with_augmentation_weights[1-3])
Guarda las gráficas de entrenamiento (figures/recommender_augmentation_[False / True]_model_[1-3])
'''


recommender_train(training_epochs=40, lr=0.00001,execute_augmentation=False, model_type=1)
recommender_train(training_epochs=40,lr=0.00001,execute_augmentation=False,model_type=2)
recommender_train(training_epochs=40,lr=0.00001,execute_augmentation=False,model_type=3)

recommender_train(training_epochs=40, lr=0.00001, execute_augmentation=True, model_type=1)
recommender_train(training_epochs=40,lr=0.00001,execute_augmentation=True,model_type=2)
recommender_train(training_epochs=40,lr=0.00001,execute_augmentation=True,model_type=3)

########################## RECOMMENDER TEST ############################################################
'''
Prueba los modelos  del recomendador

'''
recommender_test(augmentation=True,test_set=Test_set.BALANCED,model_type=1)
recommender_test(augmentation=True,test_set=Test_set.BALANCED,model_type=2)
recommender_test(augmentation=True,test_set=Test_set.BALANCED,model_type=3)



'''
###########################################ANOMMALY DETECTOR TRAINING ###########################################
Entrena los modelos de detector de anomalias
Guarda los pesos en data/weights/annomnaly_detector_weights[1-3]
Guarda las gráficas de entrenamiento en figures/annomaly_detector_train
'''

anomaly_detector_train(cl_recommender_test_features_path[1],epochs=500,lr=0.0001,model_n=1)
anomaly_detector_train(cl_recommender_test_features_path[1],epochs=500,lr=0.0001,model_n=2)
anomaly_detector_train(cl_recommender_test_features_path[1],epochs=500,lr=0.0001,model_n=3)


############################################ GENERATING ADVERSARIOUS SAMPLES #######################################
for eps in (0.005,0.007,0.009):
    print(f"eps: {eps}")
    recomender_generate(augmentation=True,test_set=Test_set.FULL,fgm=False,pgd=True,eps=3.0,eps_iter=eps,n_iter=40)
    recomender_adversarial_test(augmentation=True,test_set=Test_set.FULL,eps_iter=eps)

'''
Genera ejemplos de features adversarias del conjunto de test (o del conjunto worst de las reviews negativas)
con los metodos fgm o pgd (realmente solo uso pgd) sobre la red recommender
guarda las features en features\cl_recommender_test_features_model[1-3].npy
'''

#DENSENET GENERATE WORST
print("densenet worst")
for eps in (0.03,0.05,0.1):
    print(f"eps iter: {eps}")
    densenet_generate(augmentation=True,worst=True,fgm=False,pgd=True,eps=30.0,eps_iter=eps,n_iter=30 ,num_images=256) #EL QUE MANDA ES EPS ITER
    densenet_adversarial_test(augmentation=True,worst=True,num_images=256,eps_iter=eps)

#DENSENET GENERATE TEST
print("densenet test")
for eps in (0.03,0.05,0.1):
    print(f"eps iter: {eps}")
    densenet_generate(augmentation=True,worst=False,fgm=False,pgd=True,eps=30.0,eps_iter=eps,n_iter=30 ,num_images=256) #EL QUE MANDA ES EPS ITER
    densenet_adversarial_test(augmentation=True,worst=False,num_images=256,eps_iter=eps)




'''
Genera ejemplos de imágenes adversarias del conjunto de test (o del conjunto worst de las reviews negativas)
con los metodos fgm o pgd (realmente solo uso pgd) sobre la red densenet seguida de la red recommender
Y extrae las features de las imágenes
Guarda las imágenes adversarias en images/cleverhans_densenet_test_images_model[1-3]
y los features en features\cl_densenet_[test/worst]_features_model[1-3].npy
'''
######################## TEST ANNOMALY DETECTOT RECOMMENDER ###############################
print("recommender test")
for eps in (0.005,0.007,0.009):
    print(f"eps = {eps}")
    for mdl in (1, 2, 3):
        anomaly_detector_test(cl_recommender_test_features_path[1], model_n=mdl, worst=False,clipplot=1,eps_iter=eps)

############################################ TEST ANOMALY DETECTOR DENSENET#####################################################
print("densenet test")
for eps in (0.03,0.05,0.1):
    for mdl in (1,2,3):
        anomaly_detector_test(cl_densenet_test_features_path[1],model_n=mdl,worst=False,eps_iter=eps,clipplot=1)


###############WORST FEATURES CON EPS=0.03,0.05,0.1
print("densenet worst")
for eps in (0.03,0.05,0.1):
    for mdl in (1,2,3):
        anomaly_detector_test(cl_densenet_worst_features_path ,model_n=mdl,worst=True,eps_iter=eps)
