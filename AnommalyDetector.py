import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Models import AnomalyDetector, AnomalyDetector2, AnomalyDetector3
from Utils import deep_features_path,  test_features_path, worst_features_path, annomaly_detector_weights_path, SecondValidationSet
from clasiffier import anommaly_detector_roc

seed=1000

models=[None,
    AnomalyDetector,
    AnomalyDetector2,
    AnomalyDetector3
]

def anomaly_detector_train(adversarial_features_path,epochs=500,lr=0.0001,model_n=1,show_plot=False):
    with open(deep_features_path, 'rb') as f:
        deep_features = np.load(f)
    with open(adversarial_features_path, 'rb') as f:
        adversarial_deep_features = np.load(f)
    train_data, clean_test_data = train_test_split(deep_features, test_size=0.2, random_state=seed)
    adversarial_test_data=adversarial_deep_features

    ########################################## TRAIN ######################################

    autoencoder = models[model_n]()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    second_validation_set=SecondValidationSet(adversarial_test_data)
    history = autoencoder.fit(train_data, train_data,
              epochs=epochs,
              batch_size=64,
              validation_data=(clean_test_data,clean_test_data),
              callbacks=[second_validation_set],
              shuffle=True)

    autoencoder.save_weights(annomaly_detector_weights_path[model_n])

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Clean validation Loss")
    plt.plot(second_validation_set.history, label="Adversarial validation Loss")
    plt.axis([-2,epochs,0.1,0.7,])
    plt.legend()
    cl_method_name=adversarial_features_path.split("\\")[-1].split(".")[0]
    plt.savefig(f"figures\\anomaly_detector\\train\\{cl_method_name}_model_{model_n}.png")
    if show_plot:
        plt.show()
    plt.close()

############################################ TEST #############################################
def anomaly_detector_test(adversarial_features_path,model_n=1,worst=False,showplots=False,clipplot=2,nbins=30,eps_iter=0):
    if(eps_iter!=0):
        adversarial_features_path=adversarial_features_path+str(eps_iter)+".npy"
    with open(deep_features_path, 'rb') as f:
        deep_features = np.load(f)
    with open(adversarial_features_path, 'rb') as f:
        adversarial_deep_features = np.load(f)
    num_images = len(adversarial_deep_features)
    if worst:
        clean_features_path = worst_features_path
    else:
        clean_features_path = test_features_path
    with open(clean_features_path, 'rb') as f:
        clean_deep_features = np.load(f)

    clean_deep_features=clean_deep_features[0:num_images]

    train_data, clean_test_data = train_test_split(deep_features, test_size=0.2, random_state=seed)
    adversarial_test_data=adversarial_deep_features
    autoencoder = models[model_n]()
    autoencoder.load_weights(annomaly_detector_weights_path[model_n])#DISTINTOS WEIGHTS,MODELS;ETC?????

    #Test clean set same as adversarial images
    test_clean_predictions = autoencoder.predict(clean_deep_features)
    clean_test_mse_array = np.mean(np.power(clean_deep_features - test_clean_predictions, 2), axis=1)
    clean_test_mse =np.mean(clean_test_mse_array)
    clean_test_var=np.var(clean_test_mse_array)
    print(f"Mean Error in clean data: {clean_test_mse}")
    print(f"Error variance in clean data: {clean_test_var}")

    #Adversarial images
    test_adversarial_predictions = autoencoder.predict(adversarial_test_data)
    adversarial_mse_array = np.mean(np.power(adversarial_test_data - test_adversarial_predictions, 2), axis=1)
    adversarial_mse =np.mean(adversarial_mse_array)
    adversarial_var=np.var(adversarial_mse_array)
    #clipped_adversarial_mse_array=adversarial_mse_array[(adversarial_mse_array<=clipplot)]
    print(f"Mean error in adversarial data: {adversarial_mse}")
    print(f"Error variance in adversarial data: {adversarial_var}")

    thresholds=[-1.0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]
    tprl,fprl = anommaly_detector_roc(clean_test_mse_array, adversarial_mse_array, thresholds, clean_test_mse, adversarial_mse)
    auc=0
    for i in range(1,len(tprl)):
        aut = (fprl[i]-fprl[i-1])*(tprl[i]-tprl[i-1])/2
        aus=(fprl[i]-fprl[i-1])*tprl[i-1]
        auc=auc + aut + aus
    print(f"AUC = {auc} ")

    plt.figure(1)
    plt.plot(fprl,tprl)
    plt.title("ROC CURVE")
    plt.ylabel("TPR")
    plt.xlabel("FPR")

    plt.axis([-0.1, 1.1, -0.1, 1.1])
    if(model_n==3):
        plt.legend(["model 1", "model 2", "model 3"], loc="lower right")
        cl_method_name = adversarial_features_path.split("\\")[-1].split(".")[0]
        plt.savefig(f"figures\\anomaly_detector\\test\\{cl_method_name}_{eps_iter}_plot.png")
        if showplots:
            plt.show()
        plt.close()

    plt.figure(2)
    plt.hist(clean_test_mse_array,bins=nbins,alpha = 0.8, range=(0,clipplot), label="Clean test set Loss (same images as adversarial set)")
    plt.hist(adversarial_mse_array,bins=nbins,alpha = 0.8,range=(0,clipplot), label="Adversarial test set Loss")
    plt.legend()
    cl_method_name = adversarial_features_path.split("\\")[-1].split(".")[0]
    plt.savefig(f"figures\\anomaly_detector\\test\\{cl_method_name}_model_{model_n}_eps{eps_iter}bar.png")
    if showplots:
        plt.show()
    plt.close()


