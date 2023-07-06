from Recommender import get_test_generator, get_model
from Utils import recommender_weights_path, cl_recommender_test_features_path, cl_recommender_worst_features_path, \
    recommender_weights_with_augmentation_path, Test_set
import tensorflow as tf
import numpy as np
from recommender_fast_gradient_method import recommender_fast_gradient_method

from recommender_projected_gradient_descent import recommender_projected_gradient_descent

EPS = 0.2
NUM_IMAGES=900
model_type=1
def recomender_generate(augmentation = False,test_set=Test_set.BALANCED,fgm=False,pgd=True,eps=EPS,eps_iter=0.01,n_iter=40):

    test_generator=get_test_generator(test_set)
    num_users=len(test_generator.training_users) + 1

    model= get_model(num_users,model_type)
    if augmentation:
        model.load_weights(recommender_weights_with_augmentation_path[model_type])
    else:
        model.load_weights(recommender_weights_path[model_type])


    test_mse_fgsm = tf.metrics.MeanSquaredError()
    test_loss=tf.losses.MeanSquaredError()


    # ADVERSARIAL ATTACK (FAST GRADIENT METHOD)
    if(fgm):
        x_fgm = recommender_fast_gradient_method(model,test_generator.users,test_generator.deep_features, eps, np.inf,loss_fn=test_loss,y=np.array([1]*len(test_generator.indexes)),targeted=True)
        if test_set==Test_set.WORST :
            np.save(cl_recommender_worst_features_path,x_fgm.numpy())
        else:
            np.save(cl_recommender_test_features_path[model_type], x_fgm.numpy())

    #ADVERSARIAL ATTACK (PROGECTED GRADIENT DESCENT)
    if(pgd):
        x_pgd = recommender_projected_gradient_descent(model_fn=model,users=test_generator.users,
                                                   deep_features=test_generator.deep_features, eps=eps,eps_iter=eps_iter, nb_iter=n_iter,norm=np.inf,
                                                   loss_fn=test_loss,y=np.array([1]*len(test_generator.indexes)),targeted=True)
        if test_set==Test_set.WORST :
            np.save(cl_recommender_worst_features_path+str(eps_iter),x_pgd.numpy())
        else:
            np.save(cl_recommender_test_features_path[model_type]+str(eps_iter), x_pgd.numpy())

def recomender_adversarial_test(augmentation = False,test_set=Test_set.FULL,eps_iter=0.005):
    test_generator = get_test_generator(t_set=test_set)
    num_users = len(test_generator.training_users) + 1
    model = get_model(num_users, model_type)
    if augmentation:
        model.load_weights(recommender_weights_with_augmentation_path[model_type])
    else:
        model.load_weights(recommender_weights_path[model_type])
    if test_set==Test_set.WORST:
        x_pgd=np.load(cl_recommender_worst_features_path+str(eps_iter)+".npy")
    else:
        x_pgd = np.load(cl_recommender_test_features_path[model_type]+str(eps_iter)+".npy")
    test_mse_pgd = tf.metrics.MeanSquaredError()
    x_mean = np.mean(test_generator.ratings)
    y_pred = model.predict(test_generator)
    y_pred_mean=np.mean(y_pred)
    ##TEST
    y_pred_pgd = model([test_generator.users, x_pgd])
    test_mse_pgd(np.array([1] * len(test_generator.indexes)), y_pred_pgd)
    mean_pred_pgd = np.mean(y_pred_pgd)
    fooled = 0
    unfooled = 0
    bad = 0
    improved = 0
    worsened = 0
    for i in range(NUM_IMAGES):
        if y_pred[i] < 0.5:
            bad = bad + 1
            if y_pred_pgd[i] >= 0.5:
                fooled = fooled + 1
        if y_pred[i] >= 0.5 and y_pred_pgd[i] < 0.5:
            unfooled = unfooled + 1
        if y_pred[i] < y_pred_pgd[i]:
            improved = improved + 1
        elif y_pred[i] > y_pred_pgd[i]:
            worsened = worsened + 1
    print(f"ratings mean:{x_mean} , predictions mean: {y_pred_mean}, adversarial predictions mean: {mean_pred_pgd}, diff: {mean_pred_pgd-y_pred_mean}")
    print(f"{bad}bad recomendations, {fooled} fooled, {unfooled} unfooled {improved} improved, {worsened} worsened")
    ## \TEST