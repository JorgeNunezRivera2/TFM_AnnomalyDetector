#DEVUELVE TRUE SI CREE QUE LA IMAGEN ES ORIGINAL
def linear_classifier (error,threshold,clean_mean,adv_mean):
    diff= adv_mean-clean_mean
    if(error>(clean_mean+diff*threshold)):
        return False
    else:
        return True

def classifier (error,threshold,clean_mean,clean_var,adv_mean,adv_var):
    diff= adv_mean-clean_mean
    if(error>(clean_mean+diff*threshold)):
        return False

def anommaly_detector_roc (clean_preds,adv_preds,thresholds,clean_mean,adv_mean):
    tprl=[]
    fprl=[]
    for t in thresholds:
        tpr = 0.0
        fpr = 0.0
        p=clean_preds.size
        n= adv_preds.size
        for e in clean_preds:
            if(linear_classifier(e,t,clean_mean,adv_mean)):
                tpr=tpr+1.0
        tprl.append(tpr/p)
        for e in adv_preds:
            if(linear_classifier(e,t,clean_mean,adv_mean)):
                fpr=fpr+1.0
        fprl.append(fpr/n)
    return((tprl,fprl))



