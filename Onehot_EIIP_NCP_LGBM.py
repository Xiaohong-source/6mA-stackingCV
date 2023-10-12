from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm

from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb
import numpy as np
from data_processing import Onehot_EIIP, Onehot_NCP, Onehot_EIIP_NCP
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

model = 'Onehot_EIIP_NCP_LGBM'

def sn_sp_acc_mcc(true_label, predict_label, pos_label=1):
    import math
    pos_num = np.sum(true_label == pos_label)
    print('pos_num=', pos_num)
    neg_num = true_label.shape[0] - pos_num
    print('neg_num=', neg_num)
    tp = np.sum((true_label == pos_label) & (predict_label == pos_label))
    print('tp=', tp)
    tn = np.sum(true_label == predict_label) - tp
    print('tn=', tn)
    sn = tp / pos_num
    sp = tn / neg_num
    acc = (tp + tn) / (pos_num + neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=', fn)
    print('fp=', fp)

    tp = np.array(tp, dtype=np.float64)
    tn = np.array(tn, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    fn = np.array(fn, dtype=np.float64)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)))
    return sn, sp, acc, mcc



Rosaceae_train,Rosaceae_train_y = Onehot_EIIP_NCP('Datasets/Training_Dataset/positive_training_dataset_for_Rosaceae.txt','Datasets/Training_Dataset/negative_training_dataset_for_Rosaceae.txt')
Rosaceae_test,Rosaceae_test_y = Onehot_EIIP_NCP('Datasets/Test_Dataset/positive_test_dataset_for_Rosaceae.txt','Datasets/Test_Dataset/negative_test_dataset_for_Rosaceae.txt')
Arabidopsis_test,Arabidopsis_test_y = Onehot_EIIP_NCP('Datasets/Test_Dataset/positive_test_dataset_for_Arabidopsis_Thaliana.txt','Datasets/Test_Dataset/negative_test_dataset_for_Arabidopsis_Thaliana.txt')
Rice_test,Rice_test_y = Onehot_EIIP_NCP('Datasets/Test_Dataset/positive_test_dataset_for_Rice.txt','Datasets/Test_Dataset/negative_test_dataset_for_Rice.txt')


model1 = XGBClassifier()
model2 = GradientBoostingClassifier()
model3 = lgb.LGBMClassifier()

model4 = svm.SVC(probability=True)

stack = StackingCVClassifier(
    classifiers=[model1,model2,model3], meta_classifier=model3, random_state=10,use_probas=True,cv=5,n_jobs=-1)

print('training ' +  model +'....')
stack.fit(Rosaceae_train,Rosaceae_train_y)
print('Training complete!')
print('Predict Rosaceae species')
stack_pred = stack.predict(Rosaceae_test)
res = stack_pred
pred = res
f = pred>0.5
pred[f]=1
pred[pred<0.6]=0
sn_sp_acc_mcc1 = sn_sp_acc_mcc(Rosaceae_test_y,pred,pos_label=1)
print('SN_SP_ACC_MCC=',sn_sp_acc_mcc1)

FPR,TPR,threshold=roc_curve(Rosaceae_test_y,stack.predict_proba(Rosaceae_test)[:,1] ,pos_label=1)

AUC=auc(FPR,TPR)
print('AUC=',AUC)
print('Predict Rosaceae species task accomplished')

print('Predict Arabidopsis species')
stack_pred = stack.predict(Arabidopsis_test)
res = stack_pred
pred = res
f = pred>0.5
pred[f]=1
pred[pred<0.6]=0
sn_sp_acc_mcc1 = sn_sp_acc_mcc(Arabidopsis_test_y,pred,pos_label=1)
print('SN_SP_ACC_MCC=',sn_sp_acc_mcc1)

FPR,TPR,threshold=roc_curve(Arabidopsis_test_y,stack.predict_proba(Arabidopsis_test)[:,1] ,pos_label=1)

AUC=auc(FPR,TPR)
print('AUC=',AUC)
print('Predict Arabidopsis species task accomplished')

print('Predict Rice species')
stack_pred = stack.predict(Rice_test)
res = stack_pred
pred = res
f = pred>0.5
pred[f]=1
pred[pred<0.6]=0
sn_sp_acc_mcc1 = sn_sp_acc_mcc(Rice_test_y,pred,pos_label=1)
print('SN_SP_ACC_MCC=',sn_sp_acc_mcc1)

FPR,TPR,threshold=roc_curve(Rice_test_y,stack.predict_proba(Rice_test)[:,1] ,pos_label=1)

AUC=auc(FPR,TPR)
print('AUC=',AUC)
print('Predict Rice species task accomplished')