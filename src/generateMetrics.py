from sklearn.metrics import (classification_report, f1_score , confusion_matrix,  roc_auc_score, accuracy_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from datetime import datetime

targets = ['N', 'S','V','F','Q']

labels=[0,1,2,3,4]


#This function is needed when using softmax loss function (i.e. multiclass)
def classPrediction(arr):

    class_predictions = []
    for x in arr:
        class_predictions.append(np.argmax(x))

    return class_predictions

def summaryReport(y_actual, y_pred, stage):

    y_class = classPrediction(y_pred)
    print(stage + " stage")
    
    print(classification_report(y_actual, y_class, target_names=targets))

#Print metrics for model performance
def print_report(y_actual, y_pred, modelName, stage):

    y_class = classPrediction(y_pred)
    print('Metrics for model: ' + modelName)
    print(stage + " stage")
    
    
   # auc = roc_auc_score(y_actual, y_pred)

    accuracy = accuracy_score(y_actual, y_class)

    recall = recall_score(y_actual, y_class, average=None, labels=labels )
    f1 = f1_score(y_actual, (y_class), average=None, labels=labels )
    precision = precision_score(y_actual, y_class, average=None, labels=labels )
    
   # specificity = recall_score(y_actual, y_class, pos_label=0)
    
  #  print('AUC:', auc)
    print('accuracy:' ,accuracy)
    print('recall:' , recall)
   # print('specificity:' , specificity)
    print('precision:' , precision)
    print('F1:' , f1)
    print('')

  #  return  auc, accuracy, recall, specificity,  precision ,f1


#Plot and save confusion matrix for model predictions
def confusionMatrix(y_actual, y_pred, modelName, stage):

    plt.figure('conf_matrix')
    y_class = classPrediction(y_pred)
    sb.heatmap(confusion_matrix(y_actual, y_class, normalize='true'),
    annot = True, xticklabels = targets, yticklabels = targets, cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(stage + ' : ' + modelName)

    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    plt.savefig('./output/confusion_matrix' + stage + '_' + modelName + '_' + date + '.jpg')
    plt.show()

    