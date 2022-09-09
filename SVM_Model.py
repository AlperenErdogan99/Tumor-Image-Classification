#%%
# ROC'lar eklenecek 
# veri setinin kendin eklediğin tarafını belirt sadece. Kodda bişe yapmana gerek yok 

#%%

import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

import tensorflow as tf 
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, average_precision_score
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import pandas as pd 
from sklearn.multiclass import OneVsRestClassifier

DATASET_PATH = "SVM_Dataset_Feature_Hazir.json" #veri seti dosyası


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    #convert lists into numpy arrays
    inputs = np.array(data["Features"], dtype= object)
    targets = np.array(data["labels"], dtype= object)
    
    return inputs, targets


if __name__ == "__main__":
        
    #load data
    inputs, targets = load_data(DATASET_PATH)
    lst = ["glioma", "meningioma","pituitary"]
 
    # converting list to array
    targets_names = np.array(lst)
    
    #split the data into train set and test set 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split( inputs,
    targets,
    test_size = 0.2)
    
    #label typedef 
    targets_train=targets_train.astype('int')
    targets_test = targets_test.astype('int')
    
    #%% Create linear SVM model 
    
    
   
    clf = svm.SVC(kernel='linear') # Linear Kernel
    y_score = clf.fit(inputs_train, targets_train)
    
    y_pred = clf.predict(inputs_test)
    #print("No kFold, Linear SVM Accuracy:",metrics.accuracy_score(targets_test, y_pred))
    scores = cross_val_score(clf, inputs_test, targets_test, cv=5)
    #print("Kfold k=5 , Linear SVM Accuracy: %0.2f " % (scores.mean()))
   
  
    
    #%% Linear SVM Confusion Matrix
    titles_options = [("Linear SVM Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, inputs_test, targets_test,
                                 display_labels=targets_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    plt.show()
    #%% Linear SVM ROC curve 
    # Compute ROC curve and ROC area for each class
   

    
    #%% Linear SVM Classification Report 
    visualizer = ClassificationReport(clf, classes=targets_names, support=None)

    visualizer.fit(inputs_train, targets_train)        # Fit the visualizer and the model
    visualizer.score(inputs_test, targets_test)        # Evaluate the model on the test data
    visualizer.show()  
    target_names = ['glioma', 'meningioma','pituitary']
    print(classification_report(targets_test, y_pred, target_names=target_names))
    print("Without Kfold, Linear SVM Accuracy: ", metrics.accuracy_score(targets_test, y_pred))
    print("Kfold k=5 , Linear SVM Accuracy: %0.2f " % (scores.mean()))
    print("------------------------------------------------")
    print("------------------------------------------------")
    print(" ")
    #%% Linear SVM ROC 
                                     
   
    #%% Create Polynomial SVM model 
   
    svclassifier = SVC(kernel='poly', degree=10)
    svclassifier.fit(inputs_train, targets_train)
    y1_pred = svclassifier.predict(inputs_test)
    #print("Polynomial SVM Accuracy:",metrics.accuracy_score(targets_test, y1_pred))
    scores1 = cross_val_score(svclassifier, inputs_train, targets_train, cv=5)
    #print("Kfold k=5 , Polynomial SVM Accuracy: %0.2f " % (scores1.mean()))
    #%% Polynomail SVM Confusion Matrix
    titles_options = [("Polynomial SVM Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(svclassifier, inputs_test, targets_test,
                                 display_labels=targets_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    plt.show()
   #%% Polynomial SVM Classification Report 
    visualizer = ClassificationReport(svclassifier, classes=targets_names, support=None)

    visualizer.fit(inputs_train, targets_train)        # Fit the visualizer and the model
    visualizer.score(inputs_test, targets_test)        # Evaluate the model on the test data
    visualizer.show()  

    target_names = ['glioma', 'meningioma','pituitary']
    print(classification_report(targets_test, y1_pred, target_names=target_names))
    print("Without Kfold, Polynomial SVM Accuracy:",metrics.accuracy_score(targets_test, y1_pred))
    print("Kfold k=5 , Polynomial SVM Accuracy: %0.2f " % (scores1.mean()))
    print("------------------------------------------------")
    print("------------------------------------------------")
    #%% Polynomial SVM ROC
    
    
    
    