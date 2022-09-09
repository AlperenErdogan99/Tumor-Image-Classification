import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Conv1D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt 
from glob import glob
import tensorflow as tf 

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
    #split the data into train set and test set 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split( inputs,
    targets,
    test_size = 0.3)
    
    inputs_train = np.expand_dims(inputs_train, axis=2) # reshape (212, 441) to (212, 441, 1) 
    inputs_test = np.expand_dims(inputs_test, axis=2) #reshape (92, 441) to (92,441,1) 
    
    
    #to categorical for labels
    targets_train = tf.keras.utils.to_categorical(targets_train, num_classes=3)
    targets_test = tf.keras.utils.to_categorical(targets_test, num_classes=3)
    
    #build 1D CNN architecture
    model = Sequential()
    
    
    model.add(Conv1D(filters = 64, kernel_size = 5, activation = "relu", input_shape=(inputs_train.shape[1],inputs_train.shape[2]))) 
    #model.add(Conv1D(filters = 512, kernel_size = 3, activation = "relu", input_shape=(441,1))) üstteki satır yerine bu satırıda kullanabilrsin
    model.add(Dropout(0.4))
    #model.add(MaxPooling1D(pool_size = 2 ))
    
    model.add(Conv1D(filters = 64, kernel_size = 5, activation = "relu"))
    model.add(Dropout(0.4))
    #model.add(MaxPooling1D(pool_size = 2 ))
    
    model.add(Conv1D(filters = 32, kernel_size = 5, activation = "relu"))
    model.add(Dropout(0.4))
    
    model.add(Conv1D(filters = 128, kernel_size = 3, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size = 2 ))
    
    model.add(Conv1D(filters = 128, kernel_size = 3, activation = "relu"))
    model.add(Dropout(0.4))
    
       
    model.add(Flatten())
    model.add(Dense(1024,activation = "relu"))
    model.add(Dense(256, activation = "relu"))
    
    model.add(Dense(3, activation = "softmax"))
    
    # compile model
    model.compile(loss= "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])
    
    # train model
    hist = model.fit(inputs_train, targets_train, epochs = 30, batch_size = 32) 
    _, accuracy = model.evaluate(inputs_test, targets_test, batch_size = 32)
   
    
    model.save("deneme1.h5")