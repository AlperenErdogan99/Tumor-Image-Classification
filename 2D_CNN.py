#%%
# ROC eklenmelidir 
# kfold eklenmelidir 
# Yukarıdakiler yapıldıktan sonra 100 epochla eğitim yap 

#%%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt 
from glob import glob
import cv2 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

train_path = "tumor/Training/"
test_path = "tumor/Test/"
validation_path = "tumor/Validation"


img = cv2.imread(train_path + "glioma/209.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
shapeOfImg = x.shape #img'ların shape öğrendik
print(x.shape)#


className = glob(train_path + '/*')
numberOfClass = len(className) #class sayımız
print("NumberOfCLass",numberOfClass)

#%% CNN model 
#modelimi oluşturuyorum
model = Sequential()

# 32 adet kernel
# 3x3 boyutunda kernel
model.add(Conv2D(32,(3,3),input_shape = shapeOfImg))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(numberOfClass))
model.add(Activation("softmax"))

#modelin optimizasyon değerlerini giriyorum
model.compile(loss = "categorical_crossentropy",
              optimizer="rmsprop",
              metrics = ["accuracy"])
#her iterasyonda kaç img eğitlcek veriyoruz
batch_size = 32 

#%% Data Generation - Train - Test

train_datagen = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3
                   )
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size = shapeOfImg[:2],
                                                    batch_size=batch_size,
                                                    color_mode = "grayscale",
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size = shapeOfImg[:2],
                                                    batch_size=batch_size,
                                                    color_mode = "grayscale",
                                                    class_mode="categorical")

validation_generator =  validation_datagen.flow_from_directory(test_path,
                                                    target_size = shapeOfImg[:2],
                                                    batch_size=batch_size,
                                                    color_mode = "grayscale",
                                                    class_mode="categorical")


hist = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = 1600// batch_size, #1600 çoğaltılmış olan datanın temsili sayısı
    epochs=50, ##100kez eğitilcek 
    validation_data = test_generator,
    validation_steps = 800//batch_size)


#%% model Save
model.save("BrainTumor_CNN_Model.h5")

#%% model evaluation
#modelin sonuçlarını inceliyoruz

print(hist.history.keys())
plt.figure()
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"],label="Train accuracy")
plt.plot(hist.history["val_accuracy"],label="Test accuracy")
plt.legend()
plt.show()

Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
a = confusion_matrix(test_generator.classes, y_pred)
print(confusion_matrix(test_generator.classes, y_pred))
#%% 

target_names = ['glioma', 'meningioma','pituitary']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

