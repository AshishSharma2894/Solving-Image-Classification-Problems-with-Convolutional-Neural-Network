import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.applications.resnet import ResNet50
import keras.metrics

from sklearn.metrics import confusion_matrix
import seaborn as sns

Train_Path=r"C:\library\Data\train"
Test_Path=r"C:\library\Data\test"


Train_Covid_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/COVID19"
Train_Normal_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/NORMAL"
Train_Pne_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/PNEUMONIA"

Test_Covid_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/COVID19"
Test_Normal_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/NORMAL"
Test_Pne_Path="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/PNEUMONIA"

train_datagen= image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=image.ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(Train_Path,target_size=(224,224),batch_size=32,class_mode='categorical')

train_generator.class_indices

validation_generator=test_datagen.flow_from_directory(Test_Path,target_size=(224,224),batch_size=32,class_mode='categorical')
#print(train_generator.class_indices)
def ResNet():
  """
    Here we have created a model for training the ResNet model and measuring the accuracy of the model
  """
  epochs=50
  stepsperepoch=9
  validationsteps=1

  learningratesch=LearningRateScheduler(lambda x: 1e-3*0.95**x)

  earlystop=EarlyStopping(monitor='val_acc',mode='max',verbose=1,patience=100)
  modelcheckpoint=ModelCheckpoint("own.h5",monitor='val_loss',save_best_only=True,mode='min',verbose=1)


  ResNetModel=ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=None,pooling=None,classes=3,classifier_activation="softmax")

  ResNetModel.summary()

  ResNetModel.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=(['accuracy'],keras.metrics.Precision(),keras.metrics.Recall()))

  hist=ResNetModel.fit_generator(train_generator,epochs=epochs,callbacks=[learningratesch,modelcheckpoint,earlystop],steps_per_epoch=stepsperepoch,validation_data=validation_generator,validation_steps = validationsteps)


  preds = ResNetModel.evaluate(validation_generator)
  print ("Validation Loss = " + str(preds[0]))
  print ("Validation Accuracy = " + str(preds[1]))

  plt.plot(hist.history['accuracy'])
  plt.plot(hist.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(["Train_acc","Validation_acc"])
  plt.show()

  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(["Train_loss","Validation Loss"])
  plt.show()


  plt.rcParams.update({'font.size': 16})
  predictions = ResNetModel.predict(validation_generator)
  predictions = np.argmax(predictions,axis=1)
  predictions[:15]
  cm = confusion_matrix(validation_generator.classes, predictions)
  cm = pd.DataFrame(cm, index = ['0', '1', '2'], columns = ['0', '1', '2'])
  cm
  class_names = ['COVID','NORMAL','VIRAL PNEUMONIA']
  def plot_confusion_matrix (cm):
      plt.figure(figsize = (10,10))
      sns.heatmap(
          cm, 
          cmap = 'Blues', 
          linecolor = 'black', 
          linewidth = 1, 
          annot = True, 
          fmt = '', 
          xticklabels = class_names, 
          yticklabels = class_names)
      plt.show()
      
  plot_confusion_matrix(cm)

def Alexnet():
  """
    Here we have created a model for training the AlexNet model and measuring the accuracy of the model
  """
  DataModel = Sequential([
      Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
      BatchNormalization(),
      MaxPool2D(pool_size=(3,3), strides=(2,2)),
      Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPool2D(pool_size=(3,3), strides=(2,2)),
      Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPool2D(pool_size=(3,3), strides=(2,2)),
      Flatten(),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(3, activation='softmax')])

  DataModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=(['accuracy'],keras.metrics.Precision(),keras.metrics.Recall()))


  DataModel.summary()

  epochs = 15
  stepsperepoch=9
  validationsteps=1

  learningratesch = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

  earlystop = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
  modelcheckpoint = ModelCheckpoint("own1.h5", monitor='val_loss',save_best_only=True, mode='min',verbose=1)


  hist=DataModel.fit_generator(train_generator,epochs=epochs,callbacks=[learningratesch,modelcheckpoint,earlystop],steps_per_epoch=stepsperepoch,validation_data=validation_generator,validation_steps = validationsteps)

  preds = DataModel.evaluate(validation_generator)
  print ("Validation Loss = " + str(preds[0]))
  print ("Validation Accuracy = " + str(preds[1]))

  plt.plot(hist.history['accuracy'])
  plt.plot(hist.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(["Train_acc","Validation_acc"])
  plt.show()

  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(["Train_loss","Validation Loss"])
  plt.show()

  plt.rcParams.update({'font.size': 16})
  predictions = DataModel.predict(validation_generator)
  predictions = np.argmax(predictions,axis=1)
  predictions[:15]
  cm = confusion_matrix(validation_generator.classes, predictions)
  cm = pd.DataFrame(cm, index = ['0', '1', '2'], columns = ['0', '1', '2'])
  cm
  class_names = ['COVID','NORMAL','VIRAL PNEUMONIA']
  def plot_confusion_matrix (cm):
      plt.figure(figsize = (10,10))
      sns.heatmap(
          cm, 
          cmap = 'Blues', 
          linecolor = 'black', 
          linewidth = 1, 
          annot = True, 
          fmt = '', 
          xticklabels = class_names, 
          yticklabels = class_names)
      plt.show()
      
  plot_confusion_matrix(cm)


def FinalResNet():
  """
    Here we have created a model for training the ResNet-Tuned model and measuring the accuracy of the model
  """
  epochs=100
  stepsperepoch=9
  validationsteps=1

  learningratesch=LearningRateScheduler(lambda x: 1e-3*0.95**x)

  earlystop=EarlyStopping(monitor='val_acc',mode='max',verbose=1,patience=100)
  modelcheckpoint=ModelCheckpoint("own2.h5",monitor='val_loss',save_best_only=True,mode='min',verbose=1)


  ResNetFinalModel=ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=None,pooling=None,classes=3,classifier_activation="softmax")

  ResNetFinalModel.summary()

  ResNetFinalModel.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=(['accuracy'],keras.metrics.Precision(),keras.metrics.Recall()))

  hist=ResNetFinalModel.fit_generator(train_generator,epochs=epochs,callbacks=[learningratesch,modelcheckpoint,earlystop],steps_per_epoch=stepsperepoch,validation_data=validation_generator,validation_steps = validationsteps)


  preds = ResNetFinalModel.evaluate(validation_generator)
  print ("Validation Loss = " + str(preds[0]))
  print ("Validation Accuracy = " + str(preds[1]))

  plt.plot(hist.history['accuracy'])
  plt.plot(hist.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(["Train_acc","Validation_acc"])
  plt.show()

  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(["Train_loss","Validation Loss"])
  plt.show()


  plt.rcParams.update({'font.size': 16})
  predictions = ResNetFinalModel.predict(validation_generator)
  predictions = np.argmax(predictions,axis=1)
  predictions[:15]
  cm = confusion_matrix(validation_generator.classes, predictions)
  cm = pd.DataFrame(cm, index = ['0', '1', '2'], columns = ['0', '1', '2'])
  cm
  class_names = ['COVID','NORMAL','VIRAL PNEUMONIA']
  def plot_confusion_matrix (cm):
      plt.figure(figsize = (10,10))
      sns.heatmap(
          cm, 
          cmap = 'Blues', 
          linecolor = 'black', 
          linewidth = 1, 
          annot = True, 
          fmt = '', 
          xticklabels = class_names, 
          yticklabels = class_names)
      plt.show()
      
  plot_confusion_matrix(cm)


Alexnet()

ResNet()
FinalResNet()