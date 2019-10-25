# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:06:55 2019

@author: tatras
"""

import wavio
import pandas as pd
import os
import numpy as np
import librosa  
import tqdm as tq
train_dir = "D:/songs"
categories = os.listdir(train_dir)

tr = []
y_data = []
def create_training_data():
    ##tr = []
    ##k=0
    for category in categories:  
        path = os.path.join(train_dir,category)  
        class_num = categories.index(category)  
        for audio in (os.listdir(path)): 
            file_name = os.path.join(os.path.abspath(path), audio)
            try:
                audio_array,sample_rate = librosa.load(file_name, res_type='kaiser_fast')
                ##mfcc = nlibrosa.feature.mfcc(y=audio_array, sr=sample_rate)
                tr.append(audio_array)
                y_data.append(class_num)
                ##training_data.append([audio_array, class_num])
                ##print(type(audio_array))
            except Exception as e:  
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,audio))

create_training_data()

x_data = [np.mean(librosa.feature.mfcc(y=x).T,axis=0) for x in tr]

x_data_training = np.zeros((len(x_data),x_data[0].shape[0]))
for i in range(len(x_data)):
    x_data_training[i] = x_data[i]

"""    
def parser(row, file):
   file_name = os.path.join(os.path.abspath(path), file, str(row.ID) + '.wav')
   try:
       X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
       mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
       print("Error encountered while parsing file: ", file_name)
       return None, None
    
   data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
   mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0) 
   feature = mfccs   
   try:
       labels = row.Class
   except Exception as e:
       print("ID not found since its a test sample")
       return [feature], print(file_name)
   return [feature, labels], print(file_name)
"""    

"""
def get_test_values(test):
    temp = test[].apply(parser, args=["Test"], axis=1)
    temp = np.array(temp).tolist()
    temp_features = []   
    for i in temp:
        temp_features.append(i[0])
    temp_features = np.array(temp_features)
    temp_features = np.squeeze(temp_features)    
    temp_features= pd.DataFrame(temp_features)
    temp_features.dropna(inplace=True)
    temp_features = np.array(temp_features).tolist()
    features = []
    for i in temp_features:
        features.append(i)
    features= np.array(features)
    return features
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import cv2
# from tqdm import tqdm

# train_dir = "D:\genres"
# categories = os.listdir(train_dir)
# for category in CATEGORIES:  
#     path = os.path.join(train_dir,category)  
#     for audio in os.listdir(path):  
#         img_array = cv2.imread(os.path.join(path,img))  
#         plt.imshow(img_array, cmap='gray')  
#         plt.show() 
#         break  
#     break

#     try:
#        X, sample_rate = librosa.load(os.path.join(path,audio), res_type='kaiser_fast') 
#        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
#    except Exception as e:
#        print("Error encountered while parsing file: ", file_name)
#        return None, None
    
#    data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
#    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0) 
#    feature = mfccs   
#    try:
#        labels = row.Class
#    except Exception as e:
#        print("ID not found since its a test sample")
#        return [feature], print(file_name)
#    return [feature, labels], print(file_name)
#     for img in os.listdir(path):  
#         img_array = cv2.imread(os.path.join(path,img))  
#         plt.imshow(img_array, cmap='gray')  
#         plt.show() 
#         break  
#     break


# training_data = []
# def create_training_data():
#     for category in categories:  
#         path = os.path.join(train_dir,category)  
#         class_num = categories.index(category)  
#         for img in tqdm(os.listdir(path)):  
#             try:
#                 x, sr  = librosa.load(os.path.join(path,img))  
#                 mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T,axis=0) 
 
#                 training_data.append([append, class_num])  
#             except Exception as e:  
#                 pass
#             except OSError as e:
#                 print("OSErrroBad img most likely", e, os.path.join(path,img))
#             except Exception as e:
#                 print("general exception", e, os.path.join(path,img))
# create_training_data()
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder



def train_model(Xtrain=None, Ytrain=None, acti_lay1=None, acti_lay2=None, acti_out=None, hl1_shape=None, hl2_shape=None, batch=None, epoch=None,shuffle=True):
    Y2 = keras.utils.to_categorical(Ytrain, num_classes=10,dtype=int)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hl1_shape, input_shape=(Xtrain.shape[1],),activation=acti_lay1))
    #model.add(keras.layers.Activation(acti_lay1))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(hl2_shape,activation=acti_lay2))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(32,activation='relu'))

    ##model.add(keras.layers.Activation(acti_lay2))
    model.add(keras.layers.Dense(Y2.shape[1],activation=acti_out))
    #model.add(keras.layers.Activation(acti_out))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
    #print(Y2.shape)
    #print(Xtrain.shape)
    ##print("cocacola")
    history = model.fit(Xtrain, Y2, batch_size=batch, epochs=epoch,shuffle=shuffle)
    ##print("cocacola")    
    score = model.evaluate(Xtrain, Y2)        
    #prediction = model.predict(X_test)    
    return history,score



x_data_training = pd.DataFrame(x_data_training)
y_data = pd.DataFrame(y_data)
x_data_training['classes'] = y_data
from sklearn.utils import shuffle
x_training_data = shuffle(x_data_training)

# train = pd.DataFrame(training_data)
# y_train = train.iloc[:,1]
# x_train = train.iloc[:,0]
x_train = x_training_data.iloc[:, :20]
y_train = x_training_data.iloc[:, 20]
# x_np_array = np.array(x_train)
# x_train = np.zeros((x_np_array.shape[0],x_np_array[0].shape[0]))
# for i in range(x_np_array.shape[0]):
#     for j in range(x_np_array[0].shape[0]-1):
#         x_train[i][j] = x_np_array[i][j]
    
    
train_model(train_model(Xtrain=x_train, Ytrain=y_train,acti_lay1='relu', acti_lay2='relu', acti_out='softmax', hl1_shape=1024, hl2_shape=512, batch=8, epoch=90))

# def submission(X_test=None, prediction=None):
#     from sklearn.preprocessing import LabelEncoder
#     lb = LabelEncoder()     
#     pred_labels = np.zeros((X_test.shape[0], ))    
#     for i in range(X_test.shape[0]):
#         pred_labels[i] = np.argmax(prediction[i, :])    
#         pred_labels = pred_labels.astype(int)
#     lb.fit(Y_train)
#     labels = lb.inverse_transform(pred_labels)
#     Y_test = pd.DataFrame(columns = ["ID", "Class"])  
#     Y_test["ID"] = train["ID"]  
#     Y_test["Class"] = pd.DataFrame(labels)    
#     Y_test.to_csv(path+"submission.csv")    
#     return Y_test


# X_train, Y_train = get_train_values(train)
# X_test = get_test_values(test)
# history, score, prediction = train_model(X_train, Y_train, X_test, "sigmoid", "sigmoid", "softmax", 256, 128, 32, 500)
# Y_test = submission(X_test, prediction)



