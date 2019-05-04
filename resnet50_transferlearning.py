import os
import numpy as np
from resnet50 import ResNet50
import time
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D,Dense,Dropout,Activation,Flatten,Input
from keras.models import load_model
from imagenet_utils import preprocess_input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py

#Load datatraining
PATH = os.getcwd()
#Define datapath 
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
training_path = data_path + '/trainingset'

img_data_list =[]

for trainingset in data_dir_list :
    img_list = os.listdir(training_path)
    print('Loaded the images in trainingset folder:{}\n'.format(trainingset))
    for img in img_list:
        img_path = training_path + '/' + img
        img = image.load_img(img_path,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        x = preprocess_input(x)
        print('Input image shape: ',x.shape )
        img_data_list.append(x)
    
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = np.rollaxis(img_data,1,0)
#print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)

#Define number of classes 
num_classes = 3 

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

# fix it in data 
labels[0:130] = 0
labels[131:274] =1 
labels[274:] =2 

names = ['projector_problems','toilet_problems','sanitation_problems']

#convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels,num_classes)

#Shuffle dataset
x,y = shuffle(img_data,Y,random_state =2)
#Split dataset 
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=2)
###################################################################

#Build model based on pre-trained resnet50 model
#Training classifier alone
image_input = Input(shape=(224,224,3))
import keras
model = keras.applications.resnet50.ResNet50(input_tensor=image_input,include_top = True, weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out = Dense(num_classes,activation='softmax',name ='output_layer')(x)
resnet_model = Model(inputs= image_input,outputs = out)
resnet_model.summary()

for layer in resnet_model.layers[:-1]:
    layer.trainable= False

resnet_model.layers[-1].trainable 

resnet_model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
t = time.time()
hist = resnet_model.fit(X_train,y_train,epochs = 10, batch_size =34,verbose =1, validation_data =(X_test,y_test))
print('Training time: %s'%(t-time.time()))

#Evaluate model 
(loss,accuracy) = resnet_model.evaluate(X_test,y_test,batch_size = 5 ,verbose =0 )
print('[INFO] loss={:.4f}, accuracy={:.4f}',format(loss,accuracy*100))
#######################################################################
# Visualize loss,accuracy graph
import matplotlib.pyplot as plt
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(10)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#Test new image
import cv2
#test_image = cv2.imread('data/testset/toilet89.jpg')
#test_image=cv2.resize(test_image,(224,224))
#test_image = np.array(test_image)
#test_image = test_image.astype('float32')
#test_image /= 255
#print (test_image.shape)
test_image = cv2.imread('data/trashtest/trash131.jpg')
test_image=cv2.resize(test_image,(224,224))
x = image.img_to_array(test_image)
x = np.expand_dims(x,axis = 0)
x = preprocess_input(x)
print('Input image shape: ',x.shape )

# Predicting the test image
print((resnet_model.predict(x)))
from keras.models import Sequential 
model_predict = Sequential()
model_predict.add(resnet_model)
print(model_predict.predict_classes(x))

# serialize model to JSON
ResNet50_model_json = resnet_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(ResNet50_model_json)
# serialize weights to HDF5
resnet_model.save_weights("WeightsResNet50_model.h5")
print("Saved model to disk")

# load json and create model
from keras.models import model_from_json 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
loaded_model1.summary()
# load weights into new model
loaded_model1.load_weights("WeightsResNet50_model.h5")
print("Loaded model from disk")
print((loaded_model1.predict(x)))
resnet_model.save('ResNet50_model.hdf5')
loaded_model=load_model('ResNet50_model.hdf5')
loaded_model.summary()
print((loaded_model.predict(x)))
 


