import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from keras.utils import np_utils
from sklearn import metrics
from tensorflow import keras



def dsstore(list_of_files):
    ''' function that delete all .DS_Store files in a directory! '''
    if '.DS_Store' in list_of_files:
        list_of_files.remove('.DS_Store')
    return list_of_files

def img_reshape(size=(224,224),image='',path='./'):
    ''' reshapes image into chosen size '''
    # just for comfort.
    if image == '':
        img = Image.open(path)
    else:
        img = Image.open(path + image)
    img = img.resize(size)
    img = np.asarray(img)

    return img

answer = input('Do you want to train your network yourself?(y/n)')
if answer == 'y':
    # Writing paths for training and test images
    path_dogs = './DOGCAT/training_set/dogs/'
    path_test_dogs = './DOGCAT/test_set/dogs/'
    path_cats = './DOGCAT/training_set/cats/'
    path_test_cats = './DOGCAT/test_set/cats/'

    # Deleting all .DS_Store's 
    train_dogs = dsstore(os.listdir(path_dogs))
    test_dogs = dsstore(os.listdir(path_test_dogs))
    train_cats = dsstore(os.listdir(path_cats))
    test_cats = dsstore(os.listdir(path_test_cats))

    # Reshape form
    input_shape = (224,224,3)
    num_classes = 2

    # Creating and filling lists with 1d-images(reshaped) 
    img_resize = (224,224)
    dogs_train = [img_reshape(img_resize,image,path_dogs) 
                                    for image in train_dogs[:500]]
    cats_train = [img_reshape(img_resize,image,path_cats) 
                                    for image in train_cats[:500]]
    dogs_test = [img_reshape(img_resize,image,path_test_dogs) 
                                    for image in test_dogs[:100]]
    cats_test = [img_reshape(img_resize,image,path_test_cats) 
                                    for image in test_cats[:100]] 
    print(cats_test[0].shape)
    # Creating lists with labels(animals) for our data
    dogs_train_labels = [0 for i in range(len(dogs_train))]
    dogs_test_labels = [0 for i in range(len(dogs_test))]
    cats_train_labels = [1 for i in range(len(cats_train))]
    cats_test_labels = [1 for i in range(len(cats_test))]
    # Stacking labels for train and test ones together
    dogs_train_labels.extend(cats_train_labels)
    dogs_test_labels.extend(cats_test_labels)
    #
    label_train = np_utils.to_categorical(dogs_train_labels,num_classes)
    label_test = np_utils.to_categorical(dogs_test_labels,num_classes)
    # Doing the same thing for actual train images
    dogs_train.extend(cats_train)
    data_train = np.array(dogs_train,dtype=np.float32)
    # Normalizing values
    data_train /= 255
    # Shuffling our data!
    index = np.arange(len(data_train))
    np.random.shuffle(index)
    data_train = data_train[index]
    label_train = label_train[index]

    # Doing the same thing for our TEST images!
    dogs_test.extend(cats_test)
    data_test = np.array(dogs_test,dtype=np.float32)
    data_test /= 255
    index = np.arange(len(data_test))
    np.random.shuffle(index)
    data_test = data_test[index]
    label_test = label_test[index]

    # Deleting 1-d form
    data_train = np.squeeze(data_train)
    data_test = np.squeeze(data_test)
    label_train = np.squeeze(label_train)
    label_test = np.squeeze(label_test)

    # I don't want to work with OneHotEncoded data, so deleting the last 
    # column in labels data to make it just a 1-d vector
    # 1 - dog, 0 - cat
    label_train = np.delete(label_train,-1,axis=1).reshape(-1,)
    label_test = np.delete(label_test,-1,axis=1).reshape(-1,)

    # Testing everything!
    '''
    print('Training label set:',label_train.shape)
    print('Testing label set:',label_test.shape)
    print('Training data set:',data_train.shape)
    print('Testing data set:',data_test.shape)
    '''
    # I will use TRANSFER LEARNING WITH KERAS INSTEAD!!!!
    # Creating "base" model from the pre-trained MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    feature_image = base_model(data_train)
    # Freezing pre-trained layers!
    base_model.trainable = False
    # Pooling with keras...
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_average = global_average_layer(feature_image)
    # Creating dense network to form our predictions:
    prediction_layer = keras.layers.Dense(1)
    prediction = prediction_layer(feature_average)
    # Stacking feature extractor,pooling and prediction layer into our model!
    model = tf.keras.Sequential([
                base_model,
                global_average_layer,
                prediction_layer
                ])
    # Compiling the model!
    lr = 0.001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    initial_epochs = 5 
    # Transofrming data!!!!£$12341241234
    #data_train = tf.data.Dataset.from_tensor_slices((data_train,label_train))
    #data_test = tf.data.Dataset.from_tensor_slices((data_test,label_test))
    # Initial accuracy 0.28, training net!
    history = model.fit(data_train,label_train,epochs=initial_epochs,
                        validation_data=(data_test,label_test),
                        batch_size=50)
    # Saving API in Keras:
    saving_path = input('Where do you want me to save the network ?(quit)')
    if saving_path != 'quit':
        model.save(saving_path)
        print('Saved this neural network')
    else:
        print('I did not save the network')
