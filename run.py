import os
import tensorflow as tf
from PIL import Image
import numpy as np
from classifier import img_reshape
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


def think():
    ''' lol '''
    print('THINKING')
    print('THINKING')
    print('THINKING')
    print('THINKING')
    return None

# Hiding tensorflow warnings!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = input('Type full path to a choosen photo:\t')
think()
# preprocesing image
image = img_reshape(path=path)
# bad function! 
image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
#print(image.shape)

# Loading our model:
# Avoiding the bag!
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('./mobile_netv2/model.h5')
#model.summary()
y_pred = model.predict(image)
#print(y_pred)

if int(y_pred) == 1:
    print('I think this is a dog!')
else:
    print('I think this is a cat!')
