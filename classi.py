import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from keras.utils import np_utils
from sklearn import metrics

print(os.listdir('./DOGCAT'))

# Оформляем пути к тренировочным и тестовым изображениям котов и собак
path_dogs = './DOGCAT/training_set/dogs/'
path_test_dogs = './DOGCAT/test_set/dogs/'
path_cats = './DOGCAT/training_set/cats/'
path_test_cats = './DOGCAT/test_set/cats/'

train_dogs = os.listdir(path_dogs)
test_dogs = os.listdir(path_test_dogs)
train_cats = os.listdir(path_cats)
test_cats = os.listdir(path_test_cats)


# Удаляем во всех папках файл '.DS_Store'!
if '.DS_Store' in train_dogs:
	train_dogs.remove('.DS_Store')
if '.DS_Store' in test_dogs:
	test_dogs.remove('.DS_Store')
if '.DS_Store' in train_cats:
	train_cats.remove('.DS_Store')
if '.DS_Store' in test_cats:
	test_cats.remove('.DS_Store')

# Так как все изображения имеют разные размеры, то - 
input_shape = (224,224,3)
num_classes = 2 

# Создаем списки для хранения 1d изображений, и стандартный размер изображения
img_resize = (224,224)
dogs_train = []
cats_train = []
dogs_test = []
cats_test = []

# На каждом изобрежении проводим изменение размера и трансформацию в 1d вид
for dog in train_dogs:
	if len(dogs_train) > 500:
		break
	img = Image.open(path_dogs + dog)
	img = img.resize(img_resize)
	img = np.asarray(img)
	# Добавление всех фотографий(в 1d виде!) с собаками в список dogs_train
	dogs_train.append(img)


for cat in train_cats:
	if len(cats_train) > 500:
		break
	img = Image.open(path_cats + cat)
	img = img.resize(img_resize)
	img = np.asarray(img)
	# Добавление всех фотографий(в 1d виде!) с котами в список cats_train
	cats_train.append(img)

for dog in test_dogs:
	if len(dogs_test) > 100:
		break
	img = Image.open(path_test_dogs + dog)
	img = img.resize(img_resize)
	img = np.asarray(img)
	# Добавление всех фотографий(в 1d виде!) с собаками в список dogs_test
	dogs_test.append(img)

for cat in test_cats:
	if len(cats_test) > 100:
		break
	img = Image.open(path_test_cats + cat)
	img = img.resize(img_resize)
	img = np.asarray(img)
	# Добавление всех фотографий(в 1d виде!) с котами в список cats_test
	cats_test.append(img)


print(cats_test[0].shape,'RAZMER')
# Создаем списки, соответсвующие 1d изображениям, с метками(названиями животных)
dogs_train_labels = [0 for i in range (len(dogs_train))]
dogs_test_labels = [0 for i in range(len(dogs_test))]
cats_train_labels = [1 for i in range(len(cats_train))]
cats_test_labels = [1 for i in range(len(cats_test))]

# Добавляем лейблы котов к лейблам собак в тестовом и тренировочном наборе
dogs_train_labels.extend(cats_train_labels)
dogs_test_labels.extend(cats_test_labels)
# Создаем матрицу 8000x2, 1-0 - собака, 0-1 - кот, как обработка категори -
# альных атрибутов через OneHotEncoder
label_train = np_utils.to_categorical(dogs_train_labels,num_classes)
label_test = np_utils.to_categorical(dogs_test_labels,num_classes)

dogs_train.extend(cats_train)
data_train = np.array(dogs_train,dtype=np.float32)
# Нормализуем значения
data_train /= 255
# Далее просто случайным образом перемешиваем даннные
# Изображения и их лейблы зашафлены одинаковым образом!
index = np.arange(len(data_train))
np.random.shuffle(index)
data_train = data_train[index]
label_train = label_train[index]

# Тоже самое делаем и для испытательного набора
dogs_test.extend(cats_test)
data_test = np.array(dogs_test,dtype=np.float32)
data_test /= 255
index = np.arange(len(data_test))
np.random.shuffle(index)
data_test = data_test[index]
label_test = label_test[index]

print(data_test.shape,'ASDASDASD')
print(label_test.shape,'LABEL')
print(label_test)

# Удаление одномерной записи из формы
data_train = np.squeeze(data_train)
data_test = np.squeeze(data_test)
label_train = np.squeeze(label_train)
label_test = np.squeeze(label_test)

print(label_test)
print(label_train)
# Делаем 1д вектор, 1- собака, 0 - кошка
label_train = np.delete(label_train,-1,axis=1).reshape(-1,)
label_test = np.delete(label_test,-1,axis=1).reshape(-1,)
print(label_test)
print(label_train)


print('Training label set:',label_train.shape)
print('Testing label set:',label_test.shape)
print('Training data set:',data_train.shape)
print('Testing data set:',data_test.shape)



# Пишем архитектуру AlexNet сверточной нейронной сети!

# Определяем размеры входных данных и количество эпох
dim = data_train.shape[1]
channels = data_train.shape[3]
n_epochs = 100
# Определяем узлы-заполнители
X = tf.placeholder(tf.float32,shape=(None,dim,dim,channels))
y = tf.placeholder(tf.int32,shape=(None))

# Основная часть сверточной нейронной сети
with tf.name_scope('AlexNet'):
	cnn1 = tf.layers.conv2d(X,filters=3,kernel_size=[3,3],
									padding='same',activation=tf.nn.relu)
	# Добавляем ЛОКАЛЬНУЮ НОРМАЛИЗАЦИЮ
	lrn1 = tf.nn.local_response_normalization(cnn1)
	maxpooling2 = tf.layers.max_pooling2d(lrn1,pool_size=[3,3],strides=2)
	cnn3 = tf.layers.conv2d(maxpooling2,filters=5,kernel_size=[5,5],
									padding='same',activation=tf.nn.relu)
	lrn3 = tf.nn.local_response_normalization(cnn3)
	maxpooling4 = tf.layers.max_pooling2d(lrn3,pool_size=[3,3],strides=2)
	cnn5 = tf.layers.conv2d(maxpooling4,filters=384,kernel_size=[3,3],
									padding='same',activation=tf.nn.relu)
	cnn6 = tf.layers.conv2d(cnn5,filters=384,kernel_size=[3,3],
									padding='same',activation=tf.nn.relu)
	cnn7 = tf.layers.conv2d(cnn6,filters=256,kernel_size=[3,3],
									padding='same',activation=tf.nn.relu)
	flatten = tf.contrib.layers.flatten(cnn3)
	dense8 = tf.layers.dense(flatten,units=4096,activation=tf.nn.relu)
	dropout = tf.layers.dropout(dense8,rate=0.55)
	dense9 = tf.layers.dense(dropout,units=4096,activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(dense9,rate=0.55)
	logits = tf.layers.dense(dropout1,units=2)

# Все остальное стандартно для нейронной сети
with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																labels=y)
	loss = tf.reduce_mean(xentropy)

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits,y,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

# Инициализатор всех переменных и класс Saver, позволяющий сохранять модель НС
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Это переменная для хранения максимальной точности СНН
minimal = 0

# Сессия, во время которой происходит обучение и оценка точности
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		sess.run(training_op,feed_dict={X:data_train,y:label_train})
		acc_val = accuracy.eval(feed_dict={X:data_test,y:label_test})
		print('This is my accuracy on the test set:',acc_val)
		if acc_val > minimal:
			minimal = acc_val
			save_path = saver.save(sess,'./animal_classifier.ckpt')

print('This is the saved CNN(accuracy):',minimal)
'''
# Загружаем модель и делаем предсказания!
with tf.Session() as sess:
	saver.restore(sess,'./animal_classifier.ckpt')
	Z = logits.eval(feed_dict={X:data_test,y:label_test})
	y_pred = np.argmax(Z,axis=1)

# Определение точности
metrics = metrics.accuracy_score(label_test,y_pred)
print('This is my final test score:',metrics)
'''


