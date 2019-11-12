from config import MODEL_CONFIG, INPUT_CONFIG
import tensorflow as tf
# import keras
# from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import model_from_json, Model


class Model():
	def __init__(self):
		self.image_height = INPUT_CONFIG['image_height']
		self.image_width = INPUT_CONFIG['image_width']
		self.channels = INPUT_CONFIG['image_channels']
		self.num_classes = MODEL_CONFIG['num_classes']
		self.optimizer = MODEL_CONFIG['optimizer']
		self.pooling = MODEL_CONFIG['pooling']

	#Dummy Custom Model
	def custom_model(self):
		# model = keras.Sequential()
		# model.add(Convolution2D(32, (3,3), padding="same", input_shape =[self.image_height,self.image_width,self.channels],
		# 						 activation = "relu"))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(keras.layers.Dropout(0.5))
		# model.add(Convolution2D(64, (3,3), padding="same", activation = "relu"))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(keras.layers.Dropout(0.5))
		# model.add(Convolution2D(128, (3,3), padding="same", activation = "relu"))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(keras.layers.Dropout(0.5))
		# model.add(Flatten())
		# model.add(keras.layers.Dropout(0.5))

		# adam = keras.optimizers.adam(lr=2e-5)
		# model.compile(optimizer=adam,
        #   loss='binary_crossentropy',
        #   metrics=['accuracy'])

		# return model
		IMG_SHAPE = (INPUT_CONFIG['image_width'], INPUT_CONFIG['image_height'], 3)

		base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
													include_top=False, 
													weights='imagenet')
		base_model.trainable = False
		# Trainable classification head
		maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
		prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

		learning_rate = 0.0001

		model = tf.keras.Sequential([
			base_model,
			maxpool_layer,
			prediction_layer
		])

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
					loss='binary_crossentropy',
					metrics=['accuracy'])

		return model

	# def base_model(self, base_name_arg = None):
	# 	if base_name_arg is not None:
	# 		base_name = base_name_arg

	# 	base_name = INPUT_CONFIG['base_model']
	# 	base_model = None

	# 	if base_name.find(inception) != -1:
	# 		if base_name.find('3') != -1:
	# 			base = keras.applications.inception_v3.InceptionV3(include_top=False, weights="imagenet",
	# 										 input_tensor=None, input_shape=(self.image_height,self.image_width,self.channels),
	# 										  pooling=None, classes=self.num_classes)
	# 		elif base_name.find('2') != -1:
	# 			pass
	# 	elif base_name.find(vgg) != -1:
	# 		pass

	# 	return base	
		

	# def get_pooling_layer(self  name = None):
	# 	pool_layer = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	# 	if name is not None:
	# 		self.pooling = name

	# 	if self.pooling == 'maxpooling2d':
	# 		pool_layer = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	# 	elif self.pooling == 'globalaveragepooling2D':
	# 		pool_layer = keras.layers.GlobalAveragePooling2D(data_format=None)

	# 	return pool_layer

	# def get_optimizer(self, name = None):
	# 	if name is not None:
	# 		self.optimizer = name

	# 	optimizer = keras.optimizers.adam(lr=2e-5)
		
	# 	if self.optimizer == 'adam':
	# 		optimizer = keras.optimizers.adam(lr=2e-5)
	# 	elif self.optimizers = 'rmsprop':
	# 		optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	# 	elif self.optimizer = 'adagrad':
	# 		optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
	# 	elif self.optimizer = 'sgd':
	# 		optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
	# 	elif self.optimizer = 'adadelta':
	# 		optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
	# 	elif self.optimizer = 'adamax':
	# 		optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		
	# 	return optimizer


