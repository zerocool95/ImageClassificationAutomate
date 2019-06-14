from config import MODEL_CONFIG, INPUT_CONFIG

import keras
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json, Model


class Model():
	def __init__(self):
		self.image_height = INPUT_CONFIG['image_height']
		self.image_width = INPUT_CONFIG['image_width']
		self.channels = INPUT_CONFIG['channels']
		self.num_classes = MODEL_CONFIG['num_classes']

		
	def custom_model(self):
		pass

	def base_model(self):
		base_name = INPUT_CONFIG['base_model']
		base_model = None

		if base_name.find(inception) != -1:
			if base_name.find('3') != -1:
				base = keras.applications.inception_v3.InceptionV3(include_top=False, weights="imagenet",
											 input_tensor=None, input_shape=(self.image_height,self.image_width,self.channels),
											  pooling=None, classes=self.num_classes)
			elif base_name.find('2') != -1:
				pass

		return base
		

	def pooling_layer(self):
		pass

	def optimizer(self):
		pass

