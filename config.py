MODEL_CONFIG = {
	'optimizer': 'adam' , # rmsprop , sgd, adam, adagrad, adadelta, adamax, nadam
	'base_model': 'vgg16', # vgg19, inception
	'use_custom': False, # True, False
	'num_classes': 2, # 1,2,3,4,5,....inf
	'base_model_trainable': False, # True, False
	'pooling': 'maxpooling2d', # globalaveragepooling2D ,  https://keras.io/layers/pooling/
	'batch_size': 64
}

INPUT_CONFIG = {
	'image_height' : 200,
	'image_width' : 500,
	'image_channels': 0, #1,2,3,4

}

TRAIN_CONFIG = {
	'train_folder': 'data/train',
	'use_custom_gen': True, #True, False
	'train_test_split': 0.8, #Split percent for train and val data
	'epochs': 30,
	'validation_steps': 20
}

TEST_CONFIG = {
	
}