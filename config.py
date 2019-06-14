MODEL_CONFIG = {
	'optimizer': 'adam' , # rmsprop , 
	'base_model': 'vgg16', # vgg19, inception
	'use_custom': False, # True, False
	'classes': 0, #1,2,3,4,5,....inf
	'base_model_trainable': False, #True, False
	

}

INPUT_CONFIG = {
	'image_height' : 0,
	'image_width' : 0,
	'image_channels': 0, #1,2,3,4

}

TRAIN_CONFIG = {
	
}

TEST_CONFIG = {
	
}