import cv2
import tensorflow as tf
from config import INPUT_CONFIG

class PreProcess():
	def __init__(self):
		self.image_height = INPUT_CONFIG['image_height']
		self.image_width = INPUT_CONFIG['image_width']

	#Dummy Preprocess function
	def preprocess_image(self):
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    img = img/255. 

	    return img