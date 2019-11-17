from config import MODEL_CONFIG, INPUT_CONFIG, TRAIN_CONFIG
from model import Model
from random import shuffle 
from dataset import Dataset
import os
from versioning.versioning import Versioning

class Trainer():	
	def __init__(self):
		self.image_height = INPUT_CONFIG['image_height']
		self.image_width = INPUT_CONFIG['image_width']
		self.dataset = Dataset()
		self.ver = Versioning()
		
	def train(self):
		#Load Data
		train_data, val_data = self.dataset.read_data()
		steps_per_epoch = round(INPUT_CONFIG['num_train_samples'])//MODEL_CONFIG['batch_size']
		
		md_obj = Model()
		model = md_obj.custom_model()
		history = model.fit(train_data.repeat(),
                    epochs=TRAIN_CONFIG['epochs'],
                    steps_per_epoch = steps_per_epoch,
                    validation_data=val_data.repeat(), 
                    validation_steps=TRAIN_CONFIG['validation_steps'])

		hist_name = self.ver.write_history_to_disk(history)
		model.save_weights("data/model_meta/model_weights/" + hist_name.split('_')[0] + '_weights_final.h5')

	# def get_train_meta(self):
	# 	base = TRAIN_CONFIG['train_folder']
	# 	classes = os.listdir(base)
	# 	labels = {}
	# 	i = 0 

	# 	for c in classes:
	# 	    c_images = os.listdir(base+c)

	# 	    for image in c_images:
	# 	        labels[c+'/'+image] = i
	# 	    i+=1

	#     print('Total Classes found :  ', len(classes))
	#     print('Total Images found : ', len(labels))

	#     return labels

	# def train_model(self):
	# 	batch_size = MODEL_CONFIG['batch_size']

	# 	labels = get_train_meta()
	# 	img_ids = list(labels.keys())
	# 	shuffle(img_ids)

	# 	split = int(TRAIN_CONFIG['train_test_split'] * len(img_ids))

	# 	train_ids = img_ids[0:split]
	# 	valid_ids = img_ids[split:]

	# 	if TRAIN_CONFIG['use_custom_gen']:
	# 		train_generator = image_generator(train_ids, batch_size = batch_size)
	# 		valid_generator = image_generator(valid_ids, batch_size = batch_size)
	# 	else:
	# 		# train_generator = train_datagen.flow_from_directory(
	# 		#     TRAIN_DIR,
	# 		#     target_size=(HEIGHT, WIDTH),
	# 		#     batch_size=BATCH_SIZE,
	# 		#     color_mode='grayscale',
	# 		#     class_mode='categorical')
			    
	# 		# validation_generator = validation_datagen.flow_from_directory(
	# 		#     TEST_DIR,
	# 		#     target_size=(HEIGHT, WIDTH),
	# 		#     color_mode='grayscale',
	# 		#     batch_size=BATCH_SIZE,
	# 		#     class_mode='categorical')
	# 		pass

	# 	train_steps = len(train_ids) // batch_size 
	# 	valid_steps = len(valid_ids) // batch_size 



	# def custom_image_generator(input_ids, batch_size = 32):
	#     while True:
	#         batch_paths = np.random.choice(a= input_ids, size = batch_size + 50)

	#         batch_input = []
	#         batch_output = []
	        
	#         cnt = 0
	#         id_cnt = 0
	#         while cnt < batch_size :
	#             input = cv2.imread(base+batch_paths[id_cnt])
	            
	#             if input is None:
	#                 id_cnt += 1
	#                 continue
	#             output = labels[batch_paths[id_cnt]]

	#             input = preprocess_image(input)

	#             batch_input += [np.reshape(input, (self.image_height, self.image_width, 1))]
	#             batch_output += [output]
	#             cnt += 1
	#             id_cnt += 1

	#         batch_x = np.array(batch_input)
	#         batch_y = np.array(batch_output)

	#         yield (batch_x, batch_y)

