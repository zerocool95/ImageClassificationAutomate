import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from config import INPUT_CONFIG, MODEL_CONFIG
class Dataset():
    def __init__(self):
        pass

    def get_labels_from_folder(self, base_folder):
        base = base_folder
        file_names = []
        file_labels = []
        lb = 0
        for folder in os.listdir(base):
            for f in os.listdir(base + folder):
                file_names.append(os.path.join(base,folder,f))
                file_labels.append(lb)
            lb += 1
        
        return file_names, file_labels
            

    def read_data(self, mode = 'folder', tts = 'automatic', train_path = "data/train/", test_path = "data/test/"): # mode: csv , tts : automatic, manual
        if mode == 'folder':
            # Prepend image filenames in train/ with relative path
            
            file_name_dict = {} # filename : label
            file_names, file_labels = self.get_labels_from_folder(train_path)
                    
            if tts == 'automatic':
                train_filenames, val_filenames, train_labels, val_labels = train_test_split(file_names,
                                file_labels,
                                train_size=0.9,
                                random_state=42)
            else:
                train_filenames, train_labels = file_names, file_labels
                val_filenames, val_labels = self.get_labels_from_folder(test_path)

        INPUT_CONFIG['num_train_samples'] = len(train_filenames)

        

        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames),
        tf.constant(train_labels))).map(self._parse_fn).shuffle(buffer_size=10000).batch(MODEL_CONFIG["batch_size"])


        val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames),
        tf.constant(val_labels))).map(self._parse_fn).batch(MODEL_CONFIG["batch_size"])

        return train_data, val_data

    
    def _parse_fn(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img)
        img = (tf.cast(img, tf.float32)/127.5) - 1
        img = tf.image.resize(img, (INPUT_CONFIG["image_width"], INPUT_CONFIG["image_height"]))
        return img, label
