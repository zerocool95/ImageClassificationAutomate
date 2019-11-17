from datetime import datetime
import tensorflow as tf
import os
import pickle

class Versioning():
    def __init__(self):
        self.setup_folders()

    def version(self):
        pass

    def setup_folders(self):
        paths = ['data/model_meta/', 'data/model_meta/model_history', 'data/model_meta/model_weights']
        for path in paths:
            if os.path.exists(path) == False:
                os.mkdir(path)

    def write_history_to_disk(self, file_obj):

        cnt = 1
        for d in os.listdir('data/model_meta/model_history'):
            ver_num = d.split("_")[0][-1]
            cnt = max(int(ver_num), cnt)
        cnt += 1

        hist_name = "v" + str(cnt) + "_" + str(datetime.now().date())
        hist_path = "data/model_meta/model_history/" + hist_name
        with open(hist_path, 'wb') as f:
            pickle.dump(file_obj.history, f) 
        # tf.io.write_file(hist_name, file_obj, name= "v" + str(cnt) + "_" + str(datetime.now().date()))

        return hist_name
