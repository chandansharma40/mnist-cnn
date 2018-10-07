import numpy as np
import gzip

class DataFunc(object):
    def _load_label(self, file_name):
        file_path = "/home/chandan/Neural/ece542-2018fall/project/03/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
        # print("Done")

        return np.array(labels)

    def _load_img(self, file_name):
        #file_path = dataset_dir + "/" + file_name
        file_path = "/home/chandan/Neural/ece542-2018fall/project/03/" + file_name
        
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 784)
        # print("Done")

        return np.array(data)