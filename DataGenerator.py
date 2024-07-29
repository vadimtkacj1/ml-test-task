
import numpy as np
import cv2
from tensorflow import keras

from decoders import rle_decode

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, path, csv_file, batch_size=8, image_size=768):
        self.data = data
        self.path = path
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.data):
            self.batch_size = len(self.data) - index*self.batch_size
        
        files_batch = self.data[index*self.batch_size : (index+1)*self.batch_size]
        
        x = []
        y = []

        
        for name in files_batch:
            image_path = str(self.path + "\\"+ name)

            img = cv2.imread(image_path)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img / 255.0 

            all_masks = self.csv_file[self.csv_file['ImageId'] == name].EncodedPixels
            mask = np.zeros((self.image_size, self.image_size))

            for encoded_pixels_mask in all_masks:
                decoded_mask = rle_decode(encoded_pixels_mask)
                mask = mask + cv2.resize(decoded_mask, (self.image_size, self.image_size))


            mask = mask.astype(np.float32)

            x.append(img)
            y.append(mask)

        return np.array(x), np.array(y)
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
