'''
Image preprocessing functions
'''

#import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
#import matplotlib.pyplot as plt

# ************************************************************************************** 
#                                   FUNCTIONS:
# **************************************************************************************

def image_loader(BASE_DIR, load_sets=(True, True, True), batch_size=32, img_dim=(224,224), seed=42):
    
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VALIDATION_DIR = os.path.join(BASE_DIR, 'valid')
    TEST_DIR = os.path.join(BASE_DIR, 'test')
    
    data_list = []
    
    try:
        if load_sets[0] == True:
            train_ds = image_dataset_from_directory(TRAIN_DIR, seed=seed, image_size=img_dim, batch_size=batch_size)
            data_list.append(train_ds)
        if load_sets[1] == True:   
            val_ds = image_dataset_from_directory(VALIDATION_DIR, seed=seed, image_size=img_dim, batch_size=batch_size)
            data_list.append(val_ds)
        if load_sets[2] == True:
            test_ds = image_dataset_from_directory(TEST_DIR, seed=seed, image_size=img_dim, batch_size=batch_size)
            data_list.append(test_ds)
    except FileNotFoundError:
        raise FileNotFoundError(f'''image subdirectories must be called train, valid, test. The current subdirectories are:
                                {os.listdir(BASE_DIR)}''')
        
    return tuple(data_list)

# function for data augmentation that returns wanted image augmentations as a sequential model/layers