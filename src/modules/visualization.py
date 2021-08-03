'''
Image visualization functions
'''

import matplotlib.pyplot as plt

# ************************************************************************************** 
#                                   FUNCTIONS:
# **************************************************************************************

def plot_images(image_dataset, figsize=(10, 10), num_images=9):

    class_names = image_dataset.class_names

    plt.figure(figsize=figsize)
    for images, labels in image_dataset.take(1):
        for idx in range(num_images):
            ax = plt.subplot(3, 3, idx + 1)
            plt.imshow(images[idx].numpy().astype("uint8"))
            plt.title(class_names[labels[idx]])
            plt.axis("off")
    
    return None

def model_metrics(model_history, figsize=(8, 8), ylim_loss=1):

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,ylim_loss])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show() 

    return None
