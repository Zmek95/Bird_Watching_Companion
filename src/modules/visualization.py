'''
Image visualization functions
'''

import matplotlib.pyplot as plt
import numpy as np

# ************************************************************************************** 
#                                   FUNCTIONS:
# **************************************************************************************

def plot_images(image_dataset, figsize=(10, 10), num_images=9):

    class_names = image_dataset.class_names

    plt.figure(figsize=figsize)
    for images, labels in image_dataset.take(1):  # take one batch of images, 
                                                  # batch size is set when the dataset is created.
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

def plot_image_prediction(predictions_array, img, class_names, *true_label):
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)

    if true_label:
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)
    else:
        plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100*np.max(predictions_array)),
                                    color='blue')

def plot_value_array(predictions_array, class_names, *true_label):
    
    plt.grid(False)
    plt.xticks(rotation='vertical')
    plt.yticks([])
    
    
    max_idxs = np.argpartition(predictions_array, -5)[-5:]  # Get 5 largest elements
    max_idxs = max_idxs[np.argsort(predictions_array[max_idxs])][::-1] # Sort the indexes by element 

    if true_label:    
        if true_label in max_idxs: 
            true_idx = np.where(max_idxs == true_label)[0][0]
            
        else:# if true label not in top 5, replace lowest element with true label
            max_idxs[-1] = true_label
            true_idx = -1
    
    thisplot = plt.bar([class_names[i] for i in max_idxs], predictions_array[max_idxs], color="#777777")
    plt.ylim([0, 1])
    
    if true_label:
        thisplot[0].set_color('red') # predicted label
        thisplot[true_idx].set_color('blue')

def image_metrics(rows, columns, predictions, images, class_names, *labels):
    
    num_images = rows * columns
    
    ax = plt.figure(figsize=(2 * 2 * columns, 2 * rows))
    for i in range(num_images):
        plt.subplot(rows, 2 * columns, 2 * i + 1)
        if labels:
            plot_image_prediction(predictions[i], images[i].astype('uint8'), class_names, labels[i])
        else:
            plot_image_prediction(predictions[i], images[i].astype('uint8'), class_names)
        plt.subplot(rows, 2* columns, 2 * i + 2)
        if labels:
            plot_value_array(predictions[i], class_names, labels[i])
        else:
            plot_value_array(predictions[i], class_names)
    #plt.tight_layout()
    #plt.show()
    return ax