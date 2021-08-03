'''
CNN model functions
'''

import tensorflow as tf

# ************************************************************************************** 
#                                   FUNCTIONS:
# **************************************************************************************

# Add an option for data augmentation
def mobilenet_v2(img_dim=(224,224,3), num_classes=275, dropout=0.4, lr=0.01, metrics=['accuracy']):

    # scale pixel values to [-1,1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(input_shape=img_dim, include_top=False, weights='imagenet')
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    # Full model pipeline
    #################################################
    inputs = tf.keras.Input(shape=img_dim)
    #pipe = data_augmentation(inputs)
    pipe = preprocess_input(inputs)
    pipe = base_model(pipe, training=False)
    pipe = global_average_layer(pipe)
    pipe = tf.keras.layers.Dropout(dropout)(pipe)
    outputs = prediction_layer(pipe)
    model = tf.keras.Model(inputs,outputs)
    #################################################

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=metrics)

    return base_model, model



def mobilenet_v2_finetune(base_model, model, tune_all=False, layers_to_tune=50, lr=0.000001, metrics=['accuracy']):

    base_model.trainable = True

    if tune_all == False:
        # Freeze all the layers before the `layers_to_tune` layer
        for layer in base_model.layers[:layers_to_tune]:
            layer.trainable =  False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.losses.SparseCategoricalCrossentropy(),
                    metrics=metrics)

    return base_model, model


