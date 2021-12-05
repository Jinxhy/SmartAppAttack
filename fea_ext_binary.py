import matplotlib.pyplot as plt
from tflite_converter import lite_converter
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

# training data is based on the targeted model (CIFAR-10, GTSRB, Oxford Flowers or the dataset of a particular on-device model)
PATH = '../datasets/GTSRB'

train_dir = os.path.join(PATH, 'train_stop_sim')
test_dir = os.path.join(PATH, 'test_stop_sim')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)

print("Train:", train_dataset.class_names)
print("Test:", test_dataset.class_names)

test_batches = tf.data.experimental.cardinality(test_dataset)
test_dataset = test_dataset.take(test_batches // 2)
valid_dataset = test_dataset.skip(test_batches // 2)

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Use data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Rescale pixel values
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2, InceptionV3, ResNet50V2 or an on-device model's pre-trained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

# Freeze the convolutional base
base_model.trainable = False
base_model.summary()

# Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

# apply 2 here for the Enhance Binary Adversarial Model Attack (E-BAMA)
prediction_layer = tf.keras.layers.Dense(2)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
initial_epochs = 10

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# Evaluation and prediction
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# save the model
model.save('fea_ext_bin_models/MobileNetV2_GTSRB_stop_sim')

# convert .pb model to .tflite model
save_path = 'tflite_models/fea_ext_bin_models/MobileNetV2_GTSRB_stop_sim/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

lite_converter('fea_ext_bin_models/MobileNetV2_GTSRB_stop_sim', save_path + 'MobileNetV2_GTSRB_stop_sim.tflite')
