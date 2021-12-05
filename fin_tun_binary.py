from tflite_converter import lite_converter
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

# The fine-tuning number is varying depending on an on-device model
fine_tune_numbers = [10, 20, 30, 40, 50, 60]
fine_tune_starts = [144, 134, 124, 114, 104, 94]

for i in range(len(fine_tune_numbers)):
    print('Train a model: MobileNetV2_stop_' + str(fine_tune_numbers[i]) + '_sim')

    PATH = '../datasets/GTSRB'
    train_dir = os.path.join(PATH, 'train_sim')
    test_dir = os.path.join(PATH, 'test_sim')
    validation_dir = os.path.join(PATH, 'valid_sim')

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

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Use data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # Rescale pixel values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Create the base model from the pre-trained model MobileNet V2, InceptionV3, ResNet50V2 or an on-device model's pre-trained model
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=True,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    # Freeze the convolutional base
    base_model.trainable = False
    base_model.summary()

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    # Apply 2 here for the Enhance Binary Adversarial Model Attack (E-BAMA)
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
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    # Fine Tuning
    base_model.trainable = True

    # The number of layers in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    fine_tune_at = fine_tune_starts[i]

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    fine_tuning_layers = 0
    for layer in base_model.layers:
        if layer.trainable:
            fine_tuning_layers += 1

    print('# fine tuning layers:', fine_tuning_layers)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                  metrics=['accuracy'])

    # Continue training the model
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    # Evaluation and prediction
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # Save the model
    model.save('fin_tun_bin_models/MobileNetV2_GTSRB_' + str(fine_tuning_layers) + '_sim')

    # Convert .pb model to .tflite model
    save_path = 'tflite_models/fin_tun_bin_models/MobileNetV2_GTSRB_' + str(fine_tuning_layers) + '_sim/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lite_converter('fin_tun_bin_models/MobileNetV2_GTSRB_' + str(fine_tuning_layers) + '_sim',
                   save_path + 'MobileNetV2_GTSRB_' + str(fine_tuning_layers) + '_sim.tflite')
