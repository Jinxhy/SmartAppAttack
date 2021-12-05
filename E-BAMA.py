import tensorflow as tf
import pathlib
import shutil
from tqdm import tqdm
import numpy as np
import foolbox as fb
from PIL import Image
import os
import eagerpy as ep


def copy_raw(file_path, index, label):
    foolbox_dir = './venv/Lib/site-packages/foolbox/data/'
    shutil.copy2(file_path, foolbox_dir + 'imagenet_' + f"{index:02d}" + '_' + str(label) + '.png')


def inference(image_path, copy):
    output_labels = dict()
    img_height = 224
    img_width = 224
    index = 0

    for file in pathlib.Path(image_path).iterdir():
        # Read and resize the image
        img = tf.keras.preprocessing.image.load_img(
            file, target_size=(img_height, img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)
        normalized_input = rescale(img_array)
        interpreter.set_tensor(input_details[0]['index'], normalized_input)

        # Run the inference
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Output prediction
        max_pro_index = list(np.where(output_data[0] == np.amax(output_data[0])))
        prediction = max_pro_index[0][0]
        output_labels[str(file).split('\\')[-1]] = prediction
        print(file, prediction)

        # Copy to the foolbox directory as input images
        if copy:
            copy_raw(file, index, prediction)
            index += 1

    return output_labels


def save_advs(model_name, attack_name, advs_list, round):
    # Rescale to 0-255 and convert to uint8, then save adversarial images
    for i, advs in enumerate(advs_list):
        for index, adv in enumerate(advs):
            adv_format = (255.0 / advs[index].numpy().max() * (advs[index].numpy() - advs[index].numpy().min())).astype(
                np.uint8)
            adv_img = Image.fromarray(adv_format)
            path = 'adv_examples/' + model_name + '/' + attack_name + '/' + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            adv_img.save(path + '/adv' + str(round + index + 100) + '.png')


def generate_advs(tflite_model, input_size):
    # Get the Enhanced Binary Adversarial Model
    model = tf.keras.models.load_model('exp_models/GTSRB/MobileNetV2_GTSRB_stop_sim')

    # Specify the correct bounds and preprocessing based on the pre-trained model
    preprocessing = dict() 
    bounds = (0, 255)
    fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

    # Transform bounds
    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)

    for i in np.arange(0, 50, 10).tolist():
        images, labels = fb.utils.samples(fmodel, index=i, dataset='imagenet', batchsize=10)

        # Check the accuracy of a model to make sure the preprocessing is correct
        print("Accuracy(before attack):", fb.utils.accuracy(fmodel, images, labels))
        print("\nImage:", type(images), images.shape)
        print("Label:", type(labels), labels)

        # Adversarial attack: FGSM, C&W and CAN
        # l2gn = fb.attacks.FGSM()
        # l2gn = fb.attacks.L2CarliniWagnerAttack(steps=10)
        l2gn = fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack()

        # Epsilons for MobileNetV2
        l2gn_epsilons = np.linspace(20, 20, num=1)
        # Epsilons for InceptionV3
        # l2gn_epsilons = np.linspace(50, 50, num=1)
        # Epsilons for ResNet50
        # l2gn_epsilons = np.linspace(50, 50, num=1)

        images = ep.astensor(images)
        labels = ep.astensor(labels)

        raw, l2gn_advs_list, success = l2gn(fmodel, images, labels, epsilons=l2gn_epsilons)
        save_advs(tflite_model, 'L2ClippingAwareAdditiveGaussianNoiseAttack', l2gn_advs_list, i)
        print('L2ClippingAwareAdditiveGaussianNoiseAttack', success.float32().mean().item())


def success_rate(raw, adv):
    ori_labels = raw
    adv_labels = adv
    sum = len(ori_labels)
    no_match = 0

    for l1, l2 in zip(ori_labels, adv_labels):

        if ori_labels[l1] != adv_labels[l2]:
            print(l1, ':', ori_labels[l1], l2, ':', adv_labels[l2])
            no_match += 1

    return no_match / sum


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    model_name = 'the targeted model name'

    print('Attacked model:', model_name)

    interpreter = tf.lite.Interpreter(model_path='exp_models/GTSRB/' + model_name + '.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get model predictions
    ori_labels = inference('exp_models/GTSRB/stop', True)

    # Record attack success rate
    success = []

    # Repeat the E-BAMA multiple times to avoid bias
    for i in tqdm(range(50)):
        # Generate adversarial examples
        generate_advs(model_name + '', len(ori_labels))

        # Get adversarial attack success rate
        attacks = 'adv_examples/' + model_name + '/'
        for attack_name in os.listdir(attacks):
            attack_dir = os.path.join(attacks, attack_name)
            print('\n', attack_name)

            for i in range(1):
                adv_labels = inference(attack_dir + '/' + str(i), False)
                att_rate = success_rate(ori_labels, adv_labels)
                print('Attack success rate:', att_rate)

                if attack_name == 'L2ClippingAwareAdditiveGaussianNoiseAttack':
                    success.append(att_rate)

        print("Attacking results:")
        print('l2gn current min:', '{0:.4f}'.format(min(success)))
        print('l2gn current max:', '{0:.4f}'.format(max(success)))
        print(success)
