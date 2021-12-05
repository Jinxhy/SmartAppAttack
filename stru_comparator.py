import os
import tensorflow as tf
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz


class Model:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers


class Layer:
    def __init__(self, name, index, shape, dtype):
        self.detail = name + "," + index + "," + shape + "," + dtype


def load_repository():
    db = []

    for model_name in os.listdir("DL_models/TF_Hub/"):

        try:
            # Load TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter("DL_models/TF_Hub/" + model_name)
            interpreter.allocate_tensors()
        except:
            print(model_name, "loading error")
            continue

        details = interpreter.get_tensor_details()

        layers = []
        model = Model(model_name, layers)

        for detail in details:
            shape = re.sub(' +', ',', str(detail['shape']))[:1] + re.sub(' +', ',', str(detail['shape']))[2:]
            layer = Layer(detail['name'], str(detail['index']), shape,
                          re.sub(' +', ',', str(detail['dtype'])))

            layers.append(layer.detail)

        model.layers = layers
        db.append(model)

    return db


def load_target_models():
    targets = []

    for model_name in os.listdir("DL_models/TFLite/"):
        # Load TFLite model and allocate tensors.
        try:
            interpreter = tf.lite.Interpreter("DL_models/TFLite/" + model_name)
            interpreter.allocate_tensors()
        except:
            print(model_name, "loading error")
            continue

        details = interpreter.get_tensor_details()

        layers = []
        model = Model(model_name, layers)

        for detail in details:
            shape = re.sub(' +', ',', str(detail['shape']))[:1] + re.sub(' +', ',', str(detail['shape']))[2:]
            layer = Layer(detail['name'], str(detail['index']), shape,
                          re.sub(' +', ',', str(detail['dtype'])))

            layers.append(layer.detail)

        model.layers = layers
        targets.append(model)

    return targets


if __name__ == "__main__":
    tf_hub_db = load_repository()
    targets = load_target_models()

    # compute the structural similarity
    for target in targets:
        results = {}

        for model in tqdm(tf_hub_db):
            results[model.name] = fuzz.ratio(",".join(target.layers), ",".join(model.layers))

        result = {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}

        if result[list(result.keys())[0]] > 0:
            if result[list(result.keys())[0]] > 80:
                print(target.name)
                print(list(result.keys())[0], ":", result[list(result.keys())[0]], "%")
                print(list(result.keys())[1], ":", result[list(result.keys())[1]], "%")
                print(list(result.keys())[2], ":", result[list(result.keys())[2]], "%")
                print(list(result.keys())[3], ":", result[list(result.keys())[3]], "%\n")
