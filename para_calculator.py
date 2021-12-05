import os
import tensorflow as tf
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np
import itertools


class Model:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers


class Layer:
    def __init__(self, weight_bias):
        self.detail = weight_bias


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
            layer_details = interpreter.get_tensor(detail['index'])

            if "weights" in detail['name']:
                weight = Layer(layer_details)
                layers.append(weight.detail)

            if "bias" in detail['name']:
                bias = Layer(layer_details)
                layers.append(bias.detail)

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
            layer_details = interpreter.get_tensor(detail['index'])

            if "weights" in detail['name']:
                weight = Layer(layer_details)
                layers.append(weight.detail)

            if "bias" in detail['name']:
                bias = Layer(layer_details)
                layers.append(bias.detail)

        model.layers = layers
        targets.append(model)

    return targets


if __name__ == "__main__":
    tf_hub_db = load_repository()
    targets = load_target_models()

    # compute the parametric similarity
    for target in targets:
        print(target.name)
        for model in tf_hub_db:
            results = []
            for x, y in zip(target.layers, model.layers):
                try:
                    results.append(np.allclose(x, y, equal_nan=True))
                except:
                    print("shape incomparable")

            max_continue = [sum(1 for _ in group) for key, group in itertools.groupby(results) if key]
            if max_continue:
                similarity = max_continue[0] / len(results)
                print(target.name, model.name, "{:.2%}".format(similarity))
