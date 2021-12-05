import os
import shutil
from tqdm import tqdm

decomposed_path = './decomposed_apks'
target_path = './DL_apps/TFLite'

# TFLite model naming convention
scheme = ['.tflite', '.tfl', '.lite']

dl_app_count = 0

for cat_name in os.listdir(decomposed_path):
    cat_dir = os.path.join(decomposed_path, cat_name)

    if os.path.isdir(cat_dir):
        for apk_name in tqdm(os.listdir(cat_dir)):
            apk_dir = os.path.join(cat_dir, apk_name)
            is_dl_app = False

            # search TFLite model file
            stop_looping = False
            for root, dirs, files in os.walk(apk_dir, topdown=True):
                for file in files:
                    if file.endswith(scheme[0]) or file.endswith(scheme[1]):
                        # extract DL-based app
                        copy_path = os.path.join(target_path, cat_name, apk_name)
                        print('copying ' + apk_name + ' now...(tflite)')
                        shutil.copytree(apk_dir, copy_path)

                        is_dl_app = True
                        stop_looping = True
                        break

                if stop_looping:
                    break

            if is_dl_app:
                dl_app_count += 1

print('Extract ' + str(dl_app_count) + ' TFLite DL apps.')
