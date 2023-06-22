import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou


test_set_name = input('Enter the testing dataset: ')


""" Global parameters """
H = 384
W = 512


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

def save_results(image, mask, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 128
    
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    model_path_list = glob('files/*.h5')
    model_list = []
    for item in model_path_list:
        model_list.append(item.split('files\\')[1].split('.h5')[0].split('model_')[1])

    for model_suffix in model_list:
        """ Directory for storing files """
        result_path = "results_"+model_suffix+'_'+test_set_name
        create_dir(result_path)

        """ Loading model """
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model("files/model_"+model_suffix+".h5", compile=False)

        """ Load the dataset """
        valid_path = os.path.join(test_set_name, "test")
        test_x, test_y = load_data(valid_path)
        #print(f"Test: {len(test_x)} - {len(test_y)}")

        """ Evaluation and Prediction """
        print('\nmodel_'+model_suffix+' results: ')
        SCORE = []
        for x, y in tqdm(zip(test_x, test_y), total=len(test_x), disable=True):

            """ Extract the name """
            name = x.split("\\")[-1].split(".")[0]

            """ Reading the image """
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            x = image/255.0
            x = np.expand_dims(x, axis=0)

            """ Reading the mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.uint8)

            """ Prediction """
            y_pred = model.predict(x)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)

            """ Saving the prediction """
            save_image_path = result_path + f"/{name}.png"
            save_results(image, mask, y_pred, save_image_path)

            """ Flatten the array """
            mask = mask.flatten()
            y_pred = y_pred.flatten()

            """ Calculating the metrics values """
            acc_value = accuracy_score(mask, y_pred)
            f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
            precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
            SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

        """ Metrics values """
        score = [s[1:]for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")

        df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
        df.to_csv("files/score_"+model_suffix+'_'+test_set_name+".csv")
