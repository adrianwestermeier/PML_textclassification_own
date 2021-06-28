from classes.classifier import Classifier
from functions.preprocess import create_tokenizer, create_train_test_set, derive_text_and_labels, preprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback
# from dotenv import dotenv_values
from datetime import datetime
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--out_root", help="Please specify the out root dir", type=str)
parser.add_argument("--number_of_epochs", help="how much epochs", default=8, type=int)
parser.add_argument("--batch_size", help="batch size", default=64, type=int)
parser.add_argument("--architecture", help="please specify the architecture of the classifier", default="LSTM", type=str)
parser.add_argument("--project", type=str)
parser.add_argument("--entity", type=str)

args = parser.parse_args()
OUT_ROOT = args.out_root
EPOCHS = args.number_of_epochs
BATCH_SIZE = args.batch_size
ARCHITECTURE = args.architecture
PROJECT = args.project
ENTITY = args.entity

# emotion mapping: { 0: happiness, 1: sadness, 2: anger, 3: surprise, 4: frustration, 5: neutral, 6: excited}
def determine_label(line):
    if line == "happiness":
        return 0
    elif line == "sadness":
        return 1
    elif line == "anger":
        return 2
    elif line == "surprise":
        return 3
    elif line == "frustration":
        return 4
    elif line == "neutral":
        return 5
    else:  # "excited"
        return 6


# function for determining the predicted label by argmax
def label_to_category(max_index):
    mapping = {0: "happiness", 1: "sadness", 2: "anger", 3: "surprise", 4: "frustration", 5: "neutral", 6: "excited"}
    return mapping.get(max_index)


# get the predicted labels for all test samples by the tensor output
def get_classes(samples, predictions):
    result = list()
    for sample, pred in zip(samples, predictions):
        max_index = np.argmax(np.asarray(pred))
        category = label_to_category(max_index)
        result.append([sample, category])
    return result


# convert the model to tensorflow lite for usage on mobile phone
def convert_model(model_name, dir):
    model = tf.keras.models.load_model(dir + '/' + model_name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(dir + "/converted_model.tflite", "wb").write(tflite_model)
    print("converted model")


if __name__ == '__main__':
    # get environment variables if using wandb
    #env_variables = dotenv_values(".env")  # config = {"api_key": "", "entity": "", "project": ""}
    #wandb.login(key=env_variables.get("api_key"))

    # Set an experiment name to group training and evaluation in wandb
    experiment_name = "lstm_simple"

    # see available architectures in classifier
    my_config = {
            "architecture": ARCHITECTURE,
            "dropout": 0.2,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy",
            "metric": "accuracy",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found, using CPU')
    else:
        print('Found GPU at: {}'.format(device_name))

    project = ""
    entity = ""
    if PROJECT and ENTITY:
        project = PROJECT
        entity = ENTITY
    else:
        project = "none"
        entity = "none"
        #project = env_variables.get("project")
        #entity = env_variables.get("entity")
    # Start a run, tracking hyperparameters with wandb

    run = wandb.init(
        project=project,
        entity=entity,
        group=experiment_name,
        config=my_config)
    config = wandb.config

    timestamp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    directory = ""
    if not OUT_ROOT:
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(directory, 'models/' + timestamp)
    else:
        directory = OUT_ROOT
    if not os.path.exists(directory):
        os.makedirs(directory)

    # load data if you already preprocessed
    # X = np.load('functions/processed_data/x.npy')
    # Y = np.load('functions/processed_data/y.npy')
    # test_X = np.load('functions/processed_data/test_x.npy')
    # test_Y = np.load('functions/processed_data/test_y.npy')

    # preprocess the training and test set, get a tokenizer
    df, tokenizer, maxlen, dataset_tuple = preprocess(directory)
    X, test_X, Y, test_Y = dataset_tuple

    print(X.shape)
    print(Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    print('maxlen: ', maxlen)

    # define the text classifier
    classifier = Classifier(config=config, number_of_classes=7, maxlen=maxlen, X=X)

    load = False
    if load:  # load existing model
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(directory, 'models/current_best')
        name = 'model_LSTM_acc_0.53344070911407472021-05-25-10:10:53.h5'
        classifier.load_model(directory + '/' + name)
    else:  # train your own model
        parameters = {
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'callbacks': [WandbCallback()],
            'val_data': (test_X, test_Y)
        }

        # train the classifier, TODO: early stopping
        classifier.fit(X, Y, parameters)
        loss, accuracy = classifier.evaluate(test_X, test_Y)

        # save the model
        model_name = 'model_' + config.architecture + '_acc_' + str(accuracy) + '.h5'
        classifier.save_model(directory + '/' + model_name)
        # dump the config for the model
        with open(directory + '/config.json', 'w') as fp:
            json.dump(my_config, fp)
        # convert for mobile usage
        convert_model(model_name, directory)
        print("Loss of {}".format(loss), "Accuracy of {} %".format(accuracy * 100))

    # test the model on a few test sentences
    custom_test_sentences = [
        "i am so full of energy. really feeling great",
        "i am so sad all the time. it's such a pitty",
        "oh wow that sounds super exciting. let's do it",
        "you are such an annoying person. i hate talking to you",
        "i've never done that before, what an opportunity. i am looking forward trying it out",
        "what? I can't believe that!",
        "i used this tool several times and it just did not work. I don't know what to do anymore",
        "yesterday i saw my neighbor and talked to him"
    ]
    custom_test_sequences = tokenizer.texts_to_sequences(custom_test_sentences)
    custom_test_padded = pad_sequences(custom_test_sequences, maxlen=maxlen)
    predictions = classifier.predict(custom_test_padded)
    summary = get_classes(custom_test_sentences, predictions)
    print(predictions)
    print(summary)
