import pandas as pd
import numpy as np
import os

'''
This file is for transforming several raw multi-emotion-label data .csv files (iemocap, dailydialogue, goemtotion) into 
easy to use pandas data dictionaries
'''


# load data into pandas dict
def load_data(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, name), delimiter="\n")
    return df


# function for processing daily dialogue texts
def extend(text, labels):
    result = []
    seqs = text.split('__eou__')
    seqs = seqs[:-1]

    label_seq = labels.split(' ')
    label_seq = label_seq[:-1]
    for line, emo in zip(seqs, label_seq):
        # Get rid of the blanks at the start & end of each turns
        if line[0] == ' ':
            line = line[1:]
        if line[-1] == ' ':
            line = line[:-1]
        result.append([line, emo])
    return result


# function for flattening daily dialogue texts
def flatten(result_list):
    res = []
    for el in result_list:
        for list_el in el:
            res.append(list_el)
    return res


# assign emotion labels to category numbers
def determine_emotion(label):
    if label == "0":
        return "no emotion"
    elif label == "1":
        return "anger"
    elif label == "2":
        return "disgust"
    elif label == "3":
        return "fear"
    elif label == "4":
        return "happiness"
    elif label == "5":
        return "sadness"
    elif label == "6":
        return "surprise"
    else:
        return "no known label"


# parse the daily dialogue dataset
def parse_dailydialogue():
    # { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
    df_text = load_data("datasets/dailydialogue/dialogues_text.txt")
    df_emotion = load_data("datasets/dailydialogue/dialogues_emotion.txt")
    df_text['emotion'] = df_emotion['emotion']
    result = [extend(x, y) for x, y in zip(df_text['text'], df_text['emotion'])]
    # flattens the result and treats entries in list as rows in df
    df_flat_result = pd.DataFrame([el for sublist in result for el in sublist])
    df_flat_result.columns = ['text', 'label']
    df_flat_result["emotion"] = df_flat_result["label"].apply(lambda x: determine_emotion(x))
    # df_flat_result = df_flat_result[df_flat_result["emotion"] != "disgust"]
    # df_flat_result = df_flat_result[df_flat_result["emotion"] != "fear"]
    print(df_flat_result.head())
    print("daily dialogue parsed:\n", df_flat_result["emotion"].value_counts())
    # df_sample = df_flat_result.groupby("emotion").sample(n=1000, random_state=1)
    # print(df_sample["emotion"].value_counts())
    df_flat_result = df_flat_result.drop(columns=["label"])
    df_flat_result.to_csv('datasets/parsed/DailyDialogue_parsed.csv')
    return df_flat_result


# parse the iemocap dataset
def parse_iemocap():
    # {angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted, other}
    transcriptions = []
    emotions = []
    sessions = ["ses01", "ses02", "ses03", "ses04", "ses05"]

    # there are 5 sessions of dialogues
    for session in sessions:
        # open all transcriptions of sessions, i.e. text
        for filename in os.listdir(os.getcwd()+"/datasets/iemocap/transcriptions/"+session):
            with open(os.path.join(os.getcwd()+"/datasets/iemocap/transcriptions/"+session, filename), 'r') as trans:
                transcription_content = trans.readlines()
                for line in transcription_content:
                    entry = line.split(":")
                    if "[" in entry[0]:
                        entry[0] = entry[0][:entry[0].index("[") - 1]
                        entry[1] = entry[1][1:].strip("\n")
                        transcriptions.append(entry)
        # open all emotion labels
        for filename in os.listdir(os.getcwd()+"/datasets/iemocap/emoEvaluation/"+session):
            with open(os.path.join(os.getcwd()+"/datasets/iemocap/emoEvaluation/"+session, filename), 'r') as f:
                content = f.readlines()
                for line in content:
                    if "Ses" in line:
                        temp = line[line.index("Ses"):]
                        entry = temp[:temp.index("[")-1].split("\t")
                        emotions.append(entry)

    df = pd.DataFrame(transcriptions, columns=["session_id", "text"])
    df_emotion = pd.DataFrame(emotions, columns=["session_id", "emotion"])

    df_merged = df.merge(df_emotion, on="session_id")
    print(df_merged.head())
    print("iemocap parsed:\n", df_merged["emotion"].value_counts())
    di = {"fru": "frustration", "neu": "neutral", "ang": "anger", "sad": "sadness", "exc": "excited", "hap": "happiness",
          "sur": "surprise", "fea": "fear", "oth": "other", "dis": "disgust"}
    df_merged.replace({"emotion": di}, inplace=True)
    print("iemocap parsed:\n", df_merged["emotion"].value_counts())
    df_merged = df_merged.drop(columns=["session_id"])
    df_merged.to_csv('datasets/parsed/iemocap_parsed.csv')
    return df_merged


def parse_goemotion():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fpath1 = os.path.join(dir_path, 'datasets/goemotions/goemotions_train.csv')
    df_train = pd.read_csv(fpath1, sep='\t')
    df_train.columns = ["text", "emotion", "rnd"]
    fpath2 = os.path.join(dir_path, 'datasets/goemotions/goemotions_test.csv')
    df_test = pd.read_csv(fpath2, sep='\t')
    df_test.columns = ["text", "emotion", "rnd"]
    fpath3 = os.path.join(dir_path, 'datasets/goemotions/goemotions_dev.csv')
    df_dev = pd.read_csv(fpath3, sep='\t')
    df_dev.columns = ["text", "emotion", "rnd"]

    df_train = df_train.append(df_test)
    df_train = df_train.append(df_dev)

    df_train.drop(columns=["rnd"], inplace=True)
    df_train = df_train[df_train['emotion'].map(lambda x: len(x.split(",")) == 1)]

    emotions_dict = pd.read_csv(os.path.join(dir_path, 'datasets/goemotions/emotions_mapping.csv')).to_dict()["emotion"]
    # see https://www.aclweb.org/anthology/2020.acl-main.372.pdf for explanation of the mapping
    ekman_level_dict = {
        "anger": ["anger", "annoyance", "disapproval"],
        "disgust": ["disgust"],
        "fear": ["fear", "nervousness"],
        "happiness": ["admiration", "amusement", "approval", "caring", "desire", "gratitude", "joy", "love", "optimism",
                      "pride", "relief"],
        "excited": ["excitement"],
        "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
        "surprise": ["confusion", "curiosity", "realization", "surprise"],
        "neutral": ["neutral"]
    }

    # y = list(ekman_level_dict.keys())[list(ekman_level_dict.values()).index("neutral")]

    df_train['emotion'] = df_train['emotion'].apply(lambda x: emotions_dict[int(x)])
    df_train['emotion'] = df_train['emotion'].apply(
        lambda x: [emotion for emotion, emotion_list in ekman_level_dict.items() if x in emotion_list][0])
    print("df_goemotion value counts", df_train["emotion"].value_counts())
    print("df_goemotion head", df_train.head())
    return df_train


if __name__ == '__main__':
    df_dailydialogue = parse_dailydialogue()
    df_iemocap = parse_iemocap()
    df_goemotion = parse_goemotion()

    frames = [df_dailydialogue, df_iemocap, df_goemotion]
    result = pd.concat(frames)
    print(result["emotion"].value_counts())

    # discard labels with unfitting/ too few samples
    result = result[result['emotion'].map(
        lambda x: (x != "xxx" and x != "other" and x != "no emotion" and x != "disgust" and x != "fear")
    )]
    print(result["emotion"].value_counts())
    # df_sample = result.groupby("emotion").sample(n=5000, replace=True, random_state=1)
    result = result.reset_index(drop=True)
    # downsampling of happiness and neutral samples because there are too many
    result = result.drop(result[result['emotion'] == "happiness"].sample(frac=.6, random_state=42).index)
    result = result.drop(result[result['emotion'] == "neutral"].sample(frac=.5, random_state=42).index)
    print(result["emotion"].value_counts())
    result.to_csv('datasets/parsed/iemo_daily_goemotion.csv', index=False)
