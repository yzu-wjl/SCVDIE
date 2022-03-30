import os
import pickle
import csv
from keras.preprocessing.text import Tokenizer
from config import Glove_config


class Gen_Index:
    def __init__(self):
        self.tokenizer_path = Glove_config['model_dir'] + os.sep + Glove_config['tokenizer_name']
        self.train_data = Glove_config['train_data']

    def gen_data_list(self):
        data_list = []
        f = csv.reader(open(self.train_data))
        for lst in f:
            data_list.append(lst[1])
        return data_list

    def gen_label_list(self):
        label_list = []
        f = csv.reader(open(self.train_data))
        for lst in f:
            label_list.append(lst[0])
        return label_list

    def Tokenization(self):
        tokenizer = Tokenizer(num_words=None, filters=',', lower=False, char_level=False, oov_token=None)
        data_list = self.gen_data_list()
        tokenizer.fit_on_texts(data_list)
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle)

    def LoadPickleData(self):
        with open(self.tokenizer_path, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data

    def get_word_index(self):
        tokenizer = self.LoadPickleData()
        data_list = self.gen_data_list()
        total_sequences = tokenizer.texts_to_sequences(data_list)
        word_index = tokenizer.word_index

        return [total_sequences,word_index]
