import pickle
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from engines.utils.word2vec import Word2VecUtils
from glove import Glove
from engines.utils.Glove import GloveUtils
from gensim.models.fasttext import FastText
from engines.utils.FastText import FastTextUtils
from engines.utils.clean_data import filter_word, filter_char
from engines.utils.gen_index import Gen_Index
from config import classifier_config
from collections import Counter


class DataManager:

    def __init__(self, logger):
        self.logger = logger
        self.token_level = classifier_config['token_level']
        self.embedding_method = classifier_config['embedding_method']
        self.classifier = classifier_config['classifier']
        if self.classifier == 'Bert' and self.embedding_method is not '':
            raise Exception('If you use Bert fine-tuning, you do not need to set the embedding_method')
        if self.token_level == 'char' and self.embedding_method is not '':
            raise Exception('Word granularity should not use word embedding')
        self.w2v_util = Word2VecUtils(logger)
        self.ft_util = FastTextUtils(logger)
        self.stop_words = self.w2v_util.get_stop_words()
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'

        if self.classifier == 'Bert':
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(classifier_config['bert_op'])
            self.embedding_dim = 768
            self.vocab_size = len(self.tokenizer.get_vocab())
        else:
            if self.embedding_method == 'word2vec':
                self.w2v_model = Word2Vec.load(self.w2v_util.model_path)
                self.embedding_dim = self.w2v_model.vector_size
                self.vocab_size = len(self.w2v_model.wv.vocab)
                self.word2token = {self.PADDING: 0}
                vocab_list = [(k, self.w2v_model.wv[k]) for k, v in self.w2v_model.wv.vocab.items()]
                self.embeddings_matrix = np.zeros(
                    (len(self.w2v_model.wv.vocab.items()) + 1, self.w2v_model.vector_size))
                for i in tqdm(range(len(vocab_list))):
                    word = vocab_list[i][0]
                    self.word2token[word] = i + 1
                    self.embeddings_matrix[i + 1] = vocab_list[i][1]
                self.token_file = classifier_config['token_file']
                with open(self.token_file, 'w', encoding='utf-8') as outfile:
                    for word, token in self.word2token.items():
                        outfile.write(word + '\t' + str(token) + '\n')
            elif self.embedding_method == 'Glove':
                with open(self.Glove_util.model_path, 'rb') as f:
                    self.glove_model = pickle.load(f, encoding='latin-1')
                self.word2token = {self.PADDING: 0}
                self.key_list = list(self.glove_model['dictionary'].keys())
                self.vocab_size = len(self.key_list)
                self.word_vector_list = self.glove_model['word_vectors'].tolist()
                self.embeddings_index = {}
                for index, item in tqdm(enumerate(self.key_list)):
                    word = self.key_list[index]
                    vct = np.asarray(self.word_vector_list[index], dtype='float32')
                    self.embeddings_index[word] = vct
                print('Loaded %s word vectors.' % len(self.embeddings_index))
                self.gen_index = Gen_Index()
                self.gen_index.Tokenization()
                self.word_index = self.gen_index.get_word_index()[1]
                self.embeddings_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dim))
                self.token_file = classifier_config['token_file']
                for word, i in self.word_index.items():
                    self.word2token[word] = i
                    self.embedding_vector = self.embeddings_index.get(word)
                    if self.embedding_vector is not None:
                        self.embeddings_matrix[i] = self.embedding_vector
                with open(self.token_file, 'w', encoding='utf-8') as outfile:
                    for word, token in self.word2token.items():
                        outfile.write(word + '\t' + str(token) + '\n')
            elif self.embedding_method == 'FastText':
                self.ft_model = FastText.load(self.ft_util.model_path)
                self.embedding_dim = self.ft_model.vector_size
                self.vocab_size = len(self.ft_model.wv.vocab)
                self.word2token = {self.PADDING: 0}
                vocab_list = [(k, self.ft_model.wv[k]) for k, v in self.ft_model.wv.vocab.items()]
                self.embeddings_matrix = np.zeros(
                    (len(self.ft_model.wv.vocab.items()) + 1, self.ft_model.vector_size))
                for i in tqdm(range(len(vocab_list))):
                    word = vocab_list[i][0]
                    self.word2token[word] = i + 1
                    self.embeddings_matrix[i + 1] = vocab_list[i][1]
                self.token_file = classifier_config['token_file']
                with open(self.token_file, 'w', encoding='utf-8') as outfile:
                    for word, token in self.word2token.items():
                        outfile.write(word + '\t' + str(token) + '\n')
            else:
                self.embedding_dim = classifier_config['embedding_dim']
                self.token_file = classifier_config['token_file']
                if not os.path.isfile(self.token_file):
                    self.logger.info('vocab files not exist...')
                else:
                    self.token2id, self.id2token = self.load_vocab()
                    self.vocab_size = len(self.token2id)

        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']
        self.class_id = classifier_config['classes']
        self.class_list = [name for name, index in classifier_config['classes'].items()]
        self.max_label_number = len(self.class_id)
        self.logger.info('dataManager initialed...')

    def load_vocab(self, sentences=None):
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.token_file, sentences)
        word_token2id, id2word_token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
                word_token2id[word_token] = word_token_id
                id2word_token[word_token_id] = word_token
        self.vocab_size = len(word_token2id)
        return word_token2id, id2word_token

    def build_vocab(self, token_file, sentences):
        tokens = []
        if self.token_level == 'word':
            for sentence in tqdm(sentences):
                words = self.w2v_util.processing_sentence(sentence, self.stop_words)
                tokens.extend(words)
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        else:
            for sentence in tqdm(sentences):
                chars = list(sentence)
                tokens.extend(chars)
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if k != ' ' and filter_char(k)]
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        id2token[len(id2token)] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(id2token)
        with open(token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        self.vocab_size = len(token2id)
        return token2id, id2token

    def padding(self, sentence):
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence

    def prepare_w2v_data(self, sentences, labels):
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for word in sentence:
                if word in self.word2token:
                    tokens.append(self.word2token[word])
                else:
                    tokens.append(self.word2token[self.PADDING])
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def prepare_glove_data(self, sentences, labels):
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for word in sentence:
                if word in self.word2token:
                    tokens.append(self.word2token[word])
                else:
                    tokens.append(self.word2token[self.PADDING])
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def prepare_ft_data(self, sentences, labels):
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.ft_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for word in sentence:
                if word in self.word2token:
                    tokens.append(self.word2token[word])
                else:
                    tokens.append(self.word2token[self.PADDING])
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def prepare_bert_data(self, sentences, labels):
        self.logger.info('loading data...')
        tokens_list, y = [], []
        for record in tqdm(zip(sentences, labels)):
            label = tf.one_hot(record[1], depth=self.max_label_number)
            if len(record[0]) > self.max_sequence_length - 2:
                sentence = record[0][:self.max_sequence_length - 2]
                tokens = self.tokenizer.encode(sentence)
            else:
                tokens = self.tokenizer.encode(record[0])
            if len(tokens) < self.max_sequence_length:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            tokens_list.append(tokens)
            y.append(label)
        return np.array(tokens_list), np.array(y, dtype=np.float32)

    def prepare_bert_e_data(self, sentences, labels):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(classifier_config['bert_op'])
        self.logger.info('loading data...')
        tokens_list, y = [], []
        for record in tqdm(zip(sentences, labels)):
            label = tf.one_hot(record[1], depth=self.max_label_number)
            if len(record[0]) > self.max_sequence_length - 2:
                sentence = record[0][:self.max_sequence_length - 2]
                tokens = tokenizer.encode(sentence)
            else:
                tokens = tokenizer.encode(record[0])
            if len(tokens) < self.max_sequence_length:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            tokens_list.append(tokens)
            y.append(label)
        return np.array(tokens_list), np.array(y, dtype=np.float32)

    def prepare_data(self, sentences, labels):
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            if self.token_level == 'word':
                sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            else:
                sentence = list(record[0])
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for word in sentence:
                if word in self.token2id:
                    tokens.append(self.token2id[word])
                else:
                    tokens.append(self.token2id[self.UNKNOWN])
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def get_dataset(self, df, step=None):
        df = df.loc[df.label.isin(self.class_list)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        if self.classifier == 'Bert':
            X, y = self.prepare_bert_data(df['sentence'], df['label'])
        else:
            if self.embedding_method == 'word2vec':
                X, y = self.prepare_w2v_data(df['sentence'], df['label'])
            elif self.embedding_method == 'Bert':
                X, y = self.prepare_bert_e_data(df['sentence'], df['label'])
            elif self.embedding_method == 'Glove':
                X, y = self.prepare_glove_data(df['sentence'], df['label'])
            elif self.embedding_method == 'FastText':
                X, y = self.prepare_glove_data(df['sentence'], df['label'])
            else:
                if step == 'train' and not os.path.isfile(self.token_file):
                    self.token2id, self.id2token = self.load_vocab(df['sentence'])
                X, y = self.prepare_data(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def prepare_single_sentence(self, sentence):
        if self.classifier == 'Bert':
            if len(sentence) > self.max_sequence_length - 2:
                sentence = sentence[:self.max_sequence_length - 2]
                tokens = self.tokenizer.encode(sentence)
            else:
                tokens = self.tokenizer.encode(sentence)
            if len(tokens) < 150:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            return np.array([tokens])
        else:
            if self.embedding_method == 'word2vec':
                sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                sentence = self.padding(sentence)
                tokens = []
                for word in sentence:
                    if word in self.word2token:
                        tokens.append(self.word2token[word])
                    else:
                        tokens.append(self.word2token[self.PADDING])
                return np.array([tokens])
            else:
                if self.token_level == 'word':
                    sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                else:
                    sentence = list(sentence)
                sentence = self.padding(sentence)
                tokens = []
                for word in sentence:
                    if word in self.token2id:
                        tokens.append(self.token2id[word])
                    else:
                        tokens.append(self.token2id[self.UNKNOWN])
                return np.array([tokens])
