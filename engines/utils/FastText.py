import os
import pandas as pd
from gensim.models import FastText
from config import FastText_config


class FastTextUtils:

    def __init__(self, logger):
        self.logger = logger
        self.stop_words = FastText_config['stop_words']
        self.size = FastText_config['size']
        self.window = FastText_config['window']
        self.min_count = FastText_config['min_count']
        self.n_workers = FastText_config['n_workers']
        self.seed = FastText_config['seed']
        self.epoch = FastText_config['epoch']
        model_dir = FastText_config['model_dir']
        model_name = FastText_config['model_name']
        self.model_path = os.path.join(model_dir, model_name)
        self.train_data = FastText_config['train_data']
        self.sg = 1 if FastText_config['algorithm'] == 'skip-gram' else 0

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = str(x).strip().split()
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    def get_stop_words(self):
        stop_words_list = []
        try:
            with open(self.stop_words, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list

    def TrainFastText(self):

        self.logger.info("----------------------------------------")
        self.logger.info("Start training the FastText model. Please wait.. ")
        train_df = pd.read_csv(self.train_data, encoding='utf-8')
        stop_words = self.get_stop_words()
        self.logger.info('Processing sentence...')
        train_df['sentence'] = train_df.sentence.apply(self.processing_sentence, args=(stop_words,))
        train_df.dropna(inplace=True)
        all_cut_sentence = train_df.sentence.to_list()
        ft_Model = FastText(workers=self.n_workers, size=self.size, window=self.window,
                            min_count=self.min_count, sg=self.sg, seed=self.seed)
        ft_Model.build_vocab(all_cut_sentence)
        ft_Model.train(all_cut_sentence, total_examples=ft_Model.corpus_count, epochs=self.epoch)
        ft_Model.save(self.model_path)
        self.logger.info("Model training completed!")
        self.logger.info("----------------------------------------")
        self.logger.info("The trained FastText model: ")
        self.logger.info(ft_Model)
