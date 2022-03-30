import csv
import os
from glove import Corpus, Glove
from config import Glove_config


class GloveUtils:
    def __init__(self, logger):
        self.logger = logger
        self.components = Glove_config['components']
        self.glove_window = Glove_config['window']
        self.glove_epoch = Glove_config['epoch']
        self.glove_learning_rate = Glove_config['learning_rate']
        self.workers = Glove_config['workers']
        model_dir = Glove_config['model_dir']
        model_name = Glove_config['model_name']
        corpus_model = Glove_config['corpus_model']
        self.model_path = os.path.join(model_dir, model_name)
        self.corpus_path = os.path.join(model_dir, corpus_model)
        self.model_dir = Glove_config['model_dir']
        self.model_name = Glove_config['model_name']
        self.train_data = Glove_config['train_data']

    def TrainGlove(self):
        f = csv.reader(open(self.train_data))
        all_texts = []
        for lst in f:
            all_texts.append(lst[1].split())
        self.logger.info('Processing sentence...')

        self.logger.info("----------------------------------------")
        self.logger.info("Start training the GLoVe model. Please wait.. ")
        corpus = Corpus()
        corpus.fit(all_texts, window=self.glove_window)
        glove = Glove(no_components=self.components, learning_rate=self.glove_learning_rate)
        glove.fit(corpus.matrix, epochs=self.glove_epoch, no_threads=self.workers, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        glove.save('Glove_model.pkl')
        corpus.save(self.corpus_path)

        vector_size = self.components
        with open(self.model_dir + 'Glove_results.txt', "w", encoding='latin-1') as f:
            for word in glove.dictionary:
                f.write(word)
                f.write(" ")
                for i in range(0, vector_size):
                    f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
                    f.write(" ")
                f.write("\n")
            f.close()

        self.logger.info("Model training completed!")
        self.logger.info("GLOVE SAVE HERE: " + self.model_dir + 'glove.model')
        self.logger.info("----------------------------------------")
