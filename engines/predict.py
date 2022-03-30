import tensorflow as tf
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from engines.utils.metrics import cal_metrics
from config import classifier_config



class Predictor:
    def __init__(self, data_manager, logger):
        hidden_dim = classifier_config['hidden_dim']
        classifier = classifier_config['classifier']
        self.dataManager = data_manager
        self.seq_length = data_manager.max_sequence_length
        num_classes = data_manager.max_label_number
        self.embedding_dim = data_manager.embedding_dim
        vocab_size = data_manager.vocab_size
        self.logger = logger
        num_filters = classifier_config['num_filters']
        self.checkpoints_dir = classifier_config['checkpoints_dir']
        logger.info('loading model parameter')

        if classifier_config['embedding_method'] == 'word2vec':
            embeddings_matrix = data_manager.embeddings_matrix
        else:
            embeddings_matrix = None

        if classifier == 'textcnn':
            from engines.models.textcnn import TextCNN
            self.model = TextCNN(self.seq_length, num_filters, num_classes, self.embedding_dim, vocab_size,
                                 embeddings_matrix)
        elif classifier == 'textrcnn':
            from engines.models.textrcnn import TextRCNN
            self.model = TextRCNN(num_classes, hidden_dim, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'textrnn':
            from engines.models.textrnn import TextRNN
            self.model = TextRNN(num_classes, hidden_dim, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'transformer':
            from engines.models.transformer import Transformer
            self.model = Transformer(self.seq_length, num_classes, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'Bert':
            from engines.models.bert import BertClassification
            self.model = BertClassification(num_classes)
        else:
            raise Exception('config model is not exist')
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
        logger.info('loading model successfully')

    def predict_test(self):
        test_file = classifier_config['test_file']
        if test_file == '':
            self.logger.info('test dataset does not exist!')
            return
        test_df = pd.read_csv(test_file).sample(frac=1)
        test_dataset = self.dataManager.get_dataset(test_df)
        batch_size = self.dataManager.batch_size
        reverse_classes = {str(class_id): class_name for class_name, class_id in self.dataManager.class_id.items()}
        y_true, y_pred = np.array([]), np.array([])
        start_time = time.time()
        for step, batch in tqdm(test_dataset.shuffle(len(test_dataset)).batch(batch_size).enumerate()):
            X_test_batch, y_test_batch = batch
            logits = self.model(X_test_batch)
            predictions = tf.argmax(logits, axis=-1)
            y_test_batch = tf.argmax(y_test_batch, axis=-1)
            y_true = np.append(y_true, y_test_batch)
            y_pred = np.append(y_pred, predictions)
        self.logger.info('test time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        measures, each_classes,_ = cal_metrics(y_true=y_true, y_pred=y_pred)
        res_str = ''
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        self.logger.info('%s' % res_str)
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in reverse_classes:
                classes_val_str += ('\n' + reverse_classes[k] + ': ' + str(each_classes[k]))
        self.logger.info(classes_val_str)

    def predict_one(self, sentence):
        reverse_classes = {class_id: class_name for class_name, class_id in self.dataManager.class_id.items()}
        start_time = time.time()
        vector = self.dataManager.prepare_single_sentence(sentence)
        logits = self.model(inputs=vector)
        prediction = tf.argmax(logits, axis=-1)
        prediction = prediction.numpy()[0]
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time)*1000))
        return reverse_classes[prediction]

    def save_model(self):
        tf.saved_model.save(self.model, self.checkpoints_dir,
                            signatures=self.model.call.get_concrete_function(
                                tf.TensorSpec([None, self.seq_length], tf.int32, name='inputs')))
        self.logger.info('The model has been saved')
