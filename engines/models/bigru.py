from abc import ABC
import tensorflow as tf
from config import classifier_config


class BiGRU(tf.keras.Model, ABC):
    """
    Bi-GRU模型
    """

    def __init__(self, num_classes, hidden_dim, embedding_dim, vocab_size, embeddings_matrix=None):
        super(BiGRU, self).__init__()
        if classifier_config['embedding_method'] is '':
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embeddings_matrix],
                                                       trainable=False)
        self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hidden_dim, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        if classifier_config['use_attention']:
            self.attention_w = tf.Variable(tf.zeros([1, 2 * hidden_dim]))
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)
        bigru_outputs = self.bigru(inputs)
        if classifier_config['use_attention']:
            output = tf.nn.tanh(bigru_outputs)
            output = tf.matmul(output, self.attention_w, transpose_b=True)
            alpha = tf.nn.softmax(output, axis=1)
            outputs = alpha * bigru_outputs
            bigru_outputs = tf.nn.tanh(outputs)
        dropout_outputs = self.dropout(bigru_outputs, training)
        outputs = tf.reduce_sum(dropout_outputs, axis=1)
        outputs = self.dense(outputs)
        return outputs
