from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextDNN(tf.keras.Model, ABC):
    """
    TextDNN模型
    """

    def __init__(self, num_classes, embedding_dim, vocab_size, embeddings_matrix=None):
        super(TextDNN, self).__init__()
        if classifier_config['embedding_method'] is '':
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embeddings_matrix],
                                                       trainable=False)
        self.flatten = tf.keras.layers.Flatten()

        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(classifier_config['num_filters'],
                                           activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense'
                                           )

        self.dropout0 = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense0 = tf.keras.layers.Dense(classifier_config['num_filters'],
                                            activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense'
                                            )

        self.dense1 = tf.keras.layers.Dense(int(classifier_config['num_filters'] / 2),
                                            activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense'
                                            )
        self.dense2 = tf.keras.layers.Dense(int(classifier_config['num_filters'] / 4),
                                            activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')
        self.dense3 = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)
        flatten_outputs = self.flatten(inputs)
        dropout = self.dropout(flatten_outputs)
        dense = self.dense(dropout)
        dropout0 = self.dropout0(dense)
        dense0 = self.dense0(dropout0)
        dense1 = self.dense1(dense0)
        dense2 = self.dense2(dense1)
        outputs = self.dense3(dense2)
        return outputs
