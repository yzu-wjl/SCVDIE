import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred = tf.add(y_pred, self.epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1 - y_pred, self.gamma) * y_true
        fl = ce * weight * self.alpha
        loss = tf.reduce_sum(fl, axis=1)
        return loss
