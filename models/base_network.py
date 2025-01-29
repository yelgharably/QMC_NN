import tensorflow as tf
from ..config import CONFIG

class BaseNetwork(tf.keras.Model):
    """
    Base class for defining the neural network arch.
    """

    def __init__(self,config=CONFIG):
        super(BaseNetwork, self).__init__()
        self.config = config

        feature_conf = config['feature_extraction']
        self.hidden_layers = []
        for units in feature_conf['layers']:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    kernel_initializer='he_normal'
                )
            )
            if feature_conf.get('batch_norm', False):
                self.hidden_layers.append(tf.keras.layers.BatchNormalization())
            if feature_conf.get('dropout', 0) > 0:
                self.hidden_layers.append(tf.keras.layers.Dropout(feature_conf['dropout']))

        output_conf = config['output']
        if output_conf['real_imag']:
            self.real_output = tf.keras.layers.Dense(1, activation=output_conf['activation'])
            self.imag_output = tf.keras.layers.Dense(1, activation=output_conf['activation'])
        else:
            self.output_layer = tf.keras.layers.Dense(1, activation=output_conf['activation'])

    def call(self, inputs, training=False):
        """"
        Defines the forward pass of the network
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        if self.config['output']['real_imag']:
            return self.real_output(x), self.imag_output(x)
        else:
            return self.output_layer(x)