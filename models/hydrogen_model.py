import tensorflow as tf
from ..config import CONFIG
from .base_network import BaseNetwork

class HydrogenModel(BaseNetwork):
    """
    Specialized network for hydrogen-like wavefunctions in 2D (restricted to xy-plane).
    """

    def __init__(self, config=CONFIG):
        super(HydrogenModel, self).__init__(config)

        self.r_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu')
        ])

        self.angular_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu')
        ])

        quantum_conf = config['input_processing']['quantum']
        self.n_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(quantum_conf['nlm_layer'], activation=quantum_conf['activation'])
        ])

        self.final_output = tf.keras.layers.Dense(1, activation=config['output']['activation'])

    def call(self, inputs, training=False):
        """
        Forward pass for the hydrogen model restricted to 2D.
        """
        psi_real = tf.reshape(inputs[:, 0], (-1, 1))
        psi_imag = tf.reshape(inputs[:, 1], (-1, 1))
        r = tf.reshape(inputs[:, 2], (-1, 1))       # Radial coordinate
        theta = tf.reshape(inputs[:, 3], (-1, 1))  # Theta
        n = tf.reshape(inputs[:, 4], (-1, 1))      # Quantum number n

        # Pass inputs through layers
        r_out = self.r_layer(r, training=training)
        ang_out = self.angular_layer(theta, training=training)  # Only θ is processed here
        n_out = self.n_layer(n, training=training)

        combined_real = tf.concat([r_out, ang_out, n_out], axis=-1)
        combined_imag = tf.concat([r_out, ang_out, n_out], axis=-1)

        for layer in self.hidden_layers:
            combined_real = layer(combined_real, training=training)
            combined_imag = layer(combined_imag, training=training)

        psi_pred_real = self.final_output(combined_real)
        psi_pred_imag = self.final_output(combined_imag)

        return tf.concat([
            psi_pred_real,
            psi_pred_imag,
            r, theta, n
        ], axis=1)
