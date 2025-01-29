"""
CURRENT ISSUES:
1. Need to restructure comlpex64 into float64 because
Dense layers don't take complex 64 apparently.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
import numpy as np
import time as time

# from numba import jit, cuda

from ..config import CONFIG
from ..qmc.qmc_generator import QMC_gen
from ..models.hydrogen_model import HydrogenModel
from ..training.losses import Losses

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

n_val = CONFIG['physics']['quantum_numbers']['n']
l_val = CONFIG['physics']['quantum_numbers']['l']
m_val = CONFIG['physics']['quantum_numbers']['m']

epoch_val = CONFIG['training']['epochs']
training_rate = CONFIG['training']['learning_rate']
sample_n = CONFIG['training']['sample_size']
optimizer = CONFIG['training']['optimizer']['name']
clipnorm = CONFIG['training']['optimizer']['clipnorm']
batch_n = CONFIG['training']['batch_size']


class EpochProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}:")
        for key, value in logs.items():
            print(f"  {key}: {value:.4f}")

# @jit(target_backend='cuda')
def train_model(
        load_model_path=CONFIG['paths']['load'],
        save_model_path=CONFIG['paths']['save'],
        reset_model=CONFIG['paths']['reset'],
        sample_size=sample_n,
        epochs=epoch_val,
        batch_size=batch_n,
):
    starting_time = time.time()  
    print(f"[INFO] Generating QMC data with sample_size={sample_size}")
    
    generator = QMC_gen(n_samples=sample_size,burn_in=5000,a0=1.0)

    print("[DEBUG] QMC data generation started...")
    X_norm = generator.gen_hybrid_samples()
    psi_value = tf.convert_to_tensor(generator.trial_wfc(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], n_val, l_val, m_val), dtype=tf.complex64)

    y = {
    'psi_real': tf.cast(tf.math.real(psi_value), dtype=tf.float64),
    'psi_imag': tf.cast(tf.math.imag(psi_value), dtype=tf.float64),
    'r': tf.convert_to_tensor(X_norm[:, 0].reshape(-1, 1), dtype=tf.float64),
    'theta': tf.convert_to_tensor(X_norm[:, 1].reshape(-1, 1), dtype=tf.float64),
    'phi': tf.convert_to_tensor(X_norm[:, 2].reshape(-1, 1), dtype=tf.float64),
    'n': tf.convert_to_tensor(n_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64),
    'l': tf.convert_to_tensor(l_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64),
    'm': tf.convert_to_tensor(m_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64)
    }

    y['psi_real'] = tf.reshape(y['psi_real'], (-1, 1))
    y['psi_imag'] = tf.reshape(y['psi_imag'], (-1, 1))
    print("Tensor shapes:")
    print("psi_real shape:", y['psi_real'].shape)
    print("psi_imag shape:", y['psi_imag'].shape)
    print("r shape:", y['r'].shape)
    print("theta shape:", y['theta'].shape)
    print("phi shape:", y['phi'].shape)
    print("n shape:", y['n'].shape)
    print("l shape:", y['l'].shape)
    print("m shape:", y['m'].shape)
        
    X_unshaped = {
    'psi_real': tf.zeros_like(y['psi_real'], dtype=tf.float64),
    'psi_imag': tf.zeros_like(y['psi_imag'], dtype=tf.float64),
    'r': y['r'],
    'theta': y['theta'],
    'phi': y['phi'],
    'n': tf.convert_to_tensor(n_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64),
    'l': tf.convert_to_tensor(l_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64),
    'm': tf.convert_to_tensor(m_val * np.ones((len(X_norm[:, 0]), 1)), dtype=tf.float64)
    }

    X = tf.concat([
        tf.cast(X_unshaped['psi_real'], tf.float64),
        tf.cast(X_unshaped['psi_imag'], tf.float64),
        tf.cast(X_unshaped['r'], tf.float64),
        tf.cast(X_unshaped['theta'], tf.float64),
        tf.cast(X_unshaped['phi'], tf.float64),
        tf.cast(X_unshaped['n'], tf.float64),
        tf.cast(X_unshaped['l'], tf.float64),
        tf.cast(X_unshaped['m'], tf.float64)
    ], axis=1)
    

    y_true = tf.concat([
        y['psi_real'], 
        y['psi_imag'], 
        y['r'], 
        y['theta'], 
        y['phi'], 
        y['n'], 
        y['l'], 
        y['m']
    ], axis=1)  # Concatenate along the second dimension

    print("[DEBUG] Done with tf.convert_to_tensor.")
    print("[DEBUG] QMC data generation finished.")

    with tf.device('/GPU:0'):
        model = HydrogenModel(config=CONFIG)

        model.build(input_shape=(X.shape))

        my_losses = Losses(delta_r=CONFIG['physics']['delta_r'],config=CONFIG)
        loss_fn = my_losses.make_loss_fn(my_losses)

        # if load_model_path and os.path.exists(load_model_path) and not reset_model:
        #     print(f"[INFO] Loading model from {load_model_path}")
        #     model.load_weights(load_model_path)
        # elif load_model_path and not os.path.exists(load_model_path):
        #     print(f"[WARNING] Model path {load_model_path} does not exist. Training from scratch.")
        # else:
        #     print("[INFO] Training model from scratch.")

        
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_rate,
                                            clipnorm=clipnorm)
        model.compile(optimizer=optimizer,loss=loss_fn)
        progressor = EpochProgressCallback()

        print("[DEBUG] Starting model.fit...")
        model.fit(X,y_true,batch_size=batch_size,epochs=epochs,callbacks=progressor)
        print("[DEBUG] model.fit completed.")

        if save_model_path:
            print(f"[INFO] Saving model to {save_model_path}")
            model.save_weights(save_model_path)

    print(f"Total time taken: {time.time() - starting_time} seconds")

if __name__ == "__main__":
    train_model()