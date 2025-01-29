"""
Config file for QMC Neural Network
"""

CONFIG = {
    'paths': {
        'load': 'model_1.weights.h5',
        'save': 'model_1.weights.h5',
        'reset': False
    },
    'output': {
        'real_imag': False,  # True for separate real/imaginary
        'activation': 'linear'
    },
    'physics': {
        'r_infinity': 10.0,
        'delta_r': 0.1,
        'quantum_numbers': {
            'n': 1,
            'l': 0,
            'm': 0
        }
    },
    'training': {
        'grid_size' : 5,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'epochs': 50,
        'sample_size': 50,
        'optimizer': {
            'name': 'adam',
            'clipnorm': 1.0,
            },
        'weights': {
            'sch' : 1.0,
            'nor' : 1.0,
            'bound' : 0.1,
            'ang' : 0.1,
            'eng' : 1.0
        }
    }
}