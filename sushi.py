import tensorflow as tf
import numpy as np
import pandas as pd
from DeepFM import DeepFM
from sklearn.metrics import mean_squared_error
import pickle
import os, sys
import argparse


parser = argparse.ArgumentParser(description='Run DeepFM')
parser.add_argument('--iter', type=int, nargs='?', default=20)
parser.add_argument('--fm', type=bool, nargs='?', const=True, default=True)
parser.add_argument('--deep', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--nb_layers', type=int, nargs='?', default=2)
parser.add_argument('--nb_neurons', type=int, nargs='?', default=50)
options = parser.parse_args()

with open('sushi.pickle', 'rb') as f:
    Xi = pickle.load(f)
    Xv = pickle.load(f)
    y = pickle.load(f)

print(len(Xi), len(Xv), len(y))
print(len(Xi[0]), len(Xv[0]))

nb_fields = len(Xi[0])
print(Xi[0])
# interesting = list(set(range(nb_fields)) - {11})
# Xi = np.array(Xi)[:, interesting]
# Xv = np.array(Xv)[:, interesting]
nb_features = (1 + np.array(Xi).astype(np.int32).max(axis=0)).sum()
# nb_fields -= 1  # Foutu champ texte
print(nb_features, 'features over', nb_fields, 'fields')

Xi_train = Xi[:32000]
Xi_valid = Xi[32000:40000]
Xi_test = Xi[40000:]

Xv_train = Xv[:32000]
Xv_valid = Xv[32000:40000]
Xv_test = Xv[40000:]

y_train = y[:32000]
y_valid = y[32000:40000]
y_test = y[40000:]

# params
dfm_params = {
    "use_fm": options.fm,
    "use_deep": options.deep,
    "embedding_size": options.d,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [options.nb_neurons] * options.nb_layers,
    "dropout_deep": [0.6] * (options.nb_layers + 1),
    "deep_layers_activation": tf.nn.relu,
    "epoch": options.iter,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "loss_type": 'mse',
    "verbose": True,
    "eval_metric": mean_squared_error,
    "random_seed": 2017,
    'feature_size': nb_features,
    'field_size': nb_fields
}

# init a DeepFM model
dfm = DeepFM(**dfm_params)

# fit a DeepFM model
dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)

# evaluate a trained model
rmse_train = dfm.evaluate(Xi_train, Xv_train, y_train) ** 0.5
rmse_valid = dfm.evaluate(Xi_valid, Xv_valid, y_valid) ** 0.5
rmse_test = dfm.evaluate(Xi_test, Xv_test, y_test) ** 0.5
print('train rmse={:f} valid rmse={:f} test rmse={:f}'.format(rmse_train, rmse_valid, rmse_test))

# make prediction on test
y_pred = dfm.predict(Xi_test, Xv_test)
print(y_pred[:10])
print(y_test[:10])
# with open('y_pred-{:.3f}-{:.3f}.txt'.format(auc_train, auc_valid), 'w') as f:
#     f.write('\n'.join(map(str, y_pred)))
