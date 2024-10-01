import os
import numpy as np

import train
import eval

train_params = {}
train_params['species_set'] = 'all'
train_params['num_aux_species'] = 0
train_params['input_enc'] = 'sin_cos'
train_params['loss'] = 'an_full'

# Adjust the following hyper-parameters:
train_params['experiment_name'] = 'demo' # Set the name of the directory where results will be saved
train_params['lr'] = 0.001 # Learning rate
train_params['ratio'] = 0.5 # Implicitness
train_params['hard_cap_num_per_class'] = 10 # Observation cap per species

train.launch_training_run(train_params)

# evaluate:
for eval_type in ['snt', 'iucn', 'geo_feature']:
    eval_params = {}
    eval_params['exp_base'] = './experiments'
    eval_params['experiment_name'] = train_params['experiment_name']
    eval_params['eval_type'] = eval_type
    cur_results = eval.launch_eval_run(eval_params)
    np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)
