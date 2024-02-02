# Brief guide to major files
## train_model.py
The major file to execute OpenAI Stable Baselines with CommonRoad Gym environment. There are three main operations to be chosen from, namely 
continual training, hyperparameters/configurations optimizing, and learning from scratch.  

### Continual training  
Use `-i` (string) to indicate continual training and pass in the path to the pretrained agent.  
### Hyperparameters/Configurations optimizing   
Use `--optimize-hyperparams` (bool), `--optimize-observation-configs` (bool), or `--optimize-reward-configs` (bool) to indicate optimizing either the model 
hyperparameters, the observation configurations, the reward configurations, or all.  
For `--optimize-reward-configs` one can set the flag `--guided` which changes some rewards based on the performance in the last
trials.
#### Optimization Process
During optimization, a given number of trials are carried out, and the result of the best trial will be reported.  
For each trial, a set of hyperparameters/configurations are sampled and passed to an agent. Sampling items with their sampling methods and candidate values 
are to be specified in `./gym_commonroad/configs.yaml`, which are read into `./train_model.py` 
during execution. There is no such a file for model hyperparameters because the items are rather static for all models, and thus the settings are made directly 
within `./utils_run/hyperparams_opt.py`.  
Having obtained the sampled hyperparameters/configurations, a learning process with a given number of time steps is provoked, and there will be a predefined 
number of evaluations distributed evenly over the learning process, against the following criteria. 
#### Evaluation criteria
To enable optimization and report the best set of hyperparameters/configurations, different cost functions are used together with the [Optuna Package](https://optuna.org).  
For **model hyperparameters**, the cumulative reward within an episode is naturally the optimization criterion. The trial achieving the highest episodic reward, meaning the lowest cost, is reported.   
For **observation configurations**, the cumulative reward is also a suitable optimization criterion implemented at the moment.  
For **reward configurations**, contrarily to the two cases above, there should be a different optimization criterion other than the cumulative reward itself. 
Currently, rates of termination are employed in the cost function. These include goal reaching which has a positive effect, as well as colliding and going 
off-road which have a negative effect. The trial achieving the lowest cost is reported. Finally, with the flexibility of configurable termination conditions, the cost function is to be further improved.
#### Result folder 
If `log_path` and `best_model_save_path` are given to the function, trial information including sampled values and intermediate evaluation results are saved 
under `log_path/trial_<no.>` and `best_model_save_path/trial_<no.>` within the optimization result folders. See below for an example:  
```
└── log/ppo2/commonroad-v0_<no.>/
    ├── model_hyperparameters.yml                       -> User-specified model hyperparameters
    ├── environment_configurations.yml                  -> User-specified environment configurations
    ├── model_hyperparameter_optimization/              -> Folder containing optimization information
    |   ├── report_<settings>.yml                       -> Best set of model hyperparameters reported from optimization 
    |   └── trial_<no.>/                                -> Folder containing trial information
    |       ├── best_model.zip                          -> Best model obtained during this trial
    |       ├── evaluations.npz                         -> Evaluation results such as episode lengths and episode rewards during this trial
    | 	    └── sampled_model_hyperparameters.yml       -> Sampled values used during this trial
    ├── observation_configuration_optimization/         -> Folder containing optimization information
    |   ├── report_<settings>.yml                       -> Best set of environment configurations reported from optimization 
    |   └── trial_<no.>/                                -> Folder containing trial information
    |       ├── best_model.zip                          -> Best model obtained during this trial
    |       ├── evaluations.npz                         -> Evaluation results such as episode lengths and episode costs during this trial
    | 	    └── sampled_observation_configuration.yml   -> Sampled values used during this trial
    └── <Others>
``` 
### Learning from scratch  
Use `--hyperparams-path` (string) and `--configs-path` (string) to specify paths if any predefined settings such as a result from optimization are to be used. 
Otherwise, an ordinary training with default model hyperparameters in `./hyperparams/<model>.yml` and user-specified environment configurations in `./configs.yml` 
is conducted. A detailed documentation of environment configurations can be found in `./gym_commonroad/README.md`.
  
Additional settings such as `--eval-freq` and `--save-freq` are possible for all three operations. Please see the file for more details.   

For example the command to start the learning could look like:
```
python train_model.py --env='commonroad-v1' --algo=ppo2 --n_envs=8 -n=1000000 --save-freq=50000 --seed=0 --save_path='/path/to/your/saving/location/'
 --hyperparams_path='/path/to/your/file/model_hyperparameters.yml' --configs-path='path/to/your/file/environment_configs.yml'
```
Also check that the PATH_PARAMS in `constants.py` match the paths to your dataset. 

## plot_learning_curves.py
This file is to generate the learning curves for different trainings. Use `-f` to specify the location of the log folder, 
`--model_path` as the relative path of the model root folder from the log folder, and `--legend_name` to specify the legend
name for each model (must be same as model folder name). Modify the `PLOT_DICT` to specify which plots should be generated.
Please see the file for further arguments.

A minimal example would be:

```
python plot_learning_curves.py -f='/path/to/your/log' --model_path 'ppo2/' --legend_name 'hybrid reward' 'sparse reward'
```

## sensitivity_analysis.py
This file can be used to perform sensitivity analysis on a trained model. Use `--model_path` to specify the path to the folder containing the trained models, use `--save_path` to specify the output location of the analysis, use `--data_path` to specify the path to the pickles the model was trained on, use `--config_filename` to specify the name of the config file and use `--algo` to specify the algorithm the model was trained on. 
`--method` allows to choose a method from the set [sobol, fast, rbd_fast, morris, delta], with `-n` to set the number of samples for the analysis. `--save_fig` and `--save_data` are flags that when set lead to the figs/ data being save in the folder specified by data_path.

A minimal example would be:
```
python sensitivity_analysis.py -algo=ppo2  --model_path='/path/to/your/model/log/ppo2/commonroad-v1_1' -n 256 --algo ppo2  --save_fig --method sobol
```

## evaluate_model.py
The major file to play a result after learning, visualizing how the ego vehicle drives in scenarios across time steps.  
Use `--model_path` to specify the path to the trained model, `--test_path` the path to the pickled scenarios, and `--viz_path` the path to save the resulting images.
In default, `<model_path>/model_hyperparameters.yml` and `<model_path>/environment_configurations.yml` are read in for the environment construction. 
Otherwise, use `--hyperparam_filename` and `--config_filename` to pass in different files.  
A tip would be to change directly the settings in the yaml files, for example during rendering of different coordinate systems or surrounding obstacles.   
Please see the file for further arguments.

A minimal example would be:
```
python evaluate_model.py --env='commonroad-v1' --algo=ppo2  --model_path='/path/to/your/model/log/ppo2/commonroad-v1_1'
 --test_path='/path/to/your/test/pickle_scenario_files' --viz_path='path/to/your/saving/location'
```

## generate_solution.py
The major file to convert a trained result to a solution for [CommonRoad](https://commonroad.in.tum.de).  
Use `--model_path` to specify the path to the trained model, `--test_path` the path to the pickled scenarios, and `--solution_path` the path to save the resulting solution.
In default, `<model_path>/model_hyperparameters.yml` and `<model_path>/environment_configurations.yml` are read in for the environment construction. 
Otherwise, use `--hyperparam_filename` and `--config_filename` to pass in different files.  
Please see the file for further arguments.

A minimal example would be:
```
python generate_solution.py --algo=ppo2  --model_path='/path/to/your/model/log/ppo2/commonroad-v1_1'
 --test_path='/path/to/your/test/pickle_scenario_files' --solution_path='path/to/your/saving/location'
```
