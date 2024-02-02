[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


# Frenetix-RL

This repository includes a PPO Reinforcement Learning accelerated trajectory planning algorithm in the [CommonRoad](https://commonroad.in.tum.de/) scenario format.
The trajectories of the analytical planner are generated according to the sampling-based approach in [1-5] including two different implementations.
The Repo provides a python-based and a C++-accelerated Motion Planner [Frenetix](https://github.com/TUM-AVS/Frenetix/) implementation.


### Requirements
The software is  developed and tested on recent versions of Linux. We strongly recommend to use [Ubuntu 22.04](https://ubuntu.com/download/desktop) or higher.
For the python installation, we suggest the usage of Virtual Environment with Python 3.10 or Python 3.9
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)


<details>
<summary> <h2> 🔧 Optional Pre-Installation Instructions (Ubuntu) </h2> </summary>

Make sure that the following **dependencies** are installed on your system for the C++ implementation:
   * [Eigen3](https://eigen.tuxfamily.org/dox/) 
     * On Ubuntu: `sudo apt-get install libeigen3-dev`
   * [Boost](https://www.boost.org/)
     * On Ubuntu: `sudo apt-get install libboost-all-dev`
   * [OpenMP](https://www.openmp.org/) 
     * On Ubuntu: `sudo apt-get install libomp-dev`
   * [python3.11-full](https://packages.ubuntu.com/jammy/python3.10-full) 
        * On Ubuntu: `sudo apt-get install python3.10-full` and `sudo apt-get install python3.10-dev`

</details>

1. **Clone** this repository & create a new virtual environment `python3.10 -m venv venv`

2. **Install** the package:
    * Source & Install the package via pip: `source venv/bin/activate` & `pip install .` or with [poetry](https://python-poetry.org/) `poetry install`
    * [Frenetix](https://pypi.org/project/frenetix/) should be installed automatically. If not please write [rainer.trauth@tum.de](mailto:rainer.trauth@tum.de).

3. **Optional** download of additional [scenarios](https://commonroad.in.tum.de/scenarios) and copy them to the folder `scenarios` or `scenarios_validation`:

4. **Optional change** of configurations in the following files & folders:
   1. `configurations` --> Analytic planner configs
   2. `frenetix-rl/gym_environment/congigs.yaml` --> RL training environment config
   3. `frenetix-rl/hyperparams/ppo2.yml` --> PPO hyperparameter settings

5. There is already a best_model to execute if you do not want to train one by yourself. Skip step 6 to skip the training procedure.
6. **Start Training**  with `python3 train.py`
7. **Logs** can be found in the `logs` folder. **tensorboard_logs** can be found in the `logs_tensorboard` folder. If you want to visualize them, install tensorboard with `pip` and execute `tensorboard --logdir logs_tensorboard/PPO_1/`. 
8. **Execution** of the trained model can be done with `python3 execute.py`. The **plot** visualizations of the executed model will be saved in `logs` again.

<details>
<summary> <h2> 📈 Test Data </h2> </summary>

Additional scenarios can be found [here](https://commonroad.in.tum.de/scenarios).
Load the files and add them to `scenarios` for training data or `scenarios_validation` for validation data.

</details>

<details>
<summary> <h2> 🔧 Modules </h2> </summary>

Detailed documentation of the functionality behind the single modules can be found below.

1. [General Planning Algorithm](README.md)

2. [Frenetix Motion Planner](https://github.com/TUM-AVS/Frenetix-Motion-Planner)

3. [Frenetix C++ Trajectory Handler](https://github.com/TUM-AVS/Frenetix)

4. [Wale-Net](https://github.com/TUMFTM/Wale-Net)

5. [Risk-Assessment](https://github.com/TUMFTM/EthicalTrajectoryPlanning)

</details>

<details>
<summary> <h2> 📇 Contact Info </h2> </summary>

[Rainer Trauth](mailto:rainer.trauth@tum.de),
Institute of Automotive Technology,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

[Johannes Betz](mailto:johannes.betz@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

</details>

<details>
<summary> <h2> 📃 Citation </h2> </summary>
   
If you use this repository for any academic work, please cite our code:

```bibtex
@misc{GitHubRepo,
  author = {Rainer Trauth},
  title = {Frenetix RL},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.10078062},
  url = {https://github.com/TUM-AVS/Frenetix-RL}
}
```
</details>