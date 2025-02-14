# Discrete Abstraction for Sparse Deep Reinforcement Learning

## Introduction

This is the code for the paper "Discrete Abstraction for Sparse Deep Reinforcement Learning"

## Installation

To install the framework, follow these steps:

1. Clone the GitHub repository:

   ```bash
   git clone https://github.com/coldfrenzy/dadrl.git
   ```

2. cd into the code directory:

   ```
   cd ./Discrete_Abstraction_for_Sparse_DRL
   ```

   

3. Install the package in development mode to make changes to the code and have them immediately reflected without needing to reinstall the package:

   ```bash 
   # without additional dependencies
   pip install -e .
   
   # or with additional dependencies
   pip install -e .["tensorboard"]

## Docker Installation 

To run the code within a Docker Container, follow the instruction below:

1. cd into the docker directory:

   ```bash
   cd ./docker
   ```

   

2. Create docker image

   ```
   docker build -t dadrl .
   ```

   

3. Run the container


```bash
# to run interactively using the gpu (remove --rm to make the container persistent)
    docker run --rm  --gpus all -it --entrypoint="" dadrl /bin/bash


Note: current folder is mounted in the container in ```/opt```
```

## Run the Code

You can run a custom SAC implementation, a custom SAC+HER or the StableBaseline3 version of both algorithms:

```bash
# custom SAC
python ./src/main_sac.py
```

``` bash
# custom SAC + HER
python ./src/main_sac_her.py
```

```bash
# SB3 SAC or SAC+HER
python ./src/main_sb3.py
```

