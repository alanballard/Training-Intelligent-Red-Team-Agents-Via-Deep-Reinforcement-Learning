# Python code for executing simulations used in the "Training Intelligent Red Team Agents Via Deep Reinforcement Learning" project. 

This code uses the OpenAi Spinning Up platform and requires the installation of Python3, OpenAI Gym, and OpenMI. Instructions for installation can be found here: https://spinningup.openai.com/en/latest/user/installation.html

Spinning Up currently only supports Linux and OSX. A link to a workaround for Windows is provided in the installation link above. The steps I actually followed to run Spinning Up on Windows can be found here:
https://github.com/openai/spinningup/issues/343#issuecomment-876191439

This repository contains four folders. The contents of the folders will need to be added to the specified folders within your installation of spinningup. I have given example locations from my Ubuntu installation to give you an idea where you need to put your copy of the files.

**#1. gym_envs**

/home/alanubuntu/anaconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/

This folder contains a single file, \_\\_init\_\\_.py.

After installing Spinning Up, the simulation environment needs to be registered in the \_\_init\_\_.py file in the .../gym/envs/ folder. This \_\_init\_\_.py contains the code that needs to be registered, or you can can just copy/paste this file over the existing \_\_init\_\_.py file. This registration code contains default input values for the simulation. It's not necessary to change any of these values as they will all be specified at run-time.

**#2. gym_envs_wargame**

/home/alanubuntu/anaconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/wargame/

This folder contains two files: \_\_init\_\_.py and RL_ENV.py.
RL_ENV.py contains the definitions for environment, actions, etc. required to run this particular simulation in Spinning UP.
\_\_init\_\_.py imports the main class from RL_ENV.py

You may need to specify the output location for the epoch logger, Elogger. It's at the top of the RL_ENV.py code.
I hard-coded the output location and file name later in the Lanchester() class. You will need to update the address but leave the file name as-is. You can find it easily by searching for:
f = open("/home/alanubuntu/Downloads/spinningup/data/encounter_results.txt", "a") 

**#3. gym_envs_pytorch**

/home/alanubuntu/anaconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/pytorch/

This folder contains three Jupyter notebooks, DRL PPO Notebook, DRL VPG Notebook, and DRL TRPO Notebook.
Each of these notebooks is used to execute the simulations using either the PPO, VPG, or TRPO deep reinforcement learning algorithms. Details of the algorithms can be found here: https://spinningup.openai.com/en/latest/user/algorithms.html 

You will need to specify BASE_OUTPUT_DIR. This is where the code will look for the encounter_results.txt file produced in #2 above.
You will need to specify the location of the design matrix for each experiment using the variable Design_Matrix.

**#4. design_matrix**

/home/alanubuntu/Downloads/spinningup/data/Design Matrix/

This folder contains three csv files: Two Factor Interaction Design PPO, Two Factor Interaction Design VPG, and Two Factor Interaction Design TRPO.

These are the variable settings at which the simulation was conducted, by algorithm. This is the location and the files specified as Design_Matrix in #3 above.
