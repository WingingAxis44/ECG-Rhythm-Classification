# AI Assignment 2
Welcome to the AI-ECG Platform. We have made it that the code is simple for you to install and get up and running! We use Python virtual environments to run our python and tensorflow code. We have created a make file that will allow you to easily run the exact same experiments as we ran in our assignment. Follow the steps below to run the experiments as outlined in our assignment.


## Installation
1. The first step to installation is to check the *requirements.txt* file for the correct tensorflow version.  If you are using a Mac computer with an M1 Chip, replace the line which states: "tensorflow" with "tensorflow-macos"
2. Ensure you are in the root "AI_Ass2" directory
3. Run the following commands to set up your virtual environment and install the required dependencies:
```sh
make
```
Your packages should sucessfully be installed. If you wish to remove your virtual environment run the following command
```sh
make clean
```

## Running the experiments
We will first demonstrate how to run the experiments with predefined makefile commands we have created for you.
We will then show you how to run the program in custom ways.
The main procedures of the codebase happen in the wrapper.py python file.
Please note that the following three commands will create new data splits, therefore the results of training may differ slightly
from those reported in our report. This is due to the stochastic nature of gradient descent used in training neural networks.
These runs also do not perform over-sampling or normalization. However, results should be mostly the same as our SD was low

### Running with the baseline Fully connected model.
To run this study's baseline Fully connected model with the optimal hyperparameters found (dropout_rate = 0.2, batch_size = 128, learning_rate = 0.001), type the following:
```sh
make runSimple
```

### Running with proposed LSTM model.
To run this study's proposed LSTM classifier with the optimal hyperparameters found (dropout_rate =0.6, batch_size = 64, learning_rate = 0.001), type the following:
```sh
make runLSTM
```

### Running with proposed CNN model.
To run this study's proposed CNN classifier with the optimal hyperparameters found (dropout_rate =0.3, batch_size = 64, learning_rate = 0.001), type the following:
```sh
make runCNN
```

Note: Any one of the 3 above runs will create new data splits, so please allow time for this (about 2 minutes) before training can start.

## Running the platform in custom ways
Our platform allows you to interact with it in many custom ways. There are multiple arguments that you can pass to the platform for customized model training. 
For help with this you can run the make command:
```sh
make runHelp
```
The structure of a custom run is as follows (run from the root directory):
```sh
venv/bin/python3 src/wrapper.py "./trained_models/final_model" <args> <args> <args> .....
```
You may replace the "`<args> <args> <args>...`" with the following arguments to customize model training:

- **-p** : This is used to specify the preprocessing options you require. The options here are "oversample" and "normalize". To normalize and oversample your ECG segments use "-p oversample normalize". Default is None 
- **-e** : How many epochs do you wish to train your model for. Must be >1 and <1024. e.g. To run for 500 epochs enter: "-e 500". Default is 5.
- **-o** : Flag that can be passed to perform hyper-parameter tuning (requires a wandb account and login key).
- **-r** : Flag that can be passed to resume training of a model. The program will look for a backup of a model based on the model path arg provided. There also needs to be saved training, validation and testing data splits.
- **-s** : skip training. Flag that can be passed to skip building and training the model. Thereby moving straight to predictions. It follows then that the model provided with the model path arg is already compiled and trained for at least one epoch. There also needs to be saved training, validation and testing data splits.
- **-v** : Verbosity. Can choose either -v 1 for verbose output or -v 0 for quiet output. Default is Verbose
- **-m** : This is the model you can chose to train with. There are several options. These include the simple FFNN, LSTM, and the CNN used in our assignment. Default is the simple FFNN.
- **-d** : Flag that can be passed to disable training with GPU. 
- **-ls** : Each time data splits are created they are saved to numpy files. This argument will load the saved data splits instead of resampling the ecgs. This is a flag that can be passed to load previously saved training, validation and test data splits from disk.  Note: If resuming training or skipping to predictions, data splits will always be loaded from disk to maintain integrity in the evaulation of the model
