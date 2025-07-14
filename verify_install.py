from K2K import *

"""
This script is designed to verify whether the code runs successfully in the current environment.

It performs basic import checks and confirms the availability of essential packages.

Run this script after setting up the environment as instructed in the README file.

It is essential to update the K2KSave_path and ALl_data_path variables to the absolute storage paths of the K2K project before running the script.

If the script finishes and returns "Code installed and running successfully", it indicates that the code has been properly installed.
"""

# Please set the paths to the K2K project directory before running the script!!!
K2KSave_path = r'E:\MeMyselfAndI\PycharmProjects\K2K\Dataset'  # the storage path for K2K output results
ALl_data_path = r'E:\MeMyselfAndI\PycharmProjects\K2K\Dataset\TestFunction_Dataset.txt'  # the location of the input data file


iKanTrain = 1  # set to 1 if model training is required; otherwise, set to 0.
iKanContinue = 0  # set to 1 if the model should load existing model parameters. If symbolic training results also need to be loaded, set to 2; otherwise, set to 0.
iKanSave = 1  # set to 1 if model parameters need to be saved; otherwise, set to 0.
iKanValidation = 1  # set to 1 if model accuracy verification based on the data file is required; otherwise, set to 0.
iSymbolic = 1  # set to 1 if symbolic training is needed; otherwise, set to 0.
ipredict = 0 # set to 1 if real-time prediction is needed; otherwise, set to 0.

num_in = 2  # the number of input dimensions
num_out = 3  # the number of output dimensions
num_using = 1500  # the amount of training data
Kriging_UsingData = 100  # the amount of data used for Kriging training, which must be less than the training data amount. It is recommended to use 40-60% of the training data amount
Krigng_edge = np.array([[0, 5], [0, 5]])  # a numpy array of size “num_in*2”. The first column contains the input design domain's minimum value, and the second column contains its maximum value.
Kriging_theta = np.array([2.5, 2.5])  # a numpy array of size “num_in”, representing the initial values for Kriging training. It is recommended to use the median of the input design domain.

LOSS_Func = None
pre_train_step = 2  # the number of pre-training epochs for KAN correction.
prune_trian_step = 2  # the number of prune-training epochs for KAN correction.
symbolic_trian_step = 20  # the number of epochs for regression training after the symbolic processing of KAN correction.
train_steps = [pre_train_step, prune_trian_step, symbolic_trian_step]
Kan_structure = [2, 2, 3]
Kan_grid = 5
Kan_k = 3
Kan_seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Kan_grid_eps = 0.02

maxData = torch.tensor([5, 5])  # a torch tensor of size “num_in” represents the minimum value of the input design domain.
minData = torch.tensor([0, 0])  # a torch tensor of size “num_in” represents the maximum value of the input design domain.
K2K_Main(K2KSave_path, ALl_data_path, num_using, Kriging_UsingData, num_in, num_out, Krigng_edge, Kriging_theta, train_steps, Kan_structure, device, LOSS_Func=LOSS_Func, iKanTrain=iKanTrain, iKanContinue=iKanContinue, iKanSave=iKanSave, iKanValidation=iKanValidation, iSymbolic=iSymbolic, Kan_grid=Kan_grid, Kan_k=Kan_k,Kan_seed=Kan_seed, Kan_grid_eps=Kan_grid_eps, maxData=maxData, minData=minData, ipredict=ipredict)

print("\n\nCode installed and running successfully")