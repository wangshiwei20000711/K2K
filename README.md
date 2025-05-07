Kriging to KAN (K2K)

The repository shows an efficient method for training surrogate models to predict multiphysical process data from limited samples.


The development of the K2K model integrates the Kolmogorov-Arnold Network (KAN) proposed by Ziming Liu et al., alongside the classical Kriging algorithm. The original KAN implementation is available at https://github.com/KindXiaoming/pyKAN.


To properly run and use the K2K model, the following requirements should be ensured.

1. A complete and correctly installed Python environment is required, with Python 3.9 recommended.

2. The necessary pre-installed packages include:

   numpy, torch, warnings, time, scipy, copy, matplotlib, os, glob, tqdm, random, sklearn, and functools. 

   These packages are commonly used Python libraries and are openly accessible through the Python community. The installed packages can be checked using pip list command to confirm that all required libraries are available. Missing packages can be installed using pip install package_name command.

4. The K2K program package should be downloaded and deployed locally.



After confirming that the device is equipped with all the required support programs, the K2K model can be used following the steps below.

1. Prepare the training dataset. Currently, the format of data file supports only “.txt”. 

   Each line corresponds to one data entry, containing n-dimensional inputs and m-dimensional outputs. 

   The first n columns in the data file represent the inputs, and the last m columns represent the outputs. 

   Parameters should be separated by tab, and no textual descriptions are allowed. 

   Training and testing data should be stored together in the same “.txt” file, with the training data listed before the testing data.     

   For demonstration purposes, a sample dataset with two inputs and three outputs is provided in the K2K package.

   (The test function is y1=x1^0.5*(x2+1)^1.5; y2=exp(sin(0.1*pi*y1)+lg((x1+1)/(x2+1))); y3=y1*ln(y2).)

3. Modify the input and output file paths for K2K. 
   The main program of the K2K package is “K2K.py”. Before running “K2K.py”, the variables “K2KSave_path” and “ALl_data_path” should be modified according to the local file storage locations. It is recommended to use absolute paths for both “K2KSave_path” and “ALl_data_path”.
    - K2KSave_path: the storage path for K2K output results.
    - ALl_data_path: the location of the input data file.

4. Modify the K2K operation mode. 
   The operation mode parameters of K2K include “iKanTrain”, “iKanContinue”, “iKanSave”, “iKanValidation”, and “iSymbolic”.
    - iKanTrain: set to 1 if model training is required; otherwise, set to 0.  
    - iKanContinue: set to 1 if the model should load existing model parameters. If symbolic training results also need to be loaded, set to 2; otherwise, set to 0.  
    - iKanSave: set to 1 if model parameters need to be saved; otherwise, set to 0.  
    - iKanValidation: set to 1 if model accuracy verification based on the data file is required; otherwise, set to 0.  
    - iSymbolic: set to 1 if symbolic training is needed; otherwise, set to 0.

5. Modify the K2K execution parameters. 
    The main execution parameters of K2K include,
    - num_in: the number of input dimensions.  
    - num_out: the number of output dimensions.  
    - num_using: the amount of training data.  
    - Kriging_UsingData: the amount of data used for Kriging training, which must be less than the training data amount. It is recommended to use 40-60% of the training data amount.  
    - Kriging_edge: a numpy array of size “num_in*2”. The first column contains the input design domain's minimum value, and the second column contains its maximum value.
    - Kriging_theta: a numpy array of size “num_in”, representing the initial values for Kriging training. It is recommended to use the median of the input design domain.  
    - pre_train_step: the number of pre-training epochs for KAN correction.  
    - prune_train_step: the number of prune-training epochs for KAN correction.  
    - symbolic_train_step: the number of epochs for regression training after the symbolic processing of KAN correction.
    - maxData: a torch tensor of size “num_in” represents the minimum value of the input design domain.  
    - minData: a torch tensor of size “num_in” represents the maximum value of the input design domain.

6. Run the “K2K.py” file. 

   After running the “K2K.py” file, the model saves the prediction results for both the training and testing datasets in the “Kan_ErrorAnalysis.txt” file, located in the “K2KSave_path” directory. 

   The content of the “Kan_ErrorAnalysis.txt” file includes the dataset type (testing or training), index, n-dimensional inputs, m-dimensional predicted outputs, and m-dimensional prediction errors.

   If model parameters are selected to be saved, the model parameter file is stored in the “K2KSave_path” directory under the name "KanAgent". 

   If symbolic training is selected, the model parameters after symbolic training are saved in the “K2KSave_path” directory under the name “KanAgentSb”.
