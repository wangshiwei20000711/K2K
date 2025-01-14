Kriging to KAN (K2K)

The repository shows an efficient method for training surrogate models to predict multiphysical process data from limited samples. 

The development of the K2K model integrates the Kolmogorov-Arnold Network (KAN) proposed by Ziming Liu et al., alongside the classical Kriging algorithm. The original KAN implementation is available at https://github.com/KindXiaoming/pyKAN.

We provide scripts to run test cases in TableI.

Adjustable settings are mainly in Kan_agient.py. Before executing the code, please ensure that the file path variable File_save_path in Agent_Func.py is appropriately modified.

Executes with: $ python Kan_agient.py and may take anywhere from a few hours to a few days, depending on the model and settings used.
