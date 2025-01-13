Kriging to KAN (K2K)

The repository shows an efficient method for training surrogate models to predict multiphysical process data from limited samples. 

The development of the K2K model integrates the Kolmogorov-Arnold Network (KAN) proposed by Ziming Liu et al., alongside the classical Kriging algorithm. The original KAN implementation is available at https://github.com/KindXiaoming/pyKAN.

We provide scripts to run test cases in TableI.

Adjustable settings are in _model.py.

Executes with: $ python main_workflow.py and may take anywhere from a few hours to a few days, depending on the model and settings used. Relevant information is printed into several files:

cost.pkl: the objective function
hist.pkl: historical testing misfit
size.pkl: historical endpoint cache size
score.txt: convergence of the test score versus model evaluations
stop: a database of the endpoints of the optimizers
func.db: a database of the learned surrogates
eval: a database of evaluations of the objective
