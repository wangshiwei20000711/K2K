import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(33)
from KANpack.utils import create_dataset
from KANpack.KAN import *
import K2K_Kriging
import warnings
warnings.filterwarnings("ignore")
import time



def Func_txt_to_list(input_path, mode=0):
    path = input_path
    fp = open(path, encoding='utf-8', errors='ignore')
    lines = fp.readlines()
    NumLine = int(len(lines))
    fp.close()
    NumColumn = 0
    for i, line in enumerate(lines):
        temp = line.split()
        if len(temp) > NumColumn:
            NumColumn = len(temp)
    x = np.zeros((NumLine,NumColumn))
    for i, line in enumerate(lines):
        temp = line.split()
        for j in range(len(temp)):
            if mode==0:
                x[i][j]=temp[j]
            elif mode==1:
                if j%2==1:
                    x[i][j] = temp[j]
    return x

def K2K_Main(K2KSave_path, ALl_data_path, num_using, Kriging_UsingData, num_in, num_out, Krigng_edge, Kriging_theta, train_steps, Kan_structure, device, LOSS_Func=1, iKanTrain=1, iKanContinue=0, iKanSave=1, iKanValidation=1, iSymbolic=1, Kan_grid=5, Kan_k=3,Kan_seed=0, Kan_grid_eps=0.02, maxData=0, minData=0, ipredict=1):
    ## K2K model establishing
    model = KAN(width=Kan_structure, grid=Kan_grid, k=Kan_k, seed=Kan_seed,device=device,grid_eps=Kan_grid_eps)
    if iKanContinue==1:
        model.load_ckpt('KanAgent',K2KSave_path)
    elif iKanContinue==2:
        model.load_ckpt('KanAgent',K2KSave_path)

    pre_train_step = train_steps[0]
    prune_trian_step = train_steps[1]
    symbolic_trian_step = train_steps[2]

    ## Define data
    ALl_data = Func_txt_to_list(ALl_data_path)
    m = ALl_data.shape[0]
    Kriging_input_data = torch.from_numpy(ALl_data[:,0:num_in])
    Kriging_output_data = torch.from_numpy(ALl_data[:,num_in:(num_in+num_out)])

    Kriging_x = Kriging_input_data[0:Kriging_UsingData,:]
    Kriging_y = Kriging_output_data[0:Kriging_UsingData,:]
    Kriging_x_test = Kriging_input_data[Kriging_UsingData:m,:]
    Kriging_y_test = Kriging_output_data[Kriging_UsingData:m,:]
    diydata = [Kriging_x,Kriging_x_test,Kriging_y,Kriging_y_test]
    dataset = create_dataset(Kriging_y, n_var=2,device=device,DiY=1,DiYdata=diydata)
    print('Kriging-train using data',num_using,'/',m)

    #Kriging estimation training
    K_model = K2K_Kriging.Kriging_Main(inputnum=num_in,outputnum=num_out,using_num=Kriging_UsingData,edge=Krigng_edge,theta=Kriging_theta,Input_path=ALl_data_path,Output_path=K2KSave_path+r'\K2K_Kriging_outputs.txt',regFunc=K2K_Kriging.regpoly2,corrFunc=K2K_Kriging.corrgauss)
    print("The Kriging estimation has been successfully trained.")

    #Generation of KAN correction training data
    x = Kriging_input_data[0:num_using,:]
    y = Kriging_output_data[0:num_using,:]
    test_data = ALl_data[num_using:m,:]
    x_test = Kriging_input_data[num_using:m,:]
    y_test = Kriging_output_data[num_using:m,:]

    x_real = x.numpy()
    y_real = y.numpy()
    x_test_real = x_test.numpy()
    y_test_real = y_test.numpy()


    KAN_trainInput = open(K2KSave_path + r'\Kriging_KanTrainErrorDownData.txt', mode='w')
    for i in range(x_real.shape[0]):
        TrainData_validation,_,_,K_mSe = K2K_Kriging.predictor(x_real[i,:],K_model)
        TrainData_error =  y_real[i,:]-TrainData_validation
        for j in range(x_real.shape[1]):
            KAN_trainInput.write(str(x_real[i, j]) + '\t')
        for j in range(TrainData_error.shape[0]):
            KAN_trainInput.write(str(TrainData_error[j]) + '\t')
        KAN_trainInput.write('\n')
    KAN_trainInput.close()

    KAN_testInput = open(K2KSave_path + r'\Kriging_KanTestErrorDownData.txt', mode='w')
    for i in range(x_test_real.shape[0]):
        TestData_validation, _, _, K_mSe = K2K_Kriging.predictor(x_test_real[i,:],K_model)
        TestData_error = -TestData_validation + y_test_real[i, :]
        for j in range(x_test_real.shape[1]):
            KAN_testInput.write(str(x_test_real[i, j]) + '\t')
        for j in range(TestData_error.shape[0]):
            KAN_testInput.write(str(TestData_error[j]) + '\t')
        KAN_testInput.write('\n')
    KAN_testInput.close()

    # Loading of KAN training data
    KAN_train_data = Func_txt_to_list(K2KSave_path + r'\Kriging_KanTrainErrorDownData.txt')
    KAN_test_data = Func_txt_to_list(K2KSave_path + r'\Kriging_KanTestErrorDownData.txt')
    KAN_input_data = torch.from_numpy(np.vstack((KAN_train_data[:, 0:num_in], KAN_test_data[:, 0:num_in])))
    KAN_output_data = torch.from_numpy(np.vstack((KAN_train_data[:, num_in:(num_in+num_out)], KAN_test_data[:, num_in:(num_in+num_out)])))

    mS = torch.mean(KAN_input_data, dim=0)
    sS = torch.std(KAN_input_data, dim=0)
    mY = torch.mean(KAN_output_data, dim=0)
    sY = torch.std(KAN_output_data, dim=0)
    KAN_x_real = KAN_input_data[0:num_using,:].numpy()
    KAN_x_test_real = KAN_input_data[num_using:m,:].numpy()
    if len(KAN_output_data.shape) == 1:
        KAN_output_data = KAN_output_data.reshape(-1, 1)
    # Check for 'missing dimension'
    j = (np.where(sS == 0))[0]
    if len(j) != 0:
        sS[j.astype(int)] = 1
    j = torch.where(sY == 0)[0]
    if len(j) != 0:
        sY[j.astype(int)] = 1
    # Normalization
    KAN_input_data = ((KAN_input_data - torch.tile(mS, (m, 1))) / torch.tile(sS, (m, 1))).to(device)
    KAN_output_data = ((KAN_output_data - torch.tile(mY, (m, 1))) / torch.tile(sY, (m, 1))).to(device)

    # Define data
    KAN_x = torch.tensor(KAN_input_data[Kriging_UsingData:num_using, :], dtype=torch.float64, device=device)
    KAN_x_ea = torch.tensor(KAN_input_data[0:num_using, :], dtype=torch.float64, device=device)
    KAN_x_test = torch.tensor(KAN_input_data[num_using:(KAN_input_data.shape[0]), :], dtype=torch.float64, device=device)
    KAN_y = torch.tensor(KAN_output_data[Kriging_UsingData:num_using, :], dtype=torch.float64, device=device)
    KAN_y_test = torch.tensor(KAN_output_data[num_using:(KAN_output_data.shape[0]), :], dtype=torch.float64, device=device)
    diydata = [KAN_x, KAN_x_test, KAN_y, KAN_y_test]
    dataset = create_dataset(KAN_y, n_var=num_in, device=device, DiY=1, DiYdata=diydata)


    model(dataset['train_input'])

    if iKanTrain==1:
        ## Train the KAN correction with LBFGS optimizer
        print('1Trian:')
        error_pre_train = model.train(dataset, opt="LBFGS", steps=pre_train_step, lamb=0.01, lamb_entropy=10.,device=device,loss_fn=LOSS_Func)
        model = model.prune()
        print('2Prune is done')
        dontdelete = model(torch.ones(1,num_in,device=device))
        KanProcess = open(K2KSave_path+r'\KanProcess.txt','w')
        for i in range(pre_train_step):
            KanProcess.write('error_pre_train\t'+str(i)+'\t'+str(error_pre_train['train_loss'][i])+'\t'+str(error_pre_train['test_loss'][i])+'\t'+str(error_pre_train['reg'][i])+'\n')
        KanProcess.close()
        ## Prune training
        print('3Trian again:')
        error_prune_train = model.train(dataset, opt="LBFGS", steps=prune_trian_step,device=device,loss_fn=LOSS_Func)
        KanProcess = open(K2KSave_path + r'\KanProcess.txt', 'a')
        for i in range(prune_trian_step):
            KanProcess.write('error_prune_train\t'+str(i) + '\t' + str(error_prune_train['train_loss'][i]) + '\t' + str(error_prune_train['test_loss'][i]) + '\t' + str(error_prune_train['reg'][i]) + '\n')
        KanProcess.close()
    if iKanSave==1:
        model.save_ckpt('KanAgent',K2KSave_path)

    if iSymbolic==1:
        lib = ['x','x^2','tanh','sin','abs','x^3','x^4'] #,,'exp','sqrt','log'
        model.auto_symbolic(lib=lib)
        print('Activate function adding is done')
        if iKanContinue!=2:
            print('4Final trian:')
            error_symbolic_train = model.train(dataset, opt="LBFGS",steps=symbolic_trian_step,device=device,loss_fn=LOSS_Func, lr=0.1)
            if iKanTrain==1:
                KanProcess = open(K2KSave_path + r'\KanProcess.txt', 'a')
            else:
                KanProcess = open(K2KSave_path + r'\KanProcess.txt', 'w')
            for i in range(symbolic_trian_step):
                KanProcess.write('error_symbolic_train\t'+str(i) + '\t' + str(error_symbolic_train['train_loss'][i]) + '\t' + str(error_symbolic_train['test_loss'][i]) + '\t' + str(error_symbolic_train['reg'][i]) + '\n')
            KanProcess.close()
        elif iKanContinue==2:
            model.load_ckpt('KanAgentSym', K2KSave_path)
        print('5Symbolic formula:')
        print(model.symbolic_formula())
        if iKanSave==1:
            # Save model parameters
            model.save_ckpt('KanAgentSym',K2KSave_path)

    print('The KAN correction has been successfully trained')

    def Un_Normalise(y, sY, mY, mutiplier=123456):
        if mutiplier==123456:
            return y*torch.tile(sY, (y.shape[0], 1))+torch.tile(mY, (y.shape[0], 1))
        else:
            return (y*torch.tile(sY, (y.shape[0], 1))+torch.tile(mY, (y.shape[0], 1)))*torch.tile(mutiplier, (y.shape[0], 1))


    if iKanValidation==1:
        TrainData_validation = Un_Normalise((model(KAN_x_ea).detach().to('cpu')),sY,mY).numpy()
        Train_valid_pre = np.zeros_like(TrainData_validation)
        for i in range(Train_valid_pre.shape[0]):
            Train_valid_pre[i, :], _, _, _ = K2K_Kriging.predictor(KAN_x_real[i,:],K_model)
        Train_y_output = TrainData_validation + Train_valid_pre
        TrainData_error = Train_y_output - y.detach().to('cpu').numpy()
        Train_x_output = x.detach().to('cpu').numpy()

        K2K_Train_Valid_output = open(K2KSave_path + r'\Kan_ErrorAnalysis.txt', mode='w')
        for i in range(TrainData_validation.shape[0]):
            K2K_Train_Valid_output.write('Train_data\t '+str(i) + '\t')
            for j in range(Train_x_output.shape[1]):
                K2K_Train_Valid_output.write(str(Train_x_output[i, j]) + '\t')
            for j in range(y.shape[1]):
                K2K_Train_Valid_output.write(str(y[i, j].numpy()) + '\t')
            for j in range(Train_y_output.shape[1]):
                K2K_Train_Valid_output.write(str(Train_y_output[i, j]) + '\t')
            for j in range(TrainData_error.shape[1]):
                K2K_Train_Valid_output.write(str(TrainData_error[i,j]) + '\t')
            K2K_Train_Valid_output.write('\n')
        K2K_Train_Valid_output.close()

        TestData_validation = Un_Normalise((model(KAN_x_test).detach().to('cpu')),sY,mY).numpy()
        Test_valid_pre = np.zeros_like(TestData_validation)
        for i in range(Test_valid_pre.shape[0]):
            Test_valid_pre[i, :], _, _, _ = K2K_Kriging.predictor(KAN_x_test_real[i,:],K_model)
        Test_y_output = TestData_validation + Test_valid_pre
        TestData_error = Test_y_output-y_test.detach().to('cpu').numpy()
        Test_x_output = x_test.detach().to('cpu').numpy()

        LotofXY_output = open(K2KSave_path + r'\Kan_ErrorAnalysis.txt', mode='a')
        for i in range(TestData_validation.shape[0]):
            LotofXY_output.write('Test_data\t'+str(i) + '\t')
            for j in range(Test_x_output.shape[1]):
                LotofXY_output.write(str(Test_x_output[i, j]) + '\t')
            for j in range(y_test_real.shape[1]):
                LotofXY_output.write(str(y_test_real[i, j]) + '\t')
            for j in range(Test_y_output.shape[1]):
                LotofXY_output.write(str(Test_y_output[i, j]) + '\t')
            for j in range(TestData_error.shape[1]):
                LotofXY_output.write(str(TestData_error[i,j]) + '\t')
            for j in range(TestData_validation.shape[1]):
                LotofXY_output.write(str(TestData_validation[i,j]) + '\t')
            for j in range(Test_valid_pre.shape[1]):
                LotofXY_output.write(str(Test_valid_pre[i,j]) + '\t')
            LotofXY_output.write('\n')
        LotofXY_output.close()

    print('\n\n####### Real-time prediction #######')
    print('(To exit, please stop the program or enter any letter.)\n')
    while ipredict:
        a = np.zeros(num_in)
        print('Please enter all '+str(num_in)+' input parameters one by one.')
        for i in range(num_in):
            a[i] = float(input('Please enter the ' + str(i + 1) + '-th input parameter in its original scale:'))
        x_use = torch.from_numpy(a)
        x_use_Kriging = x_use.numpy()
        x_use_KAN = ((x_use - mS) / sS).reshape(1, -1).to(device)
        y_use_KAN = (model(x_use_KAN).to('cpu')* sY + mY).detach().numpy()
        y_use_Kriging,_,_,_ = K2K_Kriging.predictor(x_use_Kriging, K_model)
        y = y_use_KAN + y_use_Kriging
        print('Prediction results：', y,'\n')

    return [K_model,model]

if __name__ == '__main__':
    start_time = time.time()

    # Please set the paths to the directories where your data are stored before running the script!!!
    K2KSave_path = r'E:\MeMyselfAndI\PycharmProjects\K2K\Dataset'  # the storage path for K2K output results
    ALl_data_path = r'E:\MeMyselfAndI\PycharmProjects\K2K\Dataset\TestFunction_Dataset.txt'  # the location of the input data file

    iKanTrain = 1  # set to 1 if model training is required; otherwise, set to 0.
    iKanContinue = 0  # set to 1 if the model should load existing model parameters. If symbolic training results also need to be loaded, set to 2; otherwise, set to 0.
    iKanSave = 1  # set to 1 if model parameters need to be saved; otherwise, set to 0.
    iKanValidation = 1  # set to 1 if model accuracy verification based on the data file is required; otherwise, set to 0.
    iSymbolic = 0  # set to 1 if symbolic training is needed; otherwise, set to 0.
    ipredict = 1 # set to 1 if real-time prediction is needed; otherwise, set to 0.

    num_in = 2  # the number of input dimensions
    num_out = 3  # the number of output dimensions
    num_using = 1500  # the amount of training data
    Kriging_UsingData = 100  # the amount of data used for Kriging training, which must be less than the training data amount. It is recommended to use 40-60% of the training data amount
    Krigng_edge = np.array([[0, 5], [0, 5]])  # a numpy array of size “num_in*2”. The first column contains the input design domain's minimum value, and the second column contains its maximum value.
    Kriging_theta = np.array([2.5, 2.5])  # a numpy array of size “num_in”, representing the initial values for Kriging training. It is recommended to use the median of the input design domain.

    LOSS_Func = None
    pre_train_step = 2  # the number of pre-training epochs for KAN correction.
    prune_trian_step = 2  # the number of prune-training epochs for KAN correction.
    symbolic_trian_step = 200  # the number of epochs for regression training after the symbolic processing of KAN correction.
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

    end_time = time.time()
    print('Operating time:', end_time - start_time)