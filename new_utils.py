import os
import re
from options import config
from matplotlib import pyplot as plt 
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_experiment_folder(experiment_number):
    try:
        path_new_experiment = "ExperimentsSR/Experiment" + str(experiment_number)
        os.mkdir(path_new_experiment)
    except Exception as e:
        print ("Creation of the directory {} failed".format(path_new_experiment))
        print("Exception error: ",str(e)) 

def get_experiment_number():
    experiments_folders_list = os.listdir(path='ExperimentsSR/')
    if(len(experiments_folders_list) == 0): #empty folder
        return 1
    else:  
        temp_numbers=[]
        for folder in experiments_folders_list:
            number = re.findall(r'\d+', folder)
            if(len(number)>0):
                temp_numbers.append(int(number[0]))
        return max(temp_numbers) + 1
    
def create_info_training_file(experiment_number):
    filename = "ExperimentsSR/Experiment"+str(experiment_number)+"/infos_experience"+str(experiment_number)+".txt"
    with open(filename, "w") as file:
        file.write("")
        
        
def create_main_experiment_folder():
    if(not os.path.isdir("ExperimentsSR")):
        try:
            os.mkdir("ExperimentsSR")
        except Exception as e:
            print ("Creation of the main experiment directory failed")
            print("Exception error: ",str(e))
            
            
def get_folders_started():
    create_main_experiment_folder()      
    experiment_number = get_experiment_number()
    create_experiment_folder(experiment_number)
    create_summary_file(experiment_number)
    return experiment_number

def record_base_info(experiment_number,**kwargs):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "a+") as file:
        file.write("use_rescaled_MSE: {}\n".format(config["use_rescaled_MSE"]))
        file.write("epochs1: {}\n".format(config["epochs1"]))
        file.write("epochs2: {}\n".format(config["epochs2"]))  
        file.write("threshold_value: {}\n".format(config["threshold_value"]))
        file.write("a_L_0.5: {}\n".format(config["a_L_0.5"]))
        file.write("use_phase2: {}\n".format(config["use_phase2"]))
        file.write("use_thresholding_before_phase2: {}\n".format(config["use_thresholding_before_phase2"]))
        file.write("lambda_reg: {}\n".format(config["lambda_reg"]))
        file.write("batch_size: {}\n".format(config["batch_size"]))
        file.write("phase1_lr: {}\n".format(config["phase1_lr"]))
        file.write("phase2_lr: {}\n".format(config["phase2_lr"]))
        file.write("eql_number_layers: {}\n".format(config["eql_number_layers"]))
        file.write("optimizer: {}\n".format(config["optimizer"]))
        file.write("use_regularization_phase2: {}\n".format(config["use_regularization_phase2"]))
        file.write("number_trials: {}\n".format(config["number_trials"]))
        file.write("steps_ahead: {}\n".format(config["steps_ahead"]))
        file.write("phase2_from_file: {}\n".format(config["phase2_from_file"]))
        file.write("non_masked_weight_file: {}\n".format(config["non_masked_weight_file"]))
        
def create_summary_file(experiment_number):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "w") as file:
        file.write("")
    # file = open(filename, "w+")
            
def plot_train_vs_validation(experiment_number,num_epochs,train_loss,validation_loss,phase):
    x = range(num_epochs)
    plt.plot(x,train_loss,label="Training")
    plt.plot(x,validation_loss,label="Validation")
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.title(phase)
    plt.legend()
    filename = "ExperimentsSR/Experiment{}/train_vs_validation.jpg".format(experiment_number)
    plt.savefig(filename)
    plt.clf()
    
def plot_histogram(experiment_number,weights,phase,type_weights,a):
    plt.hist(weights)
    plt.title(phase+", "+type_weights+", a:{}".format(a))
    filename = "ExperimentsSR/Experiment{}/hist_{}.jpg".format(experiment_number,type_weights)
    plt.savefig(filename)
    plt.clf()
    
    
def append_text_to_summary(experiment_number,text):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "a+") as file:
        file.write(text)
        
def plot_descaled_real_vs_prediction(experiment_number,y_real,y_predicted,y_min,y_max):
    y_predicted_rescaled = y_predicted * (y_max - y_min) + y_min
    y_test_rescaled = y_real * (y_max - y_min) + y_min
    
    mse = mean_squared_error(y_predicted_rescaled,y_test_rescaled)
    mae = mean_absolute_error(y_predicted_rescaled,y_test_rescaled)
    
    plt.figure(figsize=(10,8))
    plt.plot(y_test_rescaled[0:2000],label="Real")
    plt.plot(y_predicted_rescaled[0:2000],label="Prediction from formula")
    plt.legend()
    plt.title("MAE: {:.2e}, MSE: {:.2e}".format(mae,mse))
    
    filename = "ExperimentsSR/Experiment{}/real_vs_prediction_phase2.jpg".format(experiment_number)
    plt.savefig(filename)
    
    plt.clf()
    
    


