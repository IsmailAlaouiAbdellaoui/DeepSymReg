import os
import re

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
    return experiment_number
    


