import tensorflow as tf
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNet
from scipy.io import loadmat
from utils.regularization import l12_smooth
import os 
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm
from collections import OrderedDict
from time import sleep
import sys
import numpy as np
import h5py
from pytexit import py2tex
from matplotlib import pyplot as plt 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from options import config
import new_utils as nutils
import time


filename = "scale1.mat"
data = loadmat('Denmark_data/{}'.format(filename))
y_min = data["y_min_tr"][0][0]
y_max = data["y_max_tr"][0][0]


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

def rescale(scaled,min_value,max_value):
    return scaled * (max_value - min_value) + min_value

def save_weights(weights,experiment_number,phase,best_val_loss):
    if phase == "phase1":
        filename_output = "ExperimentsSR/Experiment{}/nt_val_weights_{:.2e}.hdf5".format(experiment_number,best_val_loss)
    elif phase == "phase2":
        filename_output = "ExperimentsSR/Experiment{}/phase2_val_weights_{:.2e}.hdf5".format(experiment_number,best_val_loss)
    with h5py.File(filename_output, "w") as f:
        for i in range(len(weights)):
            f.create_dataset('dataset{}'.format(i), data=weights[i].numpy())
        
def load_weights(filename):
    weights = []
    with h5py.File(filename, "r") as f:
        for i in range(3):
            weights.append(f.get('dataset{}'.format(i))[()])
    return weights

# @tf.function
# def generate_batch_data_denmark(phase,batch_id,batch_size):
#     filename = "step1.mat"
#     # filename = "step2.mat"
#     data = loadmat('Denmark_data/{}'.format(filename))
#     if phase == "train":
#         x_temp = data["Xtr"]
#         y_temp = data["Ytr"][:,0]
        
#     elif phase == "test":
#         x_temp = data["Xtest"]
#         y_temp = data["Ytest"][:,0]
#     else:
#         return
        
#     x = x_temp.reshape((x_temp.shape[0],80))
#     x = x[batch_id*batch_size:(batch_id*batch_size)+batch_size].astype('float32')
    
#     y = y_temp.reshape(y_temp.shape[0],1)
#     y = y[batch_id*batch_size:(batch_id*batch_size)+batch_size].astype('float32')
    
#     return x,y


def matrix_to_tensor(matrix):
    len_dates = matrix.shape[0]
    number_cities = 18 #change
    number_features= 18 #change
    new_tensor = np.ones((number_features,number_cities,len_dates))      
    for i in range(len_dates):
        for j in range(number_cities):
            features = matrix[i,j*number_features:(j+1)*number_features]
            new_tensor[:,j,i] = features
    return new_tensor

def tensor_to_matrix(tensor): 
    number_features = tensor.shape[0]
    number_cities = tensor.shape[1]
    len_dates = tensor.shape[2]
    
    matrix_for_scaling = np.ones((len_dates,number_cities*number_features))
    for i in range(number_cities):
        for j in range(len_dates):
            features_city = tensor[:,i,j]
            matrix_for_scaling[j,i*number_features:(i+1)*number_features] = features_city        
    return matrix_for_scaling

def get_x_right_format(phase):
    filename = "step1.mat"    
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        x = data["Xtr"]
    elif phase == "test":
        x = data["Xtest"] 
    x = np.transpose(x,(0,2,1,3))
    all_features_all_cities = x.shape[1] * x.shape[2]
    x_output = np.zeros((x.shape[0],80))
    for i in range(x.shape[0]):
        temp_matrix = tensor_to_matrix(x[i])
        for j in range(temp_matrix.shape[0]):#iterate through lags of matrix
            x_output[i,j*all_features_all_cities:(j+1)*all_features_all_cities] = temp_matrix[j]
    return x_output


def plot_output_test_data():
    filename = "step1.mat"
    data = loadmat('Denmark_data/{}'.format(filename))
    y_temp = data["Ytest"][0:2000,0]
    y = y_temp.reshape(y_temp.shape[0],1).astype('float32')
    plt.plot(y)

# @tf.function
def generate_all_data(phase):
    filename = "step1.mat"
    # filename = "step2.mat"
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        # x_temp = data["Xtr"]
        y_temp = data["Ytr"][:,0]
        
    elif phase == "test":
        # x_temp = data["Xtest"]
        y_temp = data["Ytest"][:,0]
    else:
        return
        
    x = get_x_right_format(phase).astype("float32")
    # x = x_temp.reshape((x_temp.shape[0],80)).astype('float32')
    
    y = y_temp.reshape(y_temp.shape[0],1).astype('float32')
    
    return x,y

def generate_variable_list(alphabet_size,num_variables):
    lower_case = []
    for i in range(26):
        lower_case.append(chr(i+97))
    
    variables = lower_case[:alphabet_size]
    prefix_index = 0
    letter_index = 0
    prefix = variables[0]
    for i in range(num_variables):
        if ((i+1)) % alphabet_size == 0:   
            variables.append(prefix+lower_case[letter_index])
            prefix_index += 1
            prefix = variables[prefix_index]
            letter_index =0
            
        else:
            variables.append(prefix+lower_case[letter_index])
            letter_index +=1
            
    return variables
    
alphabet_size = 26
num_variables = 80
var_names = generate_variable_list(alphabet_size,num_variables)

dict_vars = {}
for i in range(len(var_names)):
    dict_vars[var_names[i]] = i

x_dim = 80
init_sd_first = 0.1
init_sd_middle = 0.5
init_sd_last = 1.0

batch_size = 200

# x,y = generate_data_denmark("train",0,batch_size)
activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            # *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]
n_layers = 2
n_double = functions.count_double(activation_funcs)
width = len(activation_funcs)
model = SymbolicNet(n_layers,
                          funcs=activation_funcs,
                          initial_weights=[tf.random.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
                                           tf.random.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.random.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.random.truncated_normal([width, 1], stddev=init_sd_last)])
# @tf.function
def custom_loss(model, x, y_real, use_rescaled,reg_weight,a):
    y_predicted = model(x)
    # reg_weight = 5e-3
    # reg_weight = 5e-2
    if use_rescaled:
        y_predicted_rescaled = y_predicted * (y_max - y_min) + y_min
        y_real_rescaled = y_real * (y_max - y_min) + y_min
        error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_real_rescaled, y_predicted_rescaled)#double check
    else:
        error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_real, y_predicted)
    reg_loss = l12_smooth(model.get_weights(),a)
    loss = error + reg_weight * reg_loss
    return loss

def grad(model, inputs, targets,use_rescaled_loss,reg_weight,a):
    with tf.GradientTape() as tape:
        loss_value = custom_loss(model, inputs, targets,use_rescaled_loss,reg_weight,a)
    return loss_value, tape.gradient(loss_value, model.get_weights())
    


def train_non_masked(epochs_first_phase,train_op,use_rescaled_loss,reg_weight,a,experiment_number):
    train_loss_results = []
    valid_loss_results = []
    count_loss_stagnation = 0
    x_train,y_train = generate_all_data("train")
    x_test,y_test = generate_all_data("test")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    
    best_val_loss = float('inf')
    best_val_model = model
    best_val_weights = None
    
    
    for epoch in range(epochs_first_phase):
        epoch_loss_avg = tf.keras.metrics.MeanSquaredError()

        with tqdm(train_dataset,position = 0, leave = True,colour="#2d89ef") as bar_train:
            for x,y in bar_train:               
                bar_train.set_description("Epoch {}".format(epoch+1))
                y_prediction = model(x)
                loss_value, grads = grad(model, x, y,use_rescaled_loss,reg_weight,a)
                train_op.apply_gradients(zip(grads, model.get_weights()))
                # mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y, y_prediction)
                epoch_loss_avg.update_state(y,y_prediction)  # Add current batch loss
                od = OrderedDict() 
                od["loss"] = f'{loss_value:.2f}'
                od["mse"] = f'{epoch_loss_avg.result():.2e}'
                bar_train.set_postfix(od)
                bar_train.update()
                if np.isnan(loss_value) or np.isnan(epoch_loss_avg.result()):
                    sys.exit("Nan value, stopping")
                sleep(0.05)

        train_loss_results.append(epoch_loss_avg.result())

        print("\n\nValidation\n")
        epoch_loss_avg_val = tf.keras.metrics.MeanSquaredError()
        with tqdm(val_dataset,position = 0, leave = True,colour="#99b433") as bar_val:
            for x_val,y_val in bar_val:
                sleep(0.05)
                y_prediction_val = model(x_val)
                loss_value_val, grads = grad(model, x_val, y_val,use_rescaled_loss,reg_weight,a)
                epoch_loss_avg_val.update_state(y_val,y_prediction_val)
                if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val.result()):
                    sys.exit("Nan value, stopping")
                
                od_val = OrderedDict() 
                od_val["loss_val"] = f'{loss_value_val:.2f}'
                od_val["mse_val"] = f'{epoch_loss_avg_val.result():.2e}'
                bar_val.set_description("Val Epoch {}".format(epoch+1))
                bar_val.set_postfix(od_val)
            if epoch_loss_avg_val.result() < best_val_loss:
                print("\n==> Validation loss decreased from {:.2e} to {:.2e}".format(best_val_loss,epoch_loss_avg_val.result()))
                print("==> Saving new model")
                best_val_loss = epoch_loss_avg_val.result()
                best_val_model = model
                best_val_weights = model.get_weights()
                count_loss_stagnation = 0
            else:
                count_loss_stagnation += 1
        
        valid_loss_results.append(epoch_loss_avg_val.result())     
        print("\n\n"+"-"*6+" End of Epoch "+"-"*6+"\n\n")
        if count_loss_stagnation == 4 and epoch > 70:
            print("No improvement over more than 4 epochs, and more than 70 epochs passed\nExiting phase 1...")
            break
        
        
    # filename_output = "nt_val_weights_{:.2e}.hdf5".format(best_val_loss)
    save_weights(best_val_weights,experiment_number,"phase1",best_val_loss)
    
    nutils.append_text_to_summary(experiment_number,"phase 1 best MSE validation: {}\n".format(best_val_loss))
    
    nutils.plot_train_vs_validation(experiment_number, epochs_first_phase, train_loss_results, valid_loss_results,"phase1")
    
    sparsity1 = get_sparsity_percent(best_val_weights[0].numpy())
    sparsity2 = get_sparsity_percent(best_val_weights[1].numpy())
    sparsity3 = get_sparsity_percent(best_val_weights[2].numpy())
    
    nutils.append_text_to_summary(experiment_number,"sparsity1 phase1: {}\n".format(sparsity1))
    nutils.append_text_to_summary(experiment_number,"sparsity2 phase1: {}\n".format(sparsity2))
    nutils.append_text_to_summary(experiment_number,"sparsity3 phase1: {}\n".format(sparsity3))
    
    
    nutils.plot_histogram(experiment_number,best_val_weights[0].numpy(),"phase1","weights1",a)
    nutils.plot_histogram(experiment_number,best_val_weights[1].numpy(),"phase1","weights2",a)
    nutils.plot_histogram(experiment_number,best_val_weights[2].numpy(),"phase1","weights3",a)
    
    return best_val_model,best_val_weights
















# def train_masked(best_val_weights,epochs_second_phase,use_rescaled_loss,threshold=0.01,from_file = False,filename = None):
def train_masked(best_val_weights,epochs_second_phase,**kwargs):
    # train_loss_results = []
    
    if kwargs["from_file"]:
        weights = load_weights(filename)
        masked_weights = []
        for item in weights:
            # mask = (abs(item)>threshold)
            mask = tf.cast(tf.constant(tf.abs(item) > kwargs["threshold"]),tf.float32)
            masked_weights.append(np.multiply(mask,item))
    else:
        weights = best_val_weights
        masked_weights = []
        # masks = []
        for w_i in weights:
            mask = tf.cast(tf.constant(tf.abs(w_i) > kwargs["threshold"]),tf.float32)
            # masks.append(mask)
            masked_weights.append(tf.multiply(w_i, mask))
            
    sparsity1 = get_sparsity_percent(masked_weights[0].numpy())
    sparsity2 = get_sparsity_percent(masked_weights[1].numpy())
    sparsity3 = get_sparsity_percent(masked_weights[2].numpy())
    
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity1 phase2 after masking: {}\n".format(sparsity1))
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity2 phase2 after masking: {}\n".format(sparsity2))
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity3 phase2 after masking: {}\n".format(sparsity3))
        
    masked_model = SymbolicNet(n_layers,
                              funcs=activation_funcs,
                              initial_weights=masked_weights)
    
    
    train_op = tf.keras.optimizers.RMSprop(learning_rate=kwargs["phase2_lr"])
    train_loss_results = []
    valid_loss_results = []
    x_train,y_train = generate_all_data("train")
    x_test,y_test = generate_all_data("test")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    
    best_val_loss = float('inf')
    best_val_model = masked_model
    best_val_weights = None
    
    
    for epoch in range(epochs_second_phase):
        epoch_loss_avg = tf.keras.metrics.MeanSquaredError()

        with tqdm(train_dataset,position = 0, leave = True,colour="#cc9933") as bar_train:
            for x,y in bar_train:               
                bar_train.set_description("Epoch {}".format(epoch+1))
                y_prediction = masked_model(x)
                loss_value, grads = grad(masked_model, x, y,kwargs["use_rescaled_loss"],kwargs["reg_weight"],kwargs["a"])
                train_op.apply_gradients(zip(grads, masked_model.get_weights()))
                # mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y, y_prediction)
                epoch_loss_avg.update_state(y,y_prediction)  # Add current batch loss
                od = OrderedDict() 
                od["loss"] = f'{loss_value:.2f}'
                od["mse"] = f'{epoch_loss_avg.result():.2e}'
                bar_train.set_postfix(od)
                bar_train.update()
                if np.isnan(loss_value) or np.isnan(epoch_loss_avg.result()):
                    sys.exit("Nan value, stopping")
                sleep(0.05)

        train_loss_results.append(epoch_loss_avg.result())

        print("\n\nValidation\n")
        epoch_loss_avg_val = tf.keras.metrics.MeanSquaredError()
        with tqdm(val_dataset,position = 0, leave = True,colour="#cc3333") as bar_val:
            for x_val,y_val in bar_val:
                sleep(0.05)
                y_prediction_val = masked_model(x_val)
                loss_value_val, grads = grad(masked_model, x_val, y_val,kwargs["use_rescaled_loss"],kwargs["reg_weight"],kwargs["a"])
                epoch_loss_avg_val.update_state(y_val,y_prediction_val)
                if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val.result()):
                    sys.exit("Nan value, stopping")
                
                od_val = OrderedDict() 
                od_val["loss_val"] = f'{loss_value_val:.2f}'
                od_val["mse_val"] = f'{epoch_loss_avg_val.result():.2e}'
                bar_val.set_description("Val Epoch {}".format(epoch+1))
                bar_val.set_postfix(od_val)
            if epoch_loss_avg_val.result() < best_val_loss:
                print("\n==> Validation loss decreased from {:.2e} to {:.2e}".format(best_val_loss,epoch_loss_avg_val.result()))
                print("==> Saving new masked model")
                best_val_loss = epoch_loss_avg_val.result()
                best_val_model = masked_model
                best_val_weights = masked_model.get_weights()
        
        valid_loss_results.append(epoch_loss_avg_val.result())     
        print("\n\n"+"-"*6+" End of Epoch "+"-"*6+"\n\n")
        
        
    nutils.append_text_to_summary(kwargs["experiment_number"],"phase 2 best MSE validation: {}\n".format(best_val_loss))
    
        
    final_weights_list = []
    for item in best_val_weights:
        final_weights_list.append(item.numpy())
        
    expr = pretty_print.network(final_weights_list, activation_funcs, var_names[:x_dim])
    print("Formula from pretty print:",expr)
    nutils.append_text_to_summary(kwargs["experiment_number"],"Formula after phase2: {}\n".format(expr))
    try:
        latex = get_latex_equation_from_string(str(expr))
        clean_latex = latex.replace(r"\right","").replace(r"\left","")
        nutils.append_text_to_summary(kwargs["experiment_number"],"Latex formula after phase2: {}\n".format(clean_latex))
    except Exception as e:
        print("\n\nCouldn't transform the string into latex format")
        print("=>Reason: ",e)
    
    
    # filename_output = "phase2_val_weights_{:.2e}.hdf5".format(best_val_loss)
    # save_weights(best_val_weights,filename_output)
    save_weights(best_val_weights,experiment_number,"phase2",best_val_loss)
    
    nutils.plot_train_vs_validation(kwargs["experiment_number"], epochs_second_phase, train_loss_results, valid_loss_results,"phase2")
              
    y_predicted = best_val_model(x_test)
    nutils.plot_descaled_real_vs_prediction(kwargs["experiment_number"],y_test,y_predicted,y_min,y_max)
    
    return best_val_model,best_val_weights
                
      
def test_save_load():
    model = SymbolicNet(n_layers,
                          funcs=activation_funcs,
                          initial_weights=[tf.random.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
                                           tf.random.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.random.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.random.truncated_normal([width, 1], stddev=init_sd_last)]) 
    x_test,y_test = generate_all_data("test")
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    epoch_loss_avg_val = tf.keras.metrics.MeanSquaredError()
    with tqdm(val_dataset,position = 0, leave = True,colour="#cc3333") as bar_val:
        for x_val,y_val in bar_val:
            sleep(0.05)
            y_prediction_val = model(x_val)
            loss_value_val, grads = grad(model, x_val, y_val)
            epoch_loss_avg_val.update_state(y_val,y_prediction_val)
            if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val.result()):
                sys.exit("Nan value, stopping")
            
            od_val = OrderedDict() 
            od_val["loss_val"] = f'{loss_value_val:.2f}'
            od_val["mse_val"] = f'{epoch_loss_avg_val.result():.2e}'
            bar_val.set_postfix(od_val)
        print("MSE: {}".format(epoch_loss_avg_val.result()))
    
    save_weights(model.get_weights())
    loaded_weights = load_weights("phase1_weights.hdf5")
    
    loaded_model = SymbolicNet(n_layers,
                              funcs=activation_funcs,
                              initial_weights=loaded_weights)
    
    epoch_loss_avg_val2 = tf.keras.metrics.MeanSquaredError()
    with tqdm(val_dataset,position = 0, leave = True,colour="#cc3333") as bar_val:
        for x_val,y_val in bar_val:
            sleep(0.05)
            y_prediction_val = loaded_model(x_val)
            loss_value_val, grads = grad(loaded_model, x_val, y_val)
            epoch_loss_avg_val2.update_state(y_val,y_prediction_val)
            if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val2.result()):
                sys.exit("Nan value, stopping")
            
            od_val = OrderedDict() 
            od_val["loss_val"] = f'{loss_value_val:.2f}'
            od_val["mse_val"] = f'{epoch_loss_avg_val2.result():.2e}'
            bar_val.set_postfix(od_val)
        print("MSE: {}".format(epoch_loss_avg_val2.result()))
        
    assert epoch_loss_avg_val2.result() == epoch_loss_avg_val.result(), sys.exit("Not the same weights")
        


def get_equation_from_training_weights(filename,threshold = 0.01,need_threshold = False):
    weights = load_weights(filename)
    if need_threshold:
        final_weights_list = []
        for item in weights:
            mask = (abs(item)>threshold)
            final_weights_list.append(np.multiply(mask,item))
    else:
        final_weights_list = weights
        
    expr = pretty_print.network(final_weights_list, activation_funcs, var_names[:x_dim])
    print("Formula from pretty print:",expr)
    
    latex = py2tex(str(expr))
    return latex
    # print(latex)
    
def get_latex_equation_from_string(string):
    return py2tex(string)
    
def plot_predict_from_saved_weights(filename,threshold = 0.01,need_threshold = False):
    weights = load_weights(filename)
    if need_threshold:
        final_weights_list = []
        for item in weights:
            mask = (abs(item)>threshold)
            final_weights_list.append(np.multiply(mask,item))
    
        model = SymbolicNet(n_layers,funcs=activation_funcs,initial_weights=final_weights_list) 
    else:
        model = SymbolicNet(n_layers,funcs=activation_funcs,initial_weights=weights) 
    x_test,y_test = generate_all_data("test")
    print(x_test.shape)
    y_predicted = model(x_test)
    # filename = "scale1.mat"
    # data = loadmat('Denmark_data/{}'.format(filename))
    # y_min = data["y_min_tr"][0][0]
    # y_max = data["y_max_tr"][0][0]
    # print(y_min,y_max)
    mse = mean_squared_error(y_predicted,y_test)
    print("unscaled mse:{:.2e}".format(mse))
    
    y_predicted_rescaled = y_predicted * (y_max - y_min) + y_min
    y_test_rescaled = y_test * (y_max - y_min) + y_min
    rescaled_MSE = mean_squared_error(y_predicted_rescaled,y_test_rescaled)
    rescaled_MAE = mean_absolute_error(y_predicted_rescaled,y_test_rescaled)
    print("rescaled mse:{:.2e}".format(rescaled_MSE))
    print("rescaled mae:{:.2e}".format(rescaled_MAE))
    
    plt.figure(figsize=(10,8))
    plt.plot(y_test_rescaled[0:2000],label="Real")
    plt.plot(y_predicted_rescaled[0:2000],label="Prediction from formula")
    plt.legend()
    


def lag_city_feature_from_var(target_var):
    print("target var:",target_var)
    index_target = dict_vars[target_var]
    print("index target var:",index_target)
    rest1 = index_target%20
    index_lag = index_target//20
    index_city = rest1//4
    feature_index = (index_target)- index_lag * 20 - index_city * 4
    print("index_lag:{}, index_city: {}, feature_index:{}".format(index_lag,index_city,feature_index))
    print("lag number :{}, city number: {}, feature number:{}".format(index_lag+1,index_city+1,feature_index+1))
    return (index_lag,index_city,feature_index)

def get_sparsity_percent(array):
    num_zeros = np.count_nonzero(array==0)
    num_elements = array.size
    return num_zeros/num_elements*100

if __name__ == "__main__":
    use_rescaled_loss = config["use_rescaled_MSE"]
    number_epochs1 = config["epochs1"]
    number_epochs2 = config["epochs2"]
    threshold_value = config["threshold_value"]
    a_L_05 = config["a_L_0.5"]
    use_phase2 = config["use_phase2"]
    use_threshold_before_phase2 = config["use_thresholding_before_phase2"]
    lambda_reg = config["lambda_reg"]
    batch_size = config["batch_size"]
    first_phase_lr = config["phase1_lr"]
    second_phase_lr = config["phase2_lr"]
    eql_number_layers = config["eql_number_layers"]
    optimizer = config["optimizer"]
    use_regularization_phase2 = config["optimizer"]
    number_trials = config["number_trials"]
    
    if optimizer == "rmsprop":
        phase1_optimizer = tf.keras.optimizers.RMSprop(learning_rate=first_phase_lr)
        phase2_optimizer = tf.keras.optimizers.RMSprop(learning_rate=second_phase_lr)
    elif optimizer == "adam":
        phase1_optimizer = tf.keras.optimizers.Adam(learning_rate=first_phase_lr)
        phase2_optimizer = tf.keras.optimizers.Adam(learning_rate=second_phase_lr)
    else:
        sys.exit("Invalid optimizer, exiting ...")
    
    #5 cities , 4 features, 4 lags
    # a_list=[5e-2,5e-3,5e-4]
    # for a_exp in a_list:
    #     config["a_L_0.5"] = a_exp
    #     a_L_05 = a_exp
    experiment_number = nutils.get_folders_started()
    nutils.record_base_info(experiment_number,**config)
    start = time.time()
    best_val_model,best_val_weights = train_non_masked(number_epochs1,phase1_optimizer,use_rescaled_loss,lambda_reg,a_L_05,experiment_number)
    end = time.time()
    span = end - start
    print("Time span of first phase: {}".format(span))
    if use_phase2:
        dict_phase2 = {
            "threshold":threshold_value,
            "use_threshold":use_threshold_before_phase2,
            "use_rescaled_loss":use_rescaled_loss,
            "reg_weight":lambda_reg,
            "a":a_L_05,
            "phase2_lr":second_phase_lr,
            "experiment_number":experiment_number,
            "from_file":False
            }
        print()
        print("-"*30)
        print("-"*6+" Start of phase 2 "+"-"*6)
        print("-"*30)
        print()
    #     # best_masked_model,best_masked_weights =  train_masked(best_val_weights=None,\
    #     #                                                       epochs_second_phase=epochs_second_phase,\
    #     #                                                           from_file=True,filename="nt_val_weights_5.49e-03.hdf5")
        start = time.time()
        m_model,m_weights =  train_masked(best_val_weights,number_epochs2,**dict_phase2)
        end = time.time()
        span = end - start
        print("Time span of second phase: {}".format(span))
    print("\n\n"+"*"*10+" End of Experiment {} ".format(experiment_number)+"*"*10+"\n\n")
        


        
    

    
    
    
    
    


