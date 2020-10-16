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







policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

def rescale(scaled,min_value,max_value):
    return scaled * (max_value - min_value) + min_value

def save_weights(weights):
    with h5py.File("phase1_weights.hdf5", "w") as f:
        for i in range(len(weights)):
            f.create_dataset('dataset{}'.format(i), data=weights[i].numpy())
        
def load_weights(filename):
    weights = []
    with h5py.File(filename, "r") as f:
        for i in range(3):
            weights.append(f.get('dataset{}'.format(i))[()])
    return weights

# @tf.function
def generate_batch_data_denmark(phase,batch_id,batch_size):
    filename = "step1.mat"
    # filename = "step2.mat"
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        x_temp = data["Xtr"]
        y_temp = data["Ytr"][:,0]
        
    elif phase == "test":
        x_temp = data["Xtest"]
        y_temp = data["Ytest"][:,0]
    else:
        return
        
    x = x_temp.reshape((x_temp.shape[0],80))
    x = x[batch_id*batch_size:(batch_id*batch_size)+batch_size].astype('float32')
    
    y = y_temp.reshape(y_temp.shape[0],1)
    y = y[batch_id*batch_size:(batch_id*batch_size)+batch_size].astype('float32')
    
    return x,y

# @tf.function
def generate_all_data(phase):
    filename = "step1.mat"
    # filename = "step2.mat"
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        x_temp = data["Xtr"]
        y_temp = data["Ytr"][:,0]
        
    elif phase == "test":
        x_temp = data["Xtest"]
        y_temp = data["Ytest"][:,0]
    else:
        return
        
    x = x_temp.reshape((x_temp.shape[0],80)).astype('float32')
    
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
def custom_loss(model, x, y_real):
    y_predicted = model(x)
    reg_weight = 5e-3
    error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_real, y_predicted)#double check
    reg_loss = l12_smooth(model.get_weights())
    loss = error + reg_weight * reg_loss
    return loss

# @tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = custom_loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.get_weights())
    


# for k,v in model.__dict__.items():
#     print(k)

# loss_value, grads = grad(model, x, y)

# train_op.apply_gradients(zip(grads, model.get_weights()))

# y_hat = sym(x)
# print(y_hat.shape)
# @tf.function


batch_size = 200
# number_train_batches = 86761//batch_size #86761 : total size of Denmark dataset train
first_phase_lr = 1e-4
second_phase_lr = first_phase_lr/10
def train_non_masked(epochs_first_phase):
    train_op = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    # num_epochs = 50
    # number_batches = 86761//batch_size
    train_loss_results = []
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
                loss_value, grads = grad(model, x, y)
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
        # count_val = 1
        with tqdm(val_dataset,position = 0, leave = True,colour="#99b433") as bar_val:
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
                bar_val.set_description("Val Epoch {}".format(epoch+1))
                bar_val.set_postfix(od_val)
            if epoch_loss_avg_val.result() < best_val_loss:
                print("\n*** Validation loss decreased from {:.2e} to {:.2e}".format(best_val_loss,epoch_loss_avg_val.result()))
                print("Saving new model")
                best_val_loss = epoch_loss_avg_val.result()
                best_val_model = model
                best_val_weights = model.get_weights()
                
        print("\n\n"+"-"*6+" End of Epoch "+"-"*6+"\n\n")
        
    save_weights(best_val_weights)
                
    return best_val_model,best_val_weights

def train_masked(best_val_weights,epochs_second_phase,threshold=0.01):
    weights = best_val_weights
    masked_weights = []
    masks = []
    for w_i in weights:
        mask = tf.cast(tf.constant(tf.abs(w_i) > threshold),tf.float32)
        masks.append(mask)
        masked_weights.append(tf.multiply(w_i, mask))
        
    masked_model = SymbolicNet(n_layers,
                              funcs=activation_funcs,
                              initial_weights=masked_weights)
    
    
    train_op = tf.keras.optimizers.RMSprop(learning_rate=second_phase_lr)
    # num_epochs = 50
    # number_batches = 86761//batch_size
    train_loss_results = []
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
                loss_value, grads = grad(masked_model, x, y)
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
        # count_val = 1
        with tqdm(val_dataset,position = 0, leave = True,colour="#cc3333") as bar_val:
            for x_val,y_val in bar_val:
                sleep(0.05)
                y_prediction_val = masked_model(x_val)
                loss_value_val, grads = grad(masked_model, x_val, y_val)
                epoch_loss_avg_val.update_state(y_val,y_prediction_val)
                if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val.result()):
                    sys.exit("Nan value, stopping")
                
                od_val = OrderedDict() 
                od_val["loss_val"] = f'{loss_value_val:.2f}'
                od_val["mse_val"] = f'{epoch_loss_avg_val.result():.2e}'
                bar_val.set_description("Val Epoch {}".format(epoch+1))
                bar_val.set_postfix(od_val)
            if epoch_loss_avg_val.result() < best_val_loss:
                print("\n*** Validation loss decreased from {:.2e} to {:.2e}".format(best_val_loss,epoch_loss_avg_val.result()))
                print("Saving new masked model")
                best_val_loss = epoch_loss_avg_val.result()
                best_val_model = masked_model
                best_val_weights = masked_model.get_weights()
                
        print("\n\n"+"-"*6+" End of Epoch "+"-"*6+"\n\n")
        
    
        
    final_weights_list = []
    for item in best_val_weights:
        final_weights_list.append(item.numpy())
        
    expr = pretty_print.network(final_weights_list, activation_funcs, var_names[:x_dim])
    print("Formula from pretty print:",expr)
                
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
        
    
    
    
    
    
  
# test_save_load()
# threshold=0.01
epochs_first_phase = 100
epochs_second_phase = 10

# import numpy.ma as ma

weights = load_weights("phase1_weights_5.15e-3.hdf5")

# temp = weights[2]
# temp2 = (abs(temp)>0.01)
# print(temp2)
final_weights_list = []
for item in weights:
    mask = (abs(item)>0.01)
    final_weights_list.append(np.multiply(mask,item))
    
    
# import sympy
expr = pretty_print.network(final_weights_list, activation_funcs, var_names[:x_dim])
print("Formula from pretty print:",expr)
print(str(expr))
from pytexit import py2tex
latex = py2tex(str(expr))
# latex_version = sympy.latex(eval(str(expr)))
# temp2 = np.where(temp>0.01)
# print(temp2)
# mask = ma.masked_where(temp < 0.01, temp).set_fill_value()
# print(mask)
# from matplotlib import pyplot as plt 
# plt.hist(temp,density=True) 
# plt.title("Weights between input and first layer") 
# plt.show()

# new_model =  SymbolicNet(n_layers,
#                               funcs=activation_funcs,
#                               initial_weights=weights)






# weights = model.get_weights()

# save_weights(weights)

# best_val_model,best_val_weights = train_non_masked(epochs_first_phase)
print()
print("-"*30)
print("-"*6+" Start of phase 2 "+"-"*6)
print("-"*30)
print()
# best_masked_model,best_masked_weights =  train_masked(best_val_weights,epochs_second_phase)

            
            
        
#strategy = tf.distribute.MirroredStrategy()
#print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# print(tf.config.list_physical_devices('GPU'))
# train_non_masked()
# import numpy as np
# a = np.array([1,2,4]).astype('float32')
# b = np.array([2,5,6]).astype('float32')
# temp = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(a, b)
# print(temp)
# a = np.array([[1, 2, 3], [4, 5, 6]])
# c = tf.constant(a)
# threshold = 1
# weights = model.get_weights()
# masked_weights = []
# for w_i in weights:
#     mask = tf.cast(tf.constant(tf.abs(w_i) > threshold),tf.float32)
#     masked_weights.append(tf.multiply(w_i, mask))
    
# masked_model = SymbolicNet(n_layers,
#                           funcs=activation_funcs,
#                           initial_weights=masked_weights)

# weights2 = model.get_weights()

# weights_list = []
# for item in weights2:
#     weights_list.append(item.numpy())


# expr = pretty_print.network(weights_list, activation_funcs, var_names[:x_dim])
# print("Formula from pretty print:",expr)

    
# filename = "step1.mat"
# # filename = "step2.mat"
# data = loadmat('Denmark_data/{}'.format(filename))
# print(data.keys())



