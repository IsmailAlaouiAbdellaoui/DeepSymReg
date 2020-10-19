"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import pickle
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNet, MaskedSymbolicNet
from utils.regularization import l12_smooth
from inspect import signature
import time
import argparse
import sys
from scipy.io import loadmat

N_TRAIN = 256       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset - range from which we sample x
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
N_TEST = 100        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
# var_names = ["a", "b", "c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
# var_names = ["a", "b", "c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r"]

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.1
init_sd_middle = 0.5
init_sd_last = 1.0

# init_sd_first = 0.5
# init_sd_last = 0.5
# init_sd_middle = 0.5
# init_sd_first = 0.1
# init_sd_last = 0.1
# init_sd_middle = 0.1

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


def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)     # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * np.random.random([N, x_dim]) + range_min
    y = np.random.normal([[func(*x_i)] for x_i in x], NOISE_SD)
    return x, y

def generate_data_denmark(phase,batch_id,batch_size):
    filename = "step1.mat"
    # filename = "step2.mat"
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        x_temp = data["Xtr"]
        y_temp = data["Ytr"][:,0]
        
    elif phase == "test":
        x_temp = data["Xtr"]
        y_temp = data["Ytr"][:,0]
    else:
        return
        
    x = x_temp.reshape((x_temp.shape[0],80))
    x = x[batch_id*batch_size:(batch_id*batch_size)+batch_size]
    
    y = y_temp.reshape(y_temp.shape[0],1)
    y = y[batch_id*batch_size:(batch_id*batch_size)+batch_size]
    
    return x,y
        







class Benchmark:
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""
    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            # *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]

        self.n_layers = n_layers              # Number of hidden layers
        self.reg_weight = reg_weight     # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000    # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        # with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
        #     pickle.dump(result, f)

    def benchmark(self, func, func_name, trials):
        """Benchmark the EQL network on data generated by the given function. Print the results ordered by test error.

        Arguments:
            func: lambda function to generate dataset
            func_name: string that describes the function - this will be the directory name
            trials: number of trials to train from scratch. Will save the results for each trial.
        """

        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        # Create a new sub-directory just for the specific function
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Train network!
        expr_list, error_test_list = self.train(func, func_name, trials, func_dir)

        # Sort the results by test error (increasing) and print them to file
        # This allows us to easily count how many times it fit correctly.
        error_expr_sorted = sorted(zip(error_test_list, expr_list))     # List of (error, expr)
        error_test_sorted = [x for x, _ in error_expr_sorted]   # Separating out the errors
        expr_list_sorted = [x for _, x in error_expr_sorted]    # Separating out the expr

        fi = open(os.path.join(self.results_dir, 'eq_summary.txt'), 'a')
        fi.write("\n{}\n".format(func_name))
        for i in range(trials):
            fi.write("[%f]\t\t%s\n" % (error_test_sorted[i], str(expr_list_sorted[i])))
        fi.close()

    def train(self, func, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        # x, y = generate_data(func, N_TRAIN)
        # x,y = generate_data_denmark("train")
        
        
        # print(x.shape)
        # print(y.shape)
        # sys.exit()
        # x_val, y_val = generate_data(func, N_VAL)
        
        
        
        # x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])
        # x_test, y_test = generate_data_denmark("test")


        # Setting up the symbolic regression network
        
        # x_dim = len(signature(func).parameters)  # Number of input arguments to the function
        x_dim = 80
        
        x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
        y_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        y_placeholder_test = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
        sym = SymbolicNet(self.n_layers,
                          funcs=self.activation_funcs,
                          initial_weights=[tf.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
                                           tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                           tf.truncated_normal([width, 1], stddev=init_sd_last)])
        # sym = SymbolicNet(self.n_layers, funcs=self.activation_funcs)
        y_hat = sym(x_placeholder)
        # list_weights = [tf.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
        #                 tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
        #                 tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
        #                 tf.truncated_normal([width, 1], stddev=init_sd_last)]
        
        # print(len(list_weights))
        # sys.exit()

        # Label and errors
        
        error = tf.losses.mean_squared_error(labels=y_placeholder, predictions=y_hat)
        error_test = tf.losses.mean_squared_error(labels=y_placeholder_test, predictions=y_hat)
        
        
        reg_loss = l12_smooth(sym.get_weights())
        loss = error + self.reg_weight * reg_loss
        
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train = opt.minimize(loss)
        

        # Set up TensorFlow graph for training
        # learning_rate = tf.placeholder(tf.float32)
        # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        # train = opt.minimize(loss)

        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []  # Total loss (MSE + regularization)
        error_list = []     # MSE
        reg_list = []       # Regularization
        error_test_list = []    # Test error

        error_test_final = []
        eq_list = []
        # x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
        
        # y_hat = sym(x_placeholder)
        
        # Only take GPU memory as needed - allows multiple jobs on a single GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        batch_size = 200
        number_train_batches = 86761//batch_size #86761 : total size of Denmark dataset train
        number_test_batches = 9641//batch_size #9641 : total size of Denmark dataset test
        
        # learning_rate = tf.placeholder(tf.float32)
        # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        
        with tf.Session(config=config) as sess:
            # x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
            for trial in range(trials):
                print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(trials))

                loss_val = np.nan
                # Restart training if loss goes to NaN (which happens when gradients blow up)
                while np.isnan(loss_val):
                    # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                    sess.run(tf.global_variables_initializer())

                    t0 = time.time()
                    # First stage of training, preceded by 0th warmup stage
                    for i in range(self.n_epochs1 + 2000):
                    # for i in range(self.n_epochs1):
                        # batch_id = 0
                        if i < 2000:
                            lr_i = self.learning_rate * 10
                        else:
                            lr_i = self.learning_rate
                        for batch_train_id in range(2):
                            x_batch,y_batch = generate_data_denmark("train", batch_train_id, batch_size)
                            feed_dict = {x_placeholder: x_batch, y_placeholder: y_batch, learning_rate: lr_i}
                            if batch_train_id % 5 == 0:
                                loss_val, error_val, reg_val, = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                                print("aloss: {:.2f}, error: {:.2f}".format(loss_val,error_val))
                                if np.isnan(error_val):  # If loss goes to NaN, restart training
                                    print("Nan error, breaking main loop and restarting ...")
                                    break
                            else:
                                _ = sess.run(train, feed_dict=feed_dict)

                            
                        if i % self.summary_step == 0:
                            error_test_val_list = []
                            loss_val, error_val, reg_val, = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                            for batch_test_id in range(2):
                                x_test, y_test = generate_data_denmark("test",batch_test_id, batch_size)
                                # error_test = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat)
                                feed_dict_test = {x_placeholder: x_test, y_placeholder_test: y_test}
                                error_test_val = sess.run(error_test, feed_dict=feed_dict_test)
                                error_test_val_list.append(error_test_val)
                            avg_error_test = sum(error_test_val_list)/len(error_test_val_list)
                            # print("Epoch: {}\nTotal training loss: {:.2e}\nTest error: {:.2e}".format(i, loss_val, error_test_val))
                            print("Epoch: {}\nMost recent batch training loss: {:.2e}\nTest error: {:.2e}".format(i, loss_val, avg_error_test))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            reg_list.append(reg_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                print("Nan loss, breaking main loop and restarting ...")
                                break

                    t1 = time.time()

                    # Masked network - weights below a threshold are set to 0 and frozen. This is the fine-tuning stage
                    sym_masked = MaskedSymbolicNet(sess, sym)
                    y_hat_masked = sym_masked(x_placeholder)
                    # error_masked = tf.losses.mean_squared_error(labels=y, predictions=y_hat_masked)
                    # error_masked = tf.losses.mean_squared_error(labels=y_batch, predictions=y_hat_masked)
                    
                    # x_test, y_test = generate_data_denmark("test",0, batch_size)
                    # error_test_masked = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat_masked)
                    
                    # train_masked = opt.minimize(error_masked)

                    # 2nd stage of training
                    t2 = time.time()
                    print("starting phase2")
                    for i in range(self.n_epochs2):
                        for batch_train_id in range(2):
                            x_batch,y_batch = generate_data_denmark("train", batch_train_id, batch_size)
                            error_masked = tf.losses.mean_squared_error(labels=y_batch, predictions=y_hat_masked)
                            train_masked = opt.minimize(error_masked)
                            feed_dict = {x_placeholder: x_batch, learning_rate: self.learning_rate / 10}
                            _ = sess.run(train_masked, feed_dict=feed_dict)
                        if i % self.summary_step == 0:
                            error_test_val_list = []
                            loss_val, error_val = sess.run((loss, error_masked), feed_dict=feed_dict)
                            for batch_test_id in range(2):
                                x_test, y_test = generate_data_denmark("test",batch_test_id, batch_size)
                                error_test_masked = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat_masked)
                                error_test_val = sess.run(error_test_masked, feed_dict={x_placeholder: x_test})
                                error_test_val_list.append(error_test_val)
                            avg_error_test_val = sum(error_test_val_list)/len(error_test_val_list)
                            print("Epoch: {}\nMost recent batch training loss: {:.2e}\nTest error: {:.2e}".format(i, loss_val, avg_error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break
                    t3 = time.time()
                tot_time = t1-t0 + t3-t2
                print("total time:",tot_time)

                # Print the expressions
                weights = sess.run(sym_masked.get_weights())
                print(type(weights))
                sys.exit()
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                print("Formula from pretty print:",expr)

                # Save results
                trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)
                results = {
                    "weights": weights,
                    "loss_list": loss_list,
                    "error_list": error_list,
                    "reg_list": reg_list,
                    "error_test": error_test_list,
                    "expr": expr,
                    "runtime": tot_time
                }
                with open(trial_file, "wb+") as f:
                    pickle.dump(results, f)

                error_test_final.append(error_test_list[-1])
                eq_list.append(expr)

        return eq_list, error_test_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=1, help="Number of epochs to train the first stage")
    parser.add_argument("--n-epochs2", type=int, default=1,
                        help="Number of epochs to train the second stage, after freezing weights.")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json
    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    # bench.benchmark(lambda x: x, func_name="x", trials=1)
    # bench.benchmark(lambda x: x**2, func_name="x^2", trials=5)
    # bench.benchmark(lambda x: x**3, func_name="x^3", trials=5)
    # bench.benchmark(lambda x: np.sin(2*np.pi*x), func_name="sin(2pix)", trials=5)
    # bench.benchmark(lambda x: np.exp(x), func_name="e^x", trials=5)
    # bench.benchmark(lambda x, y: x*y, func_name="xy", trials=5)
    # bench.benchmark(lambda x, y: np.sin(2 * np.pi * x) + np.sin(4*np.pi * y),
    #                 func_name="sin(2pix)+sin(4py)", trials=5)
    # bench.benchmark(lambda x, y, z: 0.5*x*y + 0.5*z, func_name="0.5xy+0.5z", trials=5)
    # bench.benchmark(lambda x, y, z,a,b,c: 0.5*x*y + 0*z, func_name="0.5xy+0z", trials=5)
    # bench.benchmark(lambda a, b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z: a**2 + b*0+r**2+d*0+e*0+f*0, func_name="denmark_test1", trials=1)
    # bench.benchmark(lambda x, y, z, a, b, c: 0.5*x*y + 0*z + 0*a + 0*b+ 0*c, func_name="0.5xy+0z+0a+0b+0c", trials=5)
    bench.benchmark(lambda x, y, z: x**2 - y - 2*z, func_name="x^2+y-2z", trials=2)
    # bench.benchmark(lambda x: np.exp(-x**2), func_name="e^-x^2", trials=5)
    # bench.benchmark(lambda x: 1 / (1 + np.exp(-10*x)), func_name="sigmoid(10x)", trials=5)
    # bench.benchmark(lambda x, y: x**2 + np.sin(2*np.pi*y), func_name="x^2+sin(2piy)", trials=5)
    
    # ,s,t,u,v,w,x,y,z

    # 3-layer functions
    # bench.benchmark(lambda x, y, z: (x+y*z)**3, func_name="(x+yz)^3", trials=5)


