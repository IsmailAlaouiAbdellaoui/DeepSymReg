"""
Train a network to perform arithmetic operations on MNIST numbers.
EQL Network is used to back out the operation.
"""

# import tensorflow as tf
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.layers import Conv2D, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import BatchNormalization, Flatten
import sys

tf.disable_v2_behavior() 
import numpy as np
import os
from utils import functions, regularization, symbolic_network, pretty_print, helpers
import argparse

# mnist = tf.keras.datasets.mnist
#dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
BATCH_SIZE = 100    # BATCH_SIZE*2 needs to divide evenly into n_train
# n_train = 10000     # Size of MNIST training dataset


def get_synth_dataset():
    total_time_steps = 2000
    cities = 3
    features = 3
    lags = 2
    tensor_input = np.random.rand(total_time_steps//lags,features,cities,lags)
    
    output_data = np.zeros((total_time_steps//lags,1))
    for i in range(total_time_steps//lags):
        output_data[i]= tensor_input[i][0][0][0]**2+tensor_input[i][0][0][1]**2
        
    return tensor_input,output_data

x,y = get_synth_dataset()

def synthetic_batch_generator():
    
    batch_size = 10
    batch_indexes = range(0, x.shape[0], batch_size)
    while True:
        for i in batch_indexes:
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            yield x_batch,y_batch
            
            
        
    
    


# def batch_generator(batch_size=BATCH_SIZE):
#     return helpers.batch_generator((X_train, y_train), n_train, batch_size)


def normalize(y):
    return 9 * y + 9


# class Encoder:
#     """Network that takes in MNIST digit and produces a single-dimensional latent variable.
#     Consists of several convolution layers followed by fully-connected layers"""
#     def __init__(self, training):
#         """
#         Arguments:
#             training: Boolean of whether to use training mode or not. Matters for Batch norm layer
#         """
#         self.y_hat = None
#         self.training = training
#         self.bn = None

#     def build(self, x, n_latent=1, name='y_hat'):
#         """Convolutional and fully-connected layers to extract a latent variable from an MNIST image

#         Arguments:
#             x: Batch of MNIST images with dimension (n_batch, width, height)
#             n_latent: Dimension of latent variable
#             name: TensorFlow name of output
#         """
#         h = tf.expand_dims(x, axis=-1)  # Input to Conv2D needs dimension (n_batch, width, height, n_channels)
#         h = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
#         h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
#         h = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
#         h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
#         h = tf.keras.layers.Flatten()(h)
#         # h = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(h)
#         h = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(h)
#         h = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(h)
#         h = tf.keras.layers.Dense(units=n_latent, name=name)(h)
#         self.bn = tf.keras.layers.BatchNormalization()
#         # Divide by 2 to make std dev close to 0.5 because distribution is uniform
#         h = self.bn(h, training=self.training) / 2
#         y_hat = tf.identity(h, name=name)
#         return y_hat

#     def __call__(self, x, n_latent=1, name='y_hat'):
#         if self.y_hat is None:
#             self.y_hat = self.build(x, n_latent, name)
#         return self.y_hat

class ConvPlusLSTM:
    def __init__(self, training):
        """
        Arguments:
            training: Boolean of whether to use training mode or not. Matters for Batch norm layer
        """
        self.y_hat = None
        self.training = training
        self.bn = None
        
    
        
    def build(self, x, name='y_hat'):
            conv1_filters = 4
            conv2_filters = 4
            conv3_filters = 1
            lstm1_nodes = 50
            lstm2_nodes = 50
            dense1_nodes = 3*3*2
            number_cities = 2
            # filters = 5
            kernSize = 7
            
            # input1 = Input(shape = (features, cities,lags))
            # block1 = Conv2D(conv1_filters, (kernSize, kernSize), padding = 'same', activation='relu',name="conv1")(x)
            
            # self.bn = tf.keras.layers.BatchNormalization()
            # block1 = self.bn(block1,training=self.training)
            
            block1 = BatchNormalization()(x)
            block1 = Flatten()(block1)
            # block1 = Conv2D(conv2_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
            # block1 = BatchNormalization()(block1)
            # block1 = Conv2D(conv3_filters, (kernSize, kernSize), padding = 'same', activation='relu',name="conv3")(block1)
            # block1 = BatchNormalization()(block1)
            # block1 = tf.squeeze(block1,axis=-1)
            # block1 = LSTM(lstm1_nodes, return_sequences=True,name = "lstm1")(block1)
            # block1 = LSTM(lstm2_nodes, return_sequences=False,name = "lstm2")(block1)
            block1 = Dense(dense1_nodes, activation='linear')(block1)
            block1 = BatchNormalization()(block1)
            block1 = Dense(dense1_nodes, activation='linear')(block1)
            block1 = BatchNormalization()(block1)
            block1 = Dense(dense1_nodes, activation='linear')(block1)
            block1 = BatchNormalization()(block1)
            block1 = Dense(number_cities, activation='linear',name=name)(block1)
            # block1 = BatchNormalization()(block1)
            # y_hat = tf.identity(block1, name=name)
            return block1
            # return Model(inputs=input1, outputs=output1)
            
    def __call__(self, x, name='y_hat'):
        self.y_hat = self.build(x, name)
        return self.y_hat



def conv_plus_lstm(lags, features, cities, filters, kernSize):#model1
    conv1_filters = 80
    conv2_filters = 40
    conv3_filters = 1
    lstm1_nodes = 100
    lstm2_nodes = 100
    dense1_nodes = 100
    number_cities = 2
    
    input1 = Input(shape = (features, cities,lags))
    block1 = Conv2D(conv1_filters, (kernSize, kernSize), padding = 'same', activation='relu')(input1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(conv2_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(conv3_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = tf.squeeze(block1,axis=-1)
    block2 = LSTM(lstm1_nodes, return_sequences=True,name = "lstm1")(block1)
    block2 = LSTM(lstm2_nodes, return_sequences=False,name = "lstm2")(block2)
    block3 = Dense(dense1_nodes, activation='relu')(block2)
    output1 = Dense(number_cities, activation='linear')(block3)
    return Model(inputs=input1, outputs=output1)

class SymbolicDigit:
    """Architecture for MNIST arithmetic. Takes care of initialization, training, and saving."""
    def __init__(self, sr_net, x=None, encoder=None, normalize=None):
        """Set up the MNIST arithmetic architecture

        Arguments:
            sr_net: EQL Network, SymbolicNet instance
        """
        if normalize is not None:
            print("normalize used !")
        n_digits = 2    # Number of inputs to the arithmetic function.
        features, cities, lags = 3,3,2
        if x is None:
            x = tf.placeholder(tf.float32, [None, features, cities,lags], name='input')
            # x2 = tf.placeholder(tf.float32, [None, 28, 28], name='x2')
            # x = [x1, x2]
        self.x = x

        # Encoder for each MNIST digit into latent variable (conv layers with batch norm at output)
        # if encoder is None:
        self.training = tf.placeholder_with_default(True, [])
        #     encoder = Encoder(self.training)
        # else:
        #     self.training = encoder.training
        # self.encoder = encoder

        # We want to feed multiple digits into the same CNN, so we flatten the input first and then reshape the output
        # x_full = tf.stack(self.x)    # shape = (n_digits, batch_size, 28, 28)
        self.training = tf.placeholder_with_default(True, [])
        batch_size = tf.shape(self.x)[0]
        # x_flat = tf.reshape(x_full, [n_digits*batch_size, 28, 28])  # Flatten to (n_digits*batch_size, 28, 28)
        
        filters = 5
        kernSize = 7
        # model = conv_plus_lstm(lags, features, cities, filters, kernSize)
        self.model = ConvPlusLSTM(True)
        z_flat = self.model(self.x)     # shape = (n_digits*batch_size, 1)
        z = tf.reshape(z_flat, [n_digits, batch_size])
        self.z = tf.unstack(z, axis=0, name='z')  # List of size n_digits. This gets saved.
        z = tf.transpose(z)     # reshape to shape = (batch_size, n_digits)
        
        self.y_hat = tf.squeeze(sr_net(z))#Output from EQL
        if normalize is not None:
            self.y_hat = normalize(self.y_hat)

        self.y_ = tf.placeholder(tf.float32, [None],name="true_output")  # Placeholder for true labels
        self.lr = tf.placeholder(tf.float32)  # Learning rate of gradient descent
        self.loss = None
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.trainer = None

        self.reg = tf.constant(0.0)
        self.loss_total = None

        correct_prediction = tf.equal(tf.round(self.y_hat), tf.round(self.y_))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.error_avg = tf.reduce_mean(tf.abs(self.y_hat - self.y_))

    def set_training(self, reg=0):
        """Set up the remainder of the Tensorflow graph for training. Call set_reg before this. This must be called
        before training the network."""
        self.loss = tf.losses.mean_squared_error(self.y_, self.y_hat)
        self.reg = reg
        self.loss_total = self.loss + reg

        self.trainer = self.optimizer.minimize(self.loss_total)
        # self.trainer = tf.group([self.trainer, self.model.bn.updates])

    def save_result(self, sess, results_dir, eq, result_str):
        # Because tf will not save the model if the directory exists, we append a number to the directory and save. This
        # allows us to run multiple trials
        dir_format = os.path.join(results_dir, "trial%d")
        i = 0
        while True:
            if os.path.isdir(dir_format % i):
                i += 1
            else:
                break
        os.makedirs("results/mnist/test/trial{}".format(i))
            
        

        # Save TensorFlow graph
        # input_dict = {"x%d" % i: self.x[i] for i in range(len(self.x))}
        # if isinstance(self.encoder, Encoder):
        #     input_dict['training'] = self.encoder.training
        # output_dict = {"z%d" % i: tf.identity(self.z[i], name='z%d' % i) for i in range(len(self.z))}
        # output_dict["y_hat"] = self.y_hat
        # tf.saved_model.simple_save(sess, dir_format % i, inputs=input_dict, outputs=output_dict)

        # Save equation and test accuracy inside the project directory
        file = open(os.path.join(dir_format, 'equation.txt') % i, 'w+')
        file.write(str(eq))
        file.write("\n")
        file.write(result_str)
        file.close()

        # Save equation and test accuracy in the higher-level directory
        file = open(os.path.join(results_dir, 'overview.txt'), 'a+')
        file.write('%d: \t%s\n' % (i, eq))
        file.write(result_str)
        file.write("\n")
        file.close()

    def train(self, sess, n_epochs, batch, func, epoch=None, lr_val=1e-3, train_fun=None):
        """Training step of the digit extractor + symbolic network"""

        if epoch is None:
            epoch = tf.placeholder_with_default(0.0, [])    # dummy variable

        loss_i = None
        for i in range(n_epochs):
            # print("epoch {}".format(i))
            batch_x1, batch_y = next(batch)
            
            batch_y = np.squeeze(batch_y,axis=1)
            # print(batch_y.shape)
            # sys.exit()
            # batch_x2, batch_y2 = next(batch)
            # batch_y = func(batch_y1, batch_y2)

            # Filtering out the batch. This lets us train on a subset of data (e.g. y<15) and then test
            # on the rest of the data (e.g. y>=15) to evaluate extrapolation
            # if train_fun is not None:
            #     ind_train = train_fun(batch_y)  # Indices for data matching the condition
            #     batch_x1 = batch_x1[ind_train]
            #     batch_x2 = batch_x2[ind_train]
            #     batch_y = batch_y[ind_train]

            # if i % 10 == 0:
            #     train_accuracy, loss_i, reg_i = \
            #         sess.run((self.accuracy, self.loss_total, self.reg),
            #                   feed_dict={self.x: batch_x1, self.y_: batch_y,
            #                             epoch: i, self.training: False})
            
            if i % 1000 == 0:
                loss_i, reg_i = \
                    sess.run((self.loss_total, self.reg),
                              feed_dict={self.x: batch_x1, self.y_: batch_y,
                                        epoch: i})
                print(loss_i)
                    
                    
                    
                print("Step %d\t Fit loss %.3f\tReg loss %.3f" %
                      (i, loss_i-reg_i, reg_i))

                if np.isnan(loss_i):  # If loss goes to NaN, restart training
                    print("loss going to nan, restarting the training ...")
                    break
            # sys.exit()
            # print(type(self.x))
            # print(type(self.y_))
            # print(type(self.lr))
            # print(type(epoch))
            # print(type(self.model.training))
            # sys.exit()
            sess.run(self.trainer, feed_dict={self.x: batch_x1, self.y_: batch_y,
                                  self.lr: lr_val, epoch: i})
        return loss_i

    # def calc_accuracy(self, X, y, func, sess, filter_fun=None):
    #     """Calculate accuracy over a given dataset"""
    #     # Grab test data, split it into two halves, and then apply function to y-values to create new dataset
    #     n_test = y.shape[0]
    #     n2 = int(n_test / 2)
    #     X1 = X[:n2, :, :]
    #     X2 = X[n2:, :, :]
    #     y1 = y[:n2]
    #     y2 = y[n2:]
    #     y_test = func(y1, y2)

    #     # To calculate test accuracy, we split it up into batches to avoid overflowing the memory
    #     acc_batch = []
    #     error_batch = []
    #     batch_test_ind = range(0, n2, BATCH_SIZE)
    #     for i_batch in batch_test_ind:
    #         X_batch1 = X1[i_batch:i_batch + BATCH_SIZE]
    #         X_batch2 = X2[i_batch:i_batch + BATCH_SIZE]
    #         Y_batch = y_test[i_batch:i_batch + BATCH_SIZE]

    #         # Only pick out data (X1, X2, y) that match a condition on y.
    #         if filter_fun is not None:
    #             ind_train = filter_fun(Y_batch)
    #             X_batch1 = X_batch1[ind_train]
    #             X_batch2 = X_batch2[ind_train]
    #             Y_batch = Y_batch[ind_train]

    #         acc_i, error_i = sess.run((self.accuracy, self.error_avg),
    #                                   feed_dict={self.x[0]: X_batch1, self.x[1]: X_batch2, self.y_: Y_batch,
    #                                              self.training: False})
    #         acc_batch.append(acc_i)
    #         error_batch.append(error_i)
    #     acc_total = np.mean(acc_batch)
    #     error_total = np.mean(error_batch)

    #     return acc_total, error_total

    @staticmethod
    def normalize(y):
        return y * 9 + 9


class SymbolicDigitMasked(SymbolicDigit):
    def __init__(self, sym_digit_net, sr_net_masked, normalize=None):
        super().__init__(sr_net_masked, x=sym_digit_net.x, normalize=normalize)
        self.sym_digit_net = sym_digit_net

        self.y_ = sym_digit_net.y_  # Placeholder for true labels
        self.lr = sym_digit_net.lr  # Learning rate of gradient descent

    def set_training(self, reg=0):
        """Set up the remainder of the Tensorflow graph for training. Call set_reg before this. This must be called
        before training the network."""
        self.loss = tf.losses.mean_squared_error(self.y_, self.y_hat)
        self.loss_total = self.loss + self.reg + reg

        self.trainer = self.sym_digit_net.optimizer.minimize(self.loss_total)
        # self.trainer = tf.group([self.trainer, self.model.bn.updates])

        # correct_prediction = tf.equal(tf.round(self.y_hat), tf.round(self.y_))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.error_avg = tf.reduce_mean(tf.abs(self.y_hat - self.y_))
        
    def train_masked(self,sess, n_epochs, batch, func,lr):
        super().train(sess, n_epochs, batch, func,lr_val=lr)


def train_add(func=lambda a, b: a + b, results_dir=None, reg_weight=5e-2, learning_rate=1e-2, n_epochs=10001):
    
    """Addition of two MNIST digits with a symbolic regression network."""
    tf.reset_default_graph()

    # Symbolic regression network to combine the conv net outputs
    PRIMITIVE_FUNCS = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        # *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product()] * 2,
    ]
    sr_net = symbolic_network.SymbolicNet(symbolic_depth=3, funcs=PRIMITIVE_FUNCS, init_stddev=0.1)  # Symbolic regression network
    # Overall architecture
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=normalize)
    # Set up regularization term and training
    penalty = regularization.l12_smooth(sr_net.get_weights())
    penalty = reg_weight * penalty
    sym_digit_network.set_training(reg=penalty)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    
    sess = tf.Session(config=config)

    # batch = batch_generator(batch_size=100)
    batch = synthetic_batch_generator()

    # Train, and restart training if loss goes to NaN
    loss_i = np.nan
    while np.isnan(loss_i):
        # with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # sys.exit()
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate)
        print(type(loss_i))
        print(loss_i)
        # sys.exit()
        
        if np.isnan(loss_i):
            continue

        # Freezing weights
        sr_net = symbolic_network.MaskedSymbolicNet(sess, sr_net, threshold=0.002)
        masked_sym_digit_network = SymbolicDigitMasked(sym_digit_network, sr_net, normalize=normalize)
        
        masked_sym_digit_network.set_training()

        # Training with frozen weights. Regularization is 0
        # sys.exit()
        # for k,v in sess.__dict__.items():
        #     print(k)
        #     print(v)
        #     print("\n\n")
        # sys.exit()
        # sess.run(tf.global_variables_initializer())
        # for v in tf.get_default_graph().as_graph_def().node:
        #     print(v.name)
        # with sess:
        # loss_i = masked_sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10)

    # Print out human-readable equation (with regularization)
    weights = sess.run(sr_net.get_weights())
    # expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["z1", "z2"])
    # expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["l1", "l2","l3","l4","l5","l6"])
    expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["x", "y"])
    print("type weights",type(weights))
    # print("weights",weights)
    # expr = normalize(expr)
    print("Unnormalized expression:",expr)
    # print("normalized expression:",expr)

    # Calculate accuracy on test dataset
    # acc_test, error_test = sym_digit_network.calc_accuracy(X_test, y_test, func, sess)
    # result_str = 'Test accuracy: %g\n' % acc_test
    # print(result_str)
    print("need to compute mse")

    # sym_digit_network.save_result(sess, results_dir, expr, result_str)
    # sym_digit_network.save_result(sess, results_dir, expr, "")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on MNIST arithmetic task.")
    parser.add_argument("--results-dir", type=str, default='results/mnist/test')
    parser.add_argument("--reg-weight", type=float, default=5e-2, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Base learning rate for training')
    parser.add_argument("--n-epochs", type=int, default=100000, help="Number of epochs to train in each stage")
    parser.add_argument('--trials', type=int, default=1, help="Number of trials to train.")
    parser.add_argument('--l0', action='store_true', help="Use relaxed L0 regularization instead of L0.5")
    parser.add_argument('--filter', action='store_true', help="Train only on y<15 data.")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json

    meta.write(json.dumps(kwargs))
    meta.close()

    trials = kwargs['trials']
    use_l0 = kwargs['l0']
    use_filter = kwargs['filter']
    del kwargs['trials']
    del kwargs['l0']
    del kwargs['filter']

    # if use_l0:
    #     for _ in range(trials):
    #         train_add_l0(**kwargs)
    # elif use_filter:
    #     for _ in range(trials):
    #         train_add_test(**kwargs)
    # else:
    for _ in range(trials):
        train_add(**kwargs)

# if __name__ == '__main__':
#     # batch = batch_generator(batch_size=100)
#     # temp = next(batch)
#     # print((temp[0].shape))
#     total_time_steps = 1000
#     batch_size = 10
#     batch_indexes = range(0, total_time_steps, batch_size)
#     # for item in batch_indexes:
#     #     print(item)
    
