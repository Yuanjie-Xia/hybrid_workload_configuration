import time

import numpy as np
from deepPerf.mlp_plain_model import MLPPlainModel
from deepPerf.mlp_sparse_model import MLPSparseModel

def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    """
    Args:
        X_train1: train input data (2/3 of the whole training data)
        Y_train1: train output data (2/3 of the whole training data)
        X_train2: validate input data (1/3 of the whole training data)
        Y_train2: validate output data (1/3 of the whole training data)
        n_layer: number of layers of the neural network
        lambd: regularized parameter

    """
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0

    dir_output = 'C:/Users/Downloads/'

    # Build and train model
    model = MLPSparseModel(config, dir_output)
    model.build_train()
    model.train(X_train1, Y_train1, lr_initial)

    # Evaluate trained model on validation data
    Y_pred_val = model.predict(X_train2)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))

    return abs_error, rel_error


def deepPerf(whole_data_df, exp_num):
    whole_data = whole_data_df.values
    (N, n) = whole_data.shape
    n = n - 1

    X_all = whole_data[:, 0:n]
    Y_all = whole_data[:, n][:, np.newaxis]

    # Some variables to store results
    result_sys = []
    len_count = 0

    train_size = X_all.shape[1]
    sample_size_all = [i * train_size for i in range(1, 7)]

    # Sample sizes need to be investigated
    for idx in range(len(sample_size_all)):

        N_train = sample_size_all[idx]
        print("Sample size: {}".format(N_train))

        rel_error_mean = []
        lambda_all = []
        error_min_all = []
        rel_error_min_all = []
        training_index_all = []
        n_layer_all = []
        lr_all = []
        abs_error_layer_lr_all = []
        time_all = []
        for m in range(1, 10):
            print("Experiment: {}".format(m))

            # Start measure time
            start = time.time()

            # Set seed and generate training data
            seed = 42 + m
            np.random.seed(seed)
            permutation = np.random.permutation(N)
            training_index = permutation[0:N_train]
            training_data = whole_data[training_index, :]
            X_train = training_data[:, 0:n]
            Y_train = training_data[:, n][:, np.newaxis]

            # Scale X_train and Y_train
            max_X = np.amax(X_train, axis=0)
            if 0 in max_X:
                max_X[max_X == 0] = 1
            X_train = np.divide(X_train, max_X)
            max_Y = np.max(Y_train)/100
            if max_Y == 0:
                max_Y = 1
            Y_train = np.divide(Y_train, max_Y)

            # Split train data into 2 parts (67-33)
            N_cross = int(np.ceil(N_train*2/3))
            X_train1 = X_train[0:N_cross, :]
            Y_train1 = Y_train[0:N_cross]
            X_train2 = X_train[N_cross:N_train, :]
            Y_train2 = Y_train[N_cross:N_train]

            # Choosing the right number of hidden layers and , start with 2
            # The best layer is when adding more layer and the testing error
            # does not increase anymore
            print('Tuning hyperparameters for the neural network ...')
            print('Step 1: Tuning the number of layers and the learning rate ...')
            config = dict()
            config['num_input'] = n
            config['num_neuron'] = 128
            config['lambda'] = 'NA'
            config['decay'] = 'NA'
            config['verbose'] = 0
            dir_output = 'C:/Users/Downloads'
            abs_error_all = np.zeros((15, 4))
            abs_error_all_train = np.zeros((15, 4))
            abs_error_layer_lr = np.zeros((15, 2))
            abs_err_layer_lr_min = 100
            count = 0
            layer_range = range(2, 15)
            lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
            for n_layer in layer_range:
                config['num_layer'] = n_layer
                for lr_index, lr_initial in enumerate(lr_range):
                    model = MLPPlainModel(config, dir_output)
                    model.build_train()
                    model.train(X_train1, Y_train1, lr_initial)

                    Y_pred_train = model.predict(X_train1)
                    abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                    Y_pred_val = model.predict(X_train2)
                    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                    abs_error_all[int(n_layer), lr_index] = abs_error

                # Pick the learning rate that has the smallest train cost
                # Save testing abs_error correspond to the chosen learning_rate
                temp = abs_error_all_train[int(n_layer), :]/np.max(abs_error_all_train)
                temp_idx = np.where(abs(temp) < 0.0001)[0]
                if len(temp_idx) > 0:
                    lr_best = lr_range[np.max(temp_idx)]
                    err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                else:
                    lr_best = lr_range[np.argmin(temp)]
                    err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                abs_error_layer_lr[int(n_layer), 0] = err_val_best
                abs_error_layer_lr[int(n_layer), 1] = lr_best

                if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                    abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                         np.argmin(temp)]
                    count = 0
                else:
                    count += 1

                if count >= 2:
                    break
            abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

            # Get the optimal number of layers
            n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])]+5

            # Find the optimal learning rate of the specific layer
            config['num_layer'] = n_layer_opt
            for lr_index, lr_initial in enumerate(lr_range):
                model = MLPPlainModel(config, dir_output)
                model.build_train()
                model.train(X_train1, Y_train1, lr_initial)

                Y_pred_train = model.predict(X_train1)
                abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                Y_pred_val = model.predict(X_train2)
                abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                abs_error_all[int(n_layer), lr_index] = abs_error

            temp = abs_error_all_train[int(n_layer), :]/np.max(abs_error_all_train)
            temp_idx = np.where(abs(temp) < 0.0001)[0]
            if len(temp_idx) > 0:
                lr_best = lr_range[np.max(temp_idx)]
            else:
                lr_best = lr_range[np.argmin(temp)]

            lr_opt = lr_best
            print('The optimal number of layers: {}'.format(n_layer_opt))
            print('The optimal learning rate: {:.4f}'.format(lr_opt))

            # Use grid search to find the right value of lambda
            lambda_range = np.logspace(-2, np.log10(1000), 30)
            error_min = np.zeros((1, len(lambda_range)))
            rel_error_min = np.zeros((1, len(lambda_range)))
            decay = 'NA'
            for idx, lambd in enumerate(lambda_range):
                val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                                       X_train2, Y_train2,
                                                       n_layer_opt, lambd, lr_opt)
                error_min[0, idx] = val_abserror
                rel_error_min[0, idx] = val_relerror

            # Find the value of lambda that minimize error_min
            lambda_f = lambda_range[np.argmin(error_min)]
            print('Step 2: Tuning the l1 regularized hyperparameter ...')
            print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

            # Store some useful results
            n_layer_all.append(n_layer_opt)
            lr_all.append(lr_opt)
            abs_error_layer_lr_all.append(abs_error_layer_lr)
            lambda_all.append(lambda_f)
            error_min_all.append(error_min)
            rel_error_min_all.append(rel_error_min)
            training_index_all.append(training_index)

            # Solve the final NN with the chosen lambda_f on the training data
            config = dict()
            config['num_neuron'] = 128
            config['num_input'] = n
            config['num_layer'] = n_layer_opt
            config['lambda'] = lambda_f
            config['verbose'] = 1
            dir_output = 'C:/Users/Downloads'
            model = MLPSparseModel(config, dir_output)
            model.build_train()
            model.train(X_train, Y_train, lr_opt)

            # End measuring time
            end = time.time()
            time_search_train = end-start
            print('Time cost (seconds): {:.2f}'.format(time_search_train))
            time_all.append(time_search_train)

            # Testing with non-training data (whole data - the training data)
            testing_index = np.setdiff1d(np.array(range(N)), training_index)
            testing_data = whole_data[testing_index, :]
            X_test = testing_data[:, 0:n]
            X_test = np.divide(X_test, max_X)
            Y_test = testing_data[:, n][:, np.newaxis]

            Y_pred_test = model.predict(X_test)
            Y_pred_test = max_Y*Y_pred_test
            rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel())))
            rel_error_mean.append(np.mean(rel_error)*100)
            print('Prediction relative error (%): {:.2f}'.format(np.mean(rel_error)*100))

        result = dict()
        result["N_train"] = N_train
        result["lambda_all"] = lambda_all
        result["n_layer_all"] = n_layer_all
        result["lr_all"] = lr_all
        result["abs_error_layer_lr_all"] = abs_error_layer_lr_all
        result["rel_error_mean"] = rel_error_mean
        result["error_min_all"] = error_min_all
        result["rel_error_min_all"] = rel_error_min_all
        result["training_index"] = training_index_all
        result["time_search_train"] = time_all
        result_sys.append(result)

        # Compute some statistics: mean, confidence interval
        result = []
        for i in range(len(result_sys)):
            temp = result_sys[i]
            sd_error_temp = np.sqrt(np.var(temp['rel_error_mean'], ddof=1))
            ci_temp = 1.96*sd_error_temp/np.sqrt(len(temp['rel_error_mean']))

            result_exp = [temp['N_train'], np.mean(temp['rel_error_mean']),
                          ci_temp]
            result.append(result_exp)

        result_arr = np.asarray(result)

        print('Finish experimenting for system with sample size {}.'.format(N_train))

        print('Mean prediction relative error (%) is: {:.2f}, Margin (%) is: {:.2f}'.format(np.mean(rel_error_mean), ci_temp))

        # Save the result statistics to a csv file after each sample
        # Save the raw results to an .npy file
        print('Save results to the current directory ...')

        filename = 'result_' + str(exp_num) + '.csv'
        np.savetxt(filename, result_arr, fmt="%f", delimiter=",",
                   header="Sample size, Mean, Margin")
        print('Save the statistics to file ' + filename + ' ...')

        filename = 'result_AutoML_veryrandom' + str(exp_num) + '.npy'
        np.save(filename, result_sys)
        print('Save the raw results to file ' + filename + ' ...')