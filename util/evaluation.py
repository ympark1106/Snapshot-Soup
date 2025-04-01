import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

import sklearn.metrics as sk

import numpy as np

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)
    
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        # correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        correct = len([x for x in filtered_tuples if np.array_equal(x[0], x[1])])

        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
    
def ECE(conf, pred, true, bin_size = 0.1):
    
    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)  
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
def OE(conf, pred, true, bin_size = 0.1):
    
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)  
        # print(acc, avg_conf, len_bin)
        if avg_conf > acc:
            ece += avg_conf * (avg_conf - acc) * len_bin / n 
        
    return ece

def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        if len_bin>5:
            cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)
     

def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15, is_spline = False):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    if is_spline:
        # print(probs[0].shape, probs[1].shape)
        preds = np.argmax(probs[0], axis=1)
        confs = probs[1]
    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
        
        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    # accuracy = accuracy_score(y_true, preds) * 100
    # error = 100 - accuracy
    # print(confs.shape, preds)
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    # mce = MCE(confs, preds, y_true, bin_size = 1/bins)

    oe = OE(confs, preds, y_true, bin_size = 1/bins)
    if is_spline:
        loss = log_loss(y_true=y_true, y_pred=probs[0])
        y_prob_true = np.array([probs[0][i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class

    else:
        loss = log_loss(y_true=y_true, y_pred=probs)
        y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class

    
    # print(y_prob_true.shape)
    # brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    
    if verbose:
        print("\n🔹 Model Calibration Metrics 🔹")
        # print("Accuracy:", accuracy)
        # print("Error:", error)
        print("ECE:", ece)
        # print("MCE:", mce)
        print("OE:", oe)
        print("Loss:", loss)
        # print("brier:", brier)
        # print(ece, mce, oe)
        
    
    # return (error, ece, mce, loss) # brier)
    
def calculate_ece(probs, y_true, normalize = False, bins = 15, is_spline = False):

    if is_spline:
        # print(probs[0].shape, probs[1].shape)
        preds = np.argmax(probs[0], axis=1)
        confs = probs[1]
    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
        
        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    # accuracy = accuracy_score(y_true, preds) * 100
    # error = 100 - accuracy
    # print(confs.shape, preds)
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    oe = OE(confs, preds, y_true, bin_size = 1/bins)
    
    return ece


def calculate_nll(outputs, targets):
    """
    Calculate the Negative Log-Likelihood (NLL) for the given outputs and targets.

    Args:
        outputs (numpy.ndarray): The predicted probabilities from the model (e.g., softmax outputs).
        targets (numpy.ndarray): The true labels.

    Returns:
        float: The calculated NLL.
    """
    # Convert numpy arrays to PyTorch tensors
    outputs_tensor = torch.tensor(outputs, requires_grad=True)
    targets_tensor = torch.tensor(targets, dtype=torch.long).view(-1)
    
    # Compute the negative log likelihood
    nll = F.nll_loss(torch.log(outputs_tensor), targets_tensor)
    
    # Return the NLL as a scalar
    return nll.item()
    