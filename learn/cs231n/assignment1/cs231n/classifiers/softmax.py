from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        # forward pass
        scores_raw = X[i].dot(W)
        # compute the probabilities in numerically stable way
        scores = scores_raw - np.max(scores_raw)
        p_raw = np.exp(scores)
        p = p_raw/p_raw.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss

        # backward pass
        # loss -= logp[y[i]]  # negative log probability is the loss
        # loss = loss + (-1)*logp[y[i]]
        dlogp = np.zeros_like(logp)
        dlogp[y[i]] = -1
        # logp = np.log(p)
        dp = 1/p * dlogp
        # p = p_raw/p_raw.sum()  # normalize
        dp_raw = (1 / p_raw.sum()) * (dp - np.sum(p * dp))
        # p_raw = np.exp(scores)
        dscores = p_raw * dp_raw
        # scores = scores_raw - np.max(scores_raw)
        max_idx = np.argmax(scores_raw)
        dscores_raw = dscores.copy()
        dscores_raw[max_idx] -= np.sum(dscores)
        # scores_raw = X[i].dot(W)
        dW += np.outer(X[i], dscores_raw)
        

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W # this step is result of taking gradient of Loss with regularization w.r.t W 

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    scores = np.dot(X, W) # (N, C)
    scores_norm = scores - np.max(scores, axis=1, keepdims=True)
    scores_exp = np.exp(scores_norm)
    probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    logits = np.log(probs[np.arange(N), y])
    loss = np.sum(-logits)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # STEP loss = np.sum(-logits)
    dlogits = -np.ones_like(logits) #(N ,)
    # STEP logits = np.log(probs[np.arange(N), y])
    dprobs = np.zeros_like(probs)
    dprobs[np.arange(N), y] = 1 / probs[np.arange(N), y] * dlogits # (N, C)
    # STEP probs = scores_exp/ np.sum(scores_exp, axis=1, keepdims=True)
    v = np.sum(scores_exp, axis=1, keepdims=True)
    row_dot = np.sum(dprobs * probs, axis=1, keepdims=True)       # (N, 1)
    dscores_exp = (dprobs - row_dot) / v                          # (N, C)
    # dprobs_sum = np.sum(dprobs, axis=1, keepdims=True)
    # dscores_exp = (dprobs - probs * dprobs_sum) / v
    # STEP scores_exp = np.exp(scores_norm)
    dscores_norm = scores_exp * dscores_exp #(N,C)
    # STEP scores_norm = scores - np.max(scores, axis=1, keepdims=True)
    max_scores = np.zeros_like(scores)
    max_scores[np.arange(N), np.argmax(scores, axis = 1)] = 1
    dscores = dscores_norm  - max_scores * np.sum(dscores_norm, axis=1, keepdims=True) #(N, C)
    # STEP scores = np.dot(X, W) # (N, C) , W = (D, C), X = (N, D)
    dW = np.dot(X.T, dscores)

    # adding impact of effect of regularization
    loss = loss / N + reg * np.sum(W*W)
    dW = dW / N +  2 * reg * W


    return loss, dW
