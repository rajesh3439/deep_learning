from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))  # (D,H)
        b1 = np.zeros(hidden_dim)  # (H)
        W2 = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))  # (H, C)
        b2 = np.zeros(num_classes)  # (C)

        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1, W2, b2 = self.params.values()

        l1_out, l1_cache = affine_forward(X, W1, b1)
        l2_out, l2_cache = relu_forward(l1_out)
        scores, scores_cache = affine_forward(l2_out, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        # backward
        dout3, dW2, db2 = affine_backward(dloss, scores_cache)
        dout2 = relu_backward(dout3, l2_cache)
        _, dW1, db1 = affine_backward(dout2, l1_cache)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def loss_oldway(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        N = X.shape[0]
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        fc1 = X @ self.params["W1"] + self.params["b1"]  # (N, H)
        relu = np.maximum(0, fc1)
        scores = relu @ self.params["W2"] + self.params["b2"]  # (N, C)
        # numerically stable softmax compuation
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        counts = np.exp(shifted)
        # counts = np.exp(scores)
        probs = counts / np.sum(counts, axis=1, keepdims=True)
        L2 = (
            0.5
            * self.reg
            * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        )

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 1 / N * (np.sum(-np.log(probs[np.arange(N), y]))) + L2, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        """
        Gradients computation for the back prop starts from loss function. The final 2 
        layers are the Loss function (NLL) & Softmax. These two layers combined has
        an elegent short for expression for gradient w.r.t to Loss. This result is 
        more efficient lets use this.
        
        dL/dscores = probs - y

        # derivation (for a single example)
        S_i = exp(z_i)/sum_j(exp(z_j)) # z_i is the logit for class i
        L = -ln(S_k) # loss (k is the correct class)

        Using chain rule, Target =  dLoss with respect to class i
        dL/dz_i = dL/dS_k * dS_k/dz_i

        1) dl/dS_k = -1/S_k
        2) dS_k/dz_i : This derivative for S_k is derived as 2 parts
            Part A : i = k (i is the correct class) (u/v)' = u'v - v'u / v^2
                dS_k/dz_i = (exp(z_i)* sum(exp(z_j)) - exp(z_i)**2) / (sum(exp(z_j)))** 2 = S_i - S_i**2 = S_i(1-S_i)
            
            Part B : i <> k (i is not the correct class)
                dS_k/dz_i = (0 * sum(exp(z_j)) - exp(z_k)*exp(z_i))/(sum(exp(z_j)))** 2 = -S_k*S_i

        Combining derivatives: Multiply dL/dS_k * dS_k/dz_i
            Part A: (i=k)
                -1/S_k * S_i(1-S_i) = -(1-S_i) = S_i - 1 #(S_i = S_k since i=k)
            Part B: (i<>k)
                -1/S_k * -S_k*S_i = S_i

        This results can be viewd as Subtract 1 from scores for the correct class and 0 for incorrect classes
        If y is the one-hot vector representing the correct classes the above result simplifies to
        
        dL/dz_i = S_i - y
        """
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), y] = 1
        dscores = 1 / N * (probs - y_onehot)  # (N, C)
        # scores = relu @ W2 + b2
        dW2 = relu.T @ dscores  # (H, C)
        db2 = np.sum(dscores, axis=0)  # (C)
        drelu = dscores @ self.params["W2"].T  # (N,H)
        dfc1 = drelu * (
            fc1 > 0
        )  # (N,H) Max gate only allows gradients for the values of fc1 > 0
        dW1 = X.T @ dfc1  # (D, H)
        db1 = np.sum(dfc1, axis=0)  # (H)
        # W contributions from regulariztion
        dW1 += self.reg * self.params["W1"]
        dW2 += self.reg * self.params["W2"]

        grads = {"W1": dW1, "W2": dW2, "b1": db1, "b2": db2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        fan_in = input_dim
        # Initalize all hidden layers
        for i in range(len(hidden_dims)):
            W, b = f"W{i + 1}", f"b{i + 1}"
            beta, gamma = f"beta{i + 1}", f"gamma{i + 1}"
            fan_out = hidden_dims[i]
            self.params[W] = np.random.normal(0, weight_scale, (fan_in, fan_out))
            self.params[b] = np.random.normal(0, weight_scale, fan_out)
            if self.normalization:
                self.params[beta] = np.zeros(fan_out)
                self.params[gamma] = np.ones(fan_out)
            fan_in = fan_out

        # Initialize output layer
        W, b = f"W{self.num_layers}", f"b{self.num_layers}"
        self.params[W] = np.random.normal(0, weight_scale, (fan_in, num_classes))
        self.params[b] = np.random.normal(0, weight_scale, num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        layer_caches = []
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        out = X
        for i in range(self.num_layers - 1):
            caches = {}
            W, b = f"W{i + 1}", f"b{i + 1}"
            gamma, beta = f"gamma{i + 1}", f"beta{i + 1}"
            # print(f"{X.shape=};{self.params[W].shape=}")
            out, cache = affine_forward(out, self.params[W], self.params[b])
            caches["affine"] = cache
            if self.normalization == "batchnorm":
                out, cache = batchnorm_forward(
                    out, self.params[gamma], self.params[beta], self.bn_params[i]
                )
                caches["bn"] = cache
            elif self.normalization == "layernorm":
                out, cache = layernorm_forward(
                    out, self.params[gamma], self.params[beta], self.bn_params[i]
                )
                caches["ln"] = cache

            out, cache = relu_forward(out)  # relu
            caches["relu"] = cache
            if self.use_dropout:
                out, cache = dropout_forward(out, self.dropout_param)
                caches["dropout"] = cache

            layer_caches.append(caches)

        scores, cache = affine_forward(
            out, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]
        )
        layer_caches.append({"affine": cache})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)

        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # L2 regularizatio
        loss += (
            0.5
            * np.sum(
                [np.sum(self.params[f"W{i + 1}"] ** 2) for i in range(self.num_layers)]
            )
            * self.reg
        )

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for i in range(self.num_layers - 1, -1, -1):
            caches = layer_caches[i]
            # dropout
            if self.use_dropout and "dropout" in caches:
                cache = caches["dropout"]
                dout = dropout_backward(dout, cache)
            # relu
            if "relu" in caches:
                cache = caches["relu"]
                dout = relu_backward(dout, cache)
            # normalizations
            if self.normalization == "batchnorm" and "bn" in caches:
                cache = caches["bn"]
                dout, dgamma, dbeta = batchnorm_backward(dout, cache)
                grads[f"beta{i + 1}"], grads[f"gamma{i + 1}"] = dbeta, dgamma
            elif self.normalization == "layernorm" and "ln" in caches:
                cache = caches["ln"]
                dout, dgamma, dbeta = layernorm_backward(dout, cache)
                grads[f"beta{i + 1}"], grads[f"gamma{i + 1}"] = dbeta, dgamma
            # affine
            cache = caches["affine"]
            dout, dW, db = affine_backward(dout, cache)
            dW += self.reg * self.params[f"W{i + 1}"]  # add regularization effect
            grads[f"W{i + 1}"], grads[f"b{i + 1}"] = dW, db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
