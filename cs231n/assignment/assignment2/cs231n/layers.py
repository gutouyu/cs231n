from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #pass
    N = x.shape[0]
    out = x.reshape((N,-1)).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    #pass
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape((N,-1)).T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #pass
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    #pass
    dx = dout
    dx[x<0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        #pass
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        xmu = x - sample_mean
        sq = xmu ** 2
        var = np.mean(sq, axis=0)
        sqrtvar = np.sqrt(var + eps)
        ivar = 1.0 / sqrtvar
        xhat = xmu * ivar
        out = xhat * gamma + beta
        cache = (x, xhat,xmu, gamma, beta, ivar, sqrtvar, var, sq, sample_mean, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        #pass
        xmu = x - running_mean
        sq = xmu ** 2
        var = np.mean(sq, axis=0)
        sqrtvar = np.sqrt(var + eps)
        ivar = 1.0 / sqrtvar
        xhat = xmu * ivar
        out = xhat * gamma + beta
        cache = (x, xhat,xmu, gamma, beta, ivar, sqrtvar, var, sq, running_mean, eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    #pass
#    x, normal_out, gamma, beta, mean,var, eps = cache
    x, xhat,xmu, gamma, beta, ivar, sqrtvar, var, sq, sample_mean, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)

    dgamma = np.sum(xhat * dout, axis=0)
    dxhat = gamma * dout

    divar = np.sum(xmu * dxhat, axis=0)
    dxmu = ivar * dxhat

    dsqrtvar = -1.0 / (sqrtvar ** 2) * divar
    dvar =  0.5 * (var + eps) ** (-0.5) * dsqrtvar # (D,)

    dsq = 1.0 / N  * np.ones((N,D)) * dvar # (N,D)

    dxmu += 2 * xmu * dsq

    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)
    dx += 1.0 / N * np.ones((N,D)) * dmu

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        #pass
        mask = np.random.rand(*x.shape) < p
        out = x * mask / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        #pass
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        #pass
        dx = dout * mask / dropout_param['p']
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #pass
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    xpad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=(0,))
    Hout = 1 + (H + 2 * pad - HH) // stride
    Wout = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, Hout, Wout), dtype=x.dtype)
    for inputIdx in xrange(N):
        for kernelIdx in xrange(F):
            for row in xrange(Hout):
                for col in xrange(Wout):
                    row_start, row_end = row*stride, row*stride+HH
                    col_start, col_end = col*stride, col*stride+WW
                    x_window = xpad[inputIdx,:,row_start:row_end,col_start:col_end]
                    w_window = w[kernelIdx,...]
                    out[inputIdx][kernelIdx][row][col] = np.sum(x_window * w_window) + b[kernelIdx]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #pass
    x,w,b, conv_param = cache
    pad, stride = conv_param['pad'], conv_param['stride']
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape
    _,_,Hout,Wout = dout.shape

    dx = np.zeros_like(x) # (N, C, H, W)
    dw = np.zeros_like(w) # (F, C, HH, WW)
    db = np.zeros_like(b) # (F,)

    # dout (N, F, Hout, Wout)

    xpad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=(0,))
    dxpad = np.zeros_like(xpad)
    for inputIdx in xrange(N):
        for filterIdx in xrange(F):
            # backprop into b
            db[filterIdx] += np.sum(dout[inputIdx, filterIdx, ...])
            for dep in xrange(C):

                # Backprop into x
                # 1.Rotate kernel 180
                kernel = w[filterIdx, dep, ...]
                kernel = np.flip(kernel, 0)
                kernel = np.flip(kernel, 1)
                # 2.Cross-correlation( 180(w), dout_with_stride_pad )
                for row in xrange(H+2*pad):
                    for col in xrange(W+2*pad):
                        kernel_h, kernel_w = kernel.shape
                        # 2.1 Need to fillup stride with zeros, so dout is the same size as stride=1
                        dout_one = dout[inputIdx, filterIdx]
                        h_after_stride = (dout_one.shape[0]-1)*stride + 1
                        w_after_stride = (dout_one.shape[1]-1)*stride + 1
                        doutstride = np.zeros((h_after_stride, w_after_stride))
                        for i in xrange(dout_one.shape[0]):
                            for j in xrange(dout_one.shape[1]):
                                doutstride[stride*i, stride*j] = dout_one[i,j]

                        # 2.2 Add Pad with size (kernel-1). We must add stride first, then pad.
                        doutstridepad = np.pad(doutstride, ((kernel_h-1,kernel_h-1), (kernel_w-1,kernel_w-1)), 'constant', constant_values=(0,))

                        # 2.3 Calculate dot
                        dot = kernel * doutstridepad[row:row+kernel_h, col:col+kernel_w]
                        dxpad[inputIdx, dep, row, col] += np.sum(dot)
                dx[inputIdx,dep,...] = dxpad[inputIdx, dep, pad:-pad, pad:-pad] 

                # Backprop into w
                # Cross-correlation(x_with_pad, dout_with_stride), No rotate
                for row in xrange(HH):
                    for col in xrange(WW):

                        # 1. Fillup dout as like stride == 1
                        dout_one = dout[inputIdx, filterIdx]
                        h_after_stride = (dout_one.shape[0]-1)*stride + 1
                        w_after_stride = (dout_one.shape[1]-1)*stride + 1
                        doutstride = np.zeros((h_after_stride, w_after_stride))
                        for i in xrange(dout_one.shape[0]):
                            for j in xrange(dout_one.shape[1]):
                                doutstride[stride*i, stride*j] = dout_one[i,j]
                        # 2. Get x_with_pad
                        dout_h, dout_w = doutstride.shape
                        x_win = xpad[inputIdx, dep, row:row+dout_h, col:col+dout_w]

                        # 3. Calculate dot
                        dot = x_win * doutstride
                        dw[filterIdx, dep, row, col] += np.sum(dot)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    #pass
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N,C,H,W = x.shape
    # out (N, C, Hout, Wout) Hout = 1 + (H-ph)/stride
    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_height) // stride
    out = np.zeros((N,C,Hout,Wout), dtype=x.dtype)
    for inputIdx in xrange(N):
        for dep in xrange(C):
            for row in xrange(Hout):
                for col in xrange(Wout):
                    out[inputIdx, dep, row, col] = np.max(x[inputIdx, dep, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    #pass
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N,C,H,W = x.shape
    _,_,Hout,Wout = dout.shape
    # 0. Init dx with zeros
    dx = np.zeros_like(x)
    for inputIdx in xrange(N):
        for dep in xrange(C):
            for row in xrange(Hout):
                for col in xrange(Wout):
                    # 1. Get Window as big as pool_window. And get the maximum idx
                    maxIdx = np.argmax(x[inputIdx,dep, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width])
                    # 2. Calculate the realtive position of the maximum in the x-pool-window
                    relative_row = maxIdx // pool_width
                    relative_col = maxIdx % pool_width
                    # 3. Fill the dx with 1 * delta, just pass the gradient to the maximum
                    dx[inputIdx, dep, row*stride+relative_row, col*stride+relative_col] += 1 * dout[inputIdx, dep, row, col]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #pass
    # 1. naive implement
    # out = np.zeros_like(x)
    # cache = np.zeros((x.shape[0], x.shape[1]), dtype=tuple)
    # for inputIdx in xrange(x.shape[0]):
    #     for dep in xrange(x.shape[1]):
    #         out[inputIdx, dep,...], cache_one = batchnorm_forward(x[inputIdx, dep, ...], gamma[dep], beta[dep], bn_param)
    #         cache[inputIdx, dep] = cache_one

    # 2. vector implement
    N, C, H, W = x.shape
    out_tmp, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*W*H, C)), gamma, beta, bn_param)
    out = out_tmp.reshape((N, W, H, C)).transpose(0,3,2,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #pass

    # 1. naive implement
    # dx = np.zeros_like(dout)
    # dgamma = np.zeros(dout.shape[1])
    # dbeta = np.zeros(dout.shape[1])
    # for inputIdx in xrange(dout.shape[0]):
    #     for dep in xrange(dout.shape[1]):
    #         _dx, _dgamma, _debta = batchnorm_backward(dout[inputIdx, dep], cache[inputIdx, dep])
    #         dx[inputIdx,dep] += _dx
    #         dgamma[dep] += np.sum(_dgamma)
    #         dbeta[dep] += np.sum(_debta)

    # 2. vector implement
    N, C, H, W = dout.shape
    dx_tmp, dgamma, dbeta = batchnorm_backward(dout.transpose(0,3,2,1).reshape((N*W*H,C)), cache)
    dx = dx_tmp.reshape((N,W,H,C)).transpose(0,3,2,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
