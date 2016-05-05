# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# ##c#ython: profile=True
# ##c#ython: linetrace=True
# ##c#ython: binding=True
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel (partial_fit support)
#         Rob Zinkov (passive-aggressive)
#         Lars Buitinck
#
# Licence: BSD 3 clause


import numpy as np
import sys
from time import time

cimport cython
from libc.math cimport exp, log, sqrt, pow, fabs, cos
cimport numpy as np
cdef extern from "sgd_fast_helpers.h":
    bint skl_isfinite(double) nogil

import scipy.linalg.blas

#    enum CBLAS_ORDER:
#        CblasRowMajor=101
#        CblasColMajor=102
#    enum CBLAS_TRANSPOSE:
#        CblasNoTrans=111
#        CblasTrans=112
#        CblasConjTrans=113
#        AtlasConj=114

from cpython cimport (PY_VERSION_HEX, PyCObject_Check,
    PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer)

cdef void* f2py_pointer(obj):
    if PY_VERSION_HEX < 0x03000000:
        if (PyCObject_Check(obj)):
            return PyCObject_AsVoidPtr(obj)
    elif PY_VERSION_HEX >= 0x02070000:
        if (PyCapsule_CheckExact(obj)):
            return PyCapsule_GetPointer(obj, NULL);
    raise ValueError("Not an object containing a void ptr")


# maybe int return
ctypedef void dgemv_t(
	char *trans,
	int *m, int *n,
	double *alpha,
	double *a, int *lda,
        double *x, int *incX,
	double *beta,
	double *y, int *incY) nogil

# Since Scipy >= 0.12.0
cdef dgemv_t *dgemv = <dgemv_t*>f2py_pointer(scipy.linalg.blas.dgemv._cpointer)

#cdef extern from "cblas.h":
#    enum CBLAS_ORDER:
#        CblasRowMajor=101
#        CblasColMajor=102
#    enum CBLAS_TRANSPOSE:
#        CblasNoTrans=111
#        CblasTrans=112
#        CblasConjTrans=113
#        AtlasConj=114
#
#    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
#                      CBLAS_TRANSPOSE TransA, int M, int N,
#                      double alpha, double *A, int lda,
#                      double *X, int incX, double beta,
#                      double *Y, int incY) nogil


def matvsvec():
    pass


cdef matvec():
    cdef int n_tests = 100

    cdef int n_samples = 5000
    cdef int n_features = 1000
    cdef int n_components = 2000
    cdef double gamma = 0.7
    cdef object random_state = np.random.RandomState()

    cdef double[:, :] x = np.random.rand(n_samples, n_features)
    cdef double[::1, :] rw = np.asarray(np.sqrt(2 * gamma) *
        random_state.normal(size=(n_features, n_components)),
        dtype=np.double, order='F')
    cdef double* x_row_ptr = &x[0,0]

    cdef np.ndarray[double, ndim=1, mode='fortran'] y =\
        np.zeros(n_components, np.double, order="F")
    cdef double* y_ptr = <double*>y.data

    ## Vector x Matrix
    cdef int m, n, lda, incX, incY
    cdef double alpha, beta

    m = n_features; n = n_components; lda = m; incX = 1; incY = 1;
    alpha = 1; beta = 0;

    start_time = time()

    with nogil:
        for _ in n_tests:
            for row in n_samples:
                dgemv('T',  # Transpose please
                    &m, &n, &alpha,
                    &rw[0, 0], &lda,
                    x_row_ptr, &incX,
                    &beta,
                    y_ptr, &incY)

                x_row_ptr += n_features

    print('%f' % time() - start_time)


def test():
    cdef double[::1,:] a
    cdef np.ndarray[double, ndim=1, mode='c'] x
    cdef double* x_ptr
    cdef np.ndarray[double, ndim=1, mode='fortran'] y
    cdef double* y_ptr
    cdef int m, n, lda, incX, incY
    cdef double alpha, beta

    cdef int n_samples = 1000
    cdef int n_components = 2000
    cdef double gamma = 0.7
    cdef object random_state = np.random.RandomState()

    a = np.asarray(np.sqrt(2 * gamma) *
        random_state.normal(size=(n_samples, n_components)),
        dtype=np.double, order='F')
#    a =  np.asarray(np.array([[1, 2, 3], [4, 5, 6]], np.double, order="c"),
#            dtype=np.double, order='F')
    x = random_state.uniform(0, 2 * np.pi, size=n_samples)
#    x = np.array([9, 10], np.double, order="c")
    x_ptr = <double*>x.data
    y = np.zeros(n_components, np.double, order="F")
    y_ptr = <double*>y.data

    alpha = 1.0
    beta = 0.0
    m = n_samples
    n = n_components
    lda = m
    incX = 1
    incY = 1
    with nogil:
        dgemv('T',  # Transpose please
                &m, &n, &alpha,
                &a[0, 0], &lda,
                x_ptr, &incX,
                &beta,
                y_ptr, &incY)
    print(np.asarray(y))
    print(np.dot(x, a))


from sklearn.utils.weight_vector cimport WeightVector
from sklearn.utils.seq_dataset cimport SequentialDataset
from sklearn.kernel_approximation import RBFSampler  # FJ just for testing

np.import_array()

# Penalty constants
DEF NO_PENALTY = 0
DEF L1 = 1
DEF L2 = 2
DEF ELASTICNET = 3

# Learning rate constants
DEF CONSTANT = 1
DEF OPTIMAL = 2
DEF INVSCALING = 3
DEF PA1 = 4
DEF PA2 = 5

# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""

    cdef double loss(self, double p, double y) nogil:
        """Evaluate the loss function.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)

        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return 0.

    def dloss(self, double p, double y):
        """Evaluate the derivative of the loss function with respect to
        the prediction `p`.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)
        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return self._dloss(p, y)

    cdef double _dloss(self, double p, double y) nogil:
        # Implementation of dloss; separate function because cpdef and nogil
        # can't be combined.
        return 0.


cdef class Regression(LossFunction):
    """Base class for loss functions for regression"""

    cdef double loss(self, double p, double y) nogil:
        return 0.

    cdef double _dloss(self, double p, double y) nogil:
        return 0.


cdef class Classification(LossFunction):
    """Base class for loss functions for classification"""

    cdef double loss(self, double p, double y) nogil:
        return 0.

    cdef double _dloss(self, double p, double y) nogil:
        return 0.


cdef class ModifiedHuber(Classification):
    """Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return (1.0 - z) * (1.0 - z)
        else:
            return -4.0 * z

    cdef double _dloss(self, double p, double y) nogil:
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return 2.0 * (1.0 - z) * -y
        else:
            return -4.0 * y

    def __reduce__(self):
        return ModifiedHuber, ()


cdef class Hinge(Classification):
    """Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    # loss(f(xi), yi) where f(xi) = wTxi + b
    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        if z <= self.threshold:
            return (self.threshold - z)
        return 0.0

    cdef double _dloss(self, double p, double y) nogil:
        cdef double z = p * y
        if z <= self.threshold:
            return -y
        return 0.0

    def __reduce__(self):
        return Hinge, (self.threshold,)


cdef class SquaredHinge(LossFunction):
    """Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    """

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cdef double loss(self, double p, double y) nogil:
        cdef double z = self.threshold - p * y
        if z > 0:
            return z * z
        return 0.0

    cdef double _dloss(self, double p, double y) nogil:
        cdef double z = self.threshold - p * y
        if z > 0:
            return -2 * y * z
        return 0.0

    def __reduce__(self):
        return SquaredHinge, (self.threshold,)


cdef class Log(Classification):
    """Logistic regression loss for binary classification with y in {-1, 1}"""

    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))

    cdef double _dloss(self, double p, double y) nogil:
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18.0:
            return exp(-z) * -y
        if z < -18.0:
            return -y
        return -y / (exp(z) + 1.0)

    def __reduce__(self):
        return Log, ()


cdef class SquaredLoss(Regression):
    """Squared loss traditional used in linear regression."""
    cdef double loss(self, double p, double y) nogil:
        return 0.5 * (p - y) * (p - y)

    cdef double _dloss(self, double p, double y) nogil:
        return p - y

    def __reduce__(self):
        return SquaredLoss, ()


cdef class Huber(Regression):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    https://en.wikipedia.org/wiki/Huber_Loss_Function
    """

    cdef double c

    def __init__(self, double c):
        self.c = c

    cdef double loss(self, double p, double y) nogil:
        cdef double r = p - y
        cdef double abs_r = fabs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5 * self.c * self.c)

    cdef double _dloss(self, double p, double y) nogil:
        cdef double r = p - y
        cdef double abs_r = fabs(r)
        if abs_r <= self.c:
            return r
        elif r > 0.0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber, (self.c,)


cdef class EpsilonInsensitive(Regression):
    """Epsilon-Insensitive loss (used by SVR).

    loss = max(0, |y - p| - epsilon)
    """

    cdef double epsilon

    def __init__(self, double epsilon):
        self.epsilon = epsilon

    cdef double loss(self, double p, double y) nogil:
        cdef double ret = fabs(y - p) - self.epsilon
        return ret if ret > 0 else 0

    cdef double _dloss(self, double p, double y) nogil:
        if y - p > self.epsilon:
            return -1
        elif p - y > self.epsilon:
            return 1
        else:
            return 0

    def __reduce__(self):
        return EpsilonInsensitive, (self.epsilon,)


cdef class SquaredEpsilonInsensitive(Regression):
    """Epsilon-Insensitive loss.

    loss = max(0, |y - p| - epsilon)^2
    """

    cdef double epsilon

    def __init__(self, double epsilon):
        self.epsilon = epsilon

    cdef double loss(self, double p, double y) nogil:
        cdef double ret = fabs(y - p) - self.epsilon
        return ret * ret if ret > 0 else 0

    cdef double _dloss(self, double p, double y) nogil:
        cdef double z
        z = y - p
        if z > self.epsilon:
            return -2 * (z - self.epsilon)
        elif z < self.epsilon:
            return 2 * (-z - self.epsilon)
        else:
            return 0

    def __reduce__(self):
        return SquaredEpsilonInsensitive, (self.epsilon,)


def plain_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
              double intercept,
              LossFunction loss,
              int penalty_type,
              double alpha, double C,
              double l1_ratio,
              SequentialDataset dataset,
              int n_iter, int fit_intercept,
              int verbose, bint shuffle, np.uint32_t seed,
              double weight_pos, double weight_neg,
              int learning_rate, double eta0,
              double power_t,
              double t=1.0,
              double intercept_decay=1.0,
              RBFSamplerInPlace rbf=None):
    """Plain SGD for generic loss functions and penalties.

    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    n_iter : int
        The number of iterations (epochs).
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) Passive Agressive-I, eta = min(alpha, loss/norm(x))
        (5) Passive Agressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    power_t : double
        The exponent for inverse scaling learning rate.
    t : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.

    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    """
#    import line_profiler
    import sys

    #profile = line_profiler.LineProfiler(_plain_sgd)

#        _, _ = profile.runcall(_plain_sgd, weights,
    standard_weights, standard_intercept,\
        _, _ = _plain_sgd(weights,
                          intercept,
                          None,
                          0,
                          loss,
                          penalty_type,
                          alpha, C,
                          l1_ratio,
                          dataset,
                          n_iter, fit_intercept,
                          verbose, shuffle, seed,
                          weight_pos, weight_neg,
                          learning_rate, eta0,
                          power_t,
                          t,
                          intercept_decay,
                          average=0,
                          rbf=rbf)
#    profile.print_stats()
    sys.exit(1)
    return standard_weights, standard_intercept


def average_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
                double intercept,
                np.ndarray[double, ndim=1, mode='c'] average_weights,
                double average_intercept,
                LossFunction loss,
                int penalty_type,
                double alpha, double C,
                double l1_ratio,
                SequentialDataset dataset,
                int n_iter, int fit_intercept,
                int verbose, bint shuffle, np.uint32_t seed,
                double weight_pos, double weight_neg,
                int learning_rate, double eta0,
                double power_t,
                double t=1.0,
                double intercept_decay=1.0,
                int average=1):
    """Average SGD for generic loss functions and penalties.

    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    average_weights : ndarray[double, ndim=1]
        The average weights as computed for ASGD
    average_intercept : double
        The average intercept for ASGD
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    n_iter : int
        The number of iterations (epochs).
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) Passive Agressive-I, eta = min(alpha, loss/norm(x))
        (5) Passive Agressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    power_t : double
        The exponent for inverse scaling learning rate.
    t : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.
    average : int
        The number of iterations before averaging starts. average=1 is
        equivalent to averaging for all iterations.

    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    average_weights : array shape=[n_features]
        The averaged weights accross iterations
    average_intercept : float
        The averaged intercept accross iterations
    """
    return _plain_sgd(weights,
                      intercept,
                      average_weights,
                      average_intercept,
                      loss,
                      penalty_type,
                      alpha, C,
                      l1_ratio,
                      dataset,
                      n_iter, fit_intercept,
                      verbose, shuffle, seed,
                      weight_pos, weight_neg,
                      learning_rate, eta0,
                      power_t,
                      t,
                      intercept_decay,
                      average)


def _plain_sgd(np.ndarray[double, ndim=1, mode='c'] weights,
               double intercept,
               np.ndarray[double, ndim=1, mode='c'] average_weights,
               double average_intercept,
               LossFunction loss,
               int penalty_type,
               double alpha, double C,
               double l1_ratio,
               SequentialDataset dataset,
               int n_iter, int fit_intercept,
               int verbose, bint shuffle, np.uint32_t seed,
               double weight_pos, double weight_neg,
               int learning_rate, double eta0,
               double power_t,
               double t=1.0,
               double intercept_decay=1.0,
               int average=0,
               RBFSamplerInPlace rbf=None):

    if rbf is not None:
        assert weights.shape[0] == rbf.n_components,\
                'weights vector not scaled appropriately for RBF'
        assert (average_weights is None or
                average_weights.shape[0] == rbf.n_components),\
                'average_weights vector not scaled appropriately for RBF'

    # BLAS
#    cdef int bl_m, bl_n, bl_lda, bl_incX, bl_incY
#    cdef double bl_alpha, bl_beta
##    cdef double* rbf_random_weights_ptr = &rbf.random_weights_[0, 0]

#    cdef np.ndarray[double, ndim=2, mode='c'] rbf_random_weights_
#    cdef double* rbf_random_weights_ptr_
#    if rbf is not None:
#        rbf_random_weights_ = rbf.random_weights_
#        rbf_random_weights_ptr_ = <double*>rbf_random_weights_.data

    # get the data information into easy vars
    cdef Py_ssize_t n_samples = dataset.n_samples
    cdef Py_ssize_t n_features = weights.shape[0]

    cdef WeightVector w = WeightVector(weights, average_weights)
    cdef double* w_ptr = &weights[0]
    cdef double *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef double* ps_ptr = NULL
    cdef int xnnz

    # Values used in the RBF case. If there is no RBF Sampler, these will point
    # to the above variables with the same name but without the '_rbf' at the
    # end. If there is an RBF Sampler, the x_ind_rbf_ptr and the xnnz_rbf
    # variables always hold the same value, namely all indices from 0 to
    # n_components-1 and n_components, respectively.
    cdef double *x_data_rbf_ptr = NULL
    cdef int *x_ind_rbf_ptr = NULL
    cdef int xnnz_rbf
    cdef np.ndarray[double, ndim=1, mode='c'] _x_data_rbf
    cdef np.ndarray[int, ndim=1, mode='c'] _x_ind_rbf

### HAND INLINED ####
    # current column in random_weights_
    cdef int col

    # current component when doing multiplication, see below
    cdef int idx

    # holds value for x_i * random_weights_[:, col] before it gets written
    cdef double out_val
### / HAND INLINED ####

    # helper variables
    cdef bint infinity = False
    cdef double eta = 0.0
    cdef double p = 0.0
    cdef double update = 0.0
    cdef double sumloss = 0.0
    cdef double y = 0.0
    cdef double sample_weight
    cdef double class_weight = 1.0
    cdef unsigned int count = 0
    cdef unsigned int epoch = 0
    cdef unsigned int i = 0
    cdef unsigned int j # FJ for debugging
    cdef int is_hinge = isinstance(loss, Hinge)
    cdef double optimal_init = 0.0
    cdef double dloss = 0.0
    cdef double MAX_DLOSS = 1e12

    # q vector is only used for L1 regularization
    cdef np.ndarray[double, ndim = 1, mode = "c"] q = None
    cdef double * q_data_ptr = NULL
    if penalty_type == L1 or penalty_type == ELASTICNET:
        assert rbf is None, 'Unsupported penalty for RBF'

        q = np.zeros((n_features,), dtype=np.float64, order="c")
        q_data_ptr = <double * > q.data

    cdef double u = 0.0

    if penalty_type == L2:
        l1_ratio = 0.0
    elif penalty_type == L1:
        l1_ratio = 1.0

    eta = eta0

    if learning_rate == OPTIMAL:
        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing eta0, the initial learning rate
        initial_eta0 = typw / max(1.0, loss.dloss(-typw, 1.0))
        # initialize t such that eta at first sample equals eta0
        optimal_init = 1.0 / (initial_eta0 * alpha)

    t_start = time()
    t_per_hundred = time()

    if rbf is not None:
        # if there is an RBF, this is the memory that holds the current
        # transformed value in each iteration.
        _x_data_rbf = np.zeros(rbf.n_components, dtype=np.double)
        x_data_rbf_ptr = <double*>_x_data_rbf.data

        # these remain fixed because the RBF transformed X is hardly sparse.
        _x_ind_rbf = np.arange(0, rbf.n_components, dtype=np.intc)
        x_ind_rbf_ptr = <int*>_x_ind_rbf.data
        xnnz_rbf = rbf.n_components

#    # BLAS
#    bl_m = n_samples
#    bl_n = rbf.n_components
#    bl_lda = bl_m
#    bl_incX = 1
#    bl_incY = 1
#    bl_alpha = 1.0
#    bl_beta = 1.0

#    print '%d %d %d %d %d %f %f' % (
#            bl_m, bl_n, bl_lda, bl_incX, bl_incY, bl_alpha, bl_beta)
#
#    cdef object random_state = np.random.RandomState()
#    cdef double[::1,:] a
#    a = np.asarray(np.sqrt(2 * 0.7) *
#        random_state.normal(size=(n_samples, rbf.n_components)),
#        dtype=np.double, order='F')

#    cdef double* rbf_random_weights_ptr = &a[0, 0]

    with nogil:
        for epoch in range(n_iter):
            if verbose > 0:
                with gil:
                    print("-- Epoch %d" % (epoch + 1))
            if shuffle:
                dataset.shuffle(seed)
            for i in range(min(n_samples, 101)):
                if i % 10 == 0:
                    with gil:
                        print('%i: %f' % (i, time() - t_per_hundred))
                        t_per_hundred = time()

                dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
                             &y, &sample_weight)

                # update RBF variables
                if rbf is not None:
#### BEGIN hand inlined ####

#                    for col in range(rbf.n_components):
#                        out_val = 0
#                        for i in range(xnnz):  # 1.
#                            idx = x_ind_ptr[i]
#                            out_val += (x_data_ptr[i] *
#                                    rbf.random_weights_[idx, col])
#                        out_val += rbf.random_offset_[col]  # 2.
#                        out_val = cos(out_val)  # 3.
#                        out_val *= rbf.factor_  # 4.
#
#                        x_data_rbf_ptr[col] = out_val

                    rbf.transform(x_data_ptr, x_ind_ptr, xnnz, x_data_rbf_ptr)

#                    # setup for gemv
#                    for col in range(rbf.n_components):
#                        x_data_rbf_ptr[col] = rbf.random_offset_[col]
#
#                    dgemv('T',  # Transpose please
#                        &bl_m, &bl_n, &bl_alpha,
#                        rbf_random_weights_ptr, &bl_lda,
##                        rbf.random_weights_ptr_, &bl_lda,
#                        x_data_ptr, &bl_incX,
#                        &bl_beta,
#                        x_data_rbf_ptr, &bl_incY)
#
#                    for col in range(rbf.n_components):
#                        x_data_rbf_ptr[col] = (rbf.factor_ *
#                                cos(x_data_rbf_ptr[col]))


#### END   hand inlined ####
                else:
                    x_data_rbf_ptr = x_data_ptr
                    x_ind_rbf_ptr = x_ind_ptr
                    xnnz_rbf = xnnz

                p = w.dot(x_data_rbf_ptr, x_ind_rbf_ptr, xnnz_rbf) + intercept

                if verbose > 1:
                    with gil:
                        print 'x: %s' % (' '.join(str(x_data_rbf_ptr[j])
                                         for j in range(xnnz_rbf)[:10]))
                        print 'wTx + %i= %f' % (intercept, p)

                if learning_rate == OPTIMAL:
                    eta = 1.0 / (alpha * (optimal_init + t - 1))
                elif learning_rate == INVSCALING:
                    eta = eta0 / pow(t, power_t)

                if verbose > 0:
                    sumloss += loss.loss(p, y)

                if y > 0.0:
                    class_weight = weight_pos
                else:
                    class_weight = weight_neg

                if learning_rate == PA1:
                    update = sqnorm(x_data_rbf_ptr, x_ind_rbf_ptr, xnnz_rbf)
                    if update == 0:
                        continue
                    update = min(C, loss.loss(p, y) / update)
                elif learning_rate == PA2:
                    update = sqnorm(x_data_rbf_ptr, x_ind_rbf_ptr, xnnz_rbf)
                    update = loss.loss(p, y) / (update + 0.5 / C)
                else:
                    dloss = loss._dloss(p, y)
                    # clip dloss with large values to avoid numerical
                    # instabilities
                    if dloss < -MAX_DLOSS:
                        dloss = -MAX_DLOSS
                    elif dloss > MAX_DLOSS:
                        dloss = MAX_DLOSS
                    update = -eta * dloss

                if learning_rate >= PA1:
                    if is_hinge:
                        # classification
                        update *= y
                    elif y - p < 0:
                        # regression
                        update *= -1

                update *= class_weight * sample_weight

                if penalty_type >= L2:
                    # do not scale to negative values when eta or alpha are too
                    # big: instead set the weights to zero
                    w.scale(max(0, 1.0 - ((1.0 - l1_ratio) * eta * alpha)))
                if update != 0.0:
                    w.add(x_data_rbf_ptr, x_ind_rbf_ptr, xnnz_rbf, update)

                    if fit_intercept == 1:
                        intercept += update * intercept_decay

                if 0 < average <= t:
                    # compute the average for the intercept and update the
                    # average weights, this is done regardless as to whether
                    # the update is 0

                    w.add_average(x_data_rbf_ptr, x_ind_rbf_ptr, xnnz_rbf,
                                  update, (t - average + 1))
                    average_intercept += ((intercept - average_intercept) /
                                          (t - average + 1))

                if penalty_type == L1 or penalty_type == ELASTICNET:
                    # FJ unmodified
                    u += (l1_ratio * eta * alpha)
                    l1penalty(w, q_data_ptr, x_ind_ptr, xnnz, u)

                t += 1
                count += 1

            # report epoch information
            if verbose > 0:
                with gil:
                    print("Norm: %.2f, NNZs: %d, "
                          "Bias: %.6f, T: %d, Avg. loss: %.6f"
                          % (w.norm(), weights.nonzero()[0].shape[0],
                             intercept, count, sumloss / count))
                    print("Total training time: %.2f seconds."
                          % (time() - t_start))

            # floating-point under-/overflow check.
            if (not skl_isfinite(intercept)
                or any_nonfinite(<double *>weights.data, n_features)):
                infinity = True
                break

    if infinity:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % (epoch + 1))

    w.reset_wscale()

    return weights, intercept, average_weights, average_intercept


#def test_dot(X, rw):
#    return _test_dot(X, rw)
#
#
#cdef object _test_dot(
#        np.ndarray[double, ndim = 2, mode = "c"] X,
#        np.ndarray[double, ndim = 2, mode = "c"] rw):
#    cdef int n_components = rw.shape[1]
#    cdef int n_features = rw.shape[0]
#    cdef float gamma = 1
#    cdef double* x_data_ptr = <double*>X.data
#    cdef np.ndarray[int, ndim = 1, mode = "c"] x_ind = np.arange(n_features, dtype=np.intc)
#    print x_ind
#
#    cdef int* x_ind_ptr = <int*>x_ind.data
#    cdef int xnnz = n_features
#
#    cdef np.ndarray[double, ndim = 1, mode = "c"] x_data_rbf = np.zeros(n_components)
#    cdef double* x_data_rbf_ptr = <double*>x_data_rbf.data
#
#    cdef int i
#
#    cdef RBFSamplerInPlace rbf = RBFSamplerInPlace(gamma, n_components)
#    rbf.random_weights_ = rw
#    rbf.random_offset_ = np.zeros(n_components)
#    rbf.factor_ = np.sqrt(2.) / np.sqrt(n_components)
#
#    with nogil:
#        rbf.transform(x_data_ptr, x_ind_ptr, xnnz, x_data_rbf_ptr)
#
#    print 'output RBF'
#    for i in range(n_components):
#        print x_data_rbf[i]
#
#    real_rbf = rbf.get_RBFSampler()
#    return real_rbf
#

cdef class RBFSamplerInPlace:
    cdef public float gamma
    cdef public int n_components
    cdef double[::1, :] random_weights_  # FORTRAN style
    cdef double* random_weights_ptr_
    cdef double[:] random_offset_
    cdef public double factor_

    def __init__(self, float gamma, int n_components):
        self.gamma = gamma
        self.n_components = n_components
        self.random_weights_ = None
        self.random_offset_ = None

    # FJ just to debug
    def get_RBFSampler(self):
        rbf = RBFSampler(self.gamma, self.n_components)
        rbf.random_weights_ = self.random_weights_
        rbf.random_offset_ = self.random_offset_
        return rbf

    def fit(self, n_features, random_state):
        self.random_weights_ = np.asarray(np.sqrt(2 * self.gamma) *
            random_state.normal(size=(n_features, self.n_components)),
            dtype=np.double, order='F')
        self.random_weights_ptr_ = &self.random_weights_[0, 0]

        self.random_offset_ = random_state.uniform(0, 2 * np.pi,
                                                   size=self.n_components)

        # calculate factor from step 4. below only once
        self.factor_ = np.sqrt(2.) / np.sqrt(self.n_components)

    cdef void transform(self,
            double* x_data_ptr, int* x_ind_ptr, int xnnz,  # data to transform
            double* x_data_rbf_ptr) nogil:  # output
        """
        Calculates
        1. projection = safe_sparse_dot(X, self.random_weights_)  # dot product
        2. projection += self.random_offset_
        3. np.cos(projection, projection)  # second argument is output
        4. projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        """

#        cdef int bl_m, bl_n, bl_lda, bl_incX, bl_incY
#        cdef double bl_alpha, bl_beta
#
#        bl_m = xnnz
#        bl_n = self.n_components
#        bl_lda = bl_m
#        bl_incX = 1
#        bl_incY = 1
#        bl_alpha = 1.0
#        bl_beta = 1.0
#
#        # setup for gemv
#        for col in range(self.n_components):
#            x_data_rbf_ptr[col] = self.random_offset_[col]
#
#        dgemv('T',  # Transpose please
#            &bl_m, &bl_n, &bl_alpha,
#            self.random_weights_ptr_, &bl_lda,
#            x_data_ptr, &bl_incX,
#            &bl_beta,
#            x_data_rbf_ptr, &bl_incY)
#
#        for col in range(self.n_components):
#            x_data_rbf_ptr[col] = self.factor_ * cos(x_data_rbf_ptr[col])

        # current column in random_weights_
        cdef int col

        # current component when doing multiplication, see below
        cdef int idx

        # holds value for x_i * random_weights_[:, col] before it gets written
        cdef double out_val

        # iterate over columns of random_weights_
        for col in range(self.n_components):
            out_val = 0
            for i in range(xnnz):  # 1.
                idx = x_ind_ptr[i]  # index of the i-th non-zero element of x
                out_val += x_data_ptr[i] * self.random_weights_[idx, col]
            out_val += self.random_offset_[col]  # 2.
            out_val = cos(out_val)  # 3.
            out_val *= self.factor_  # 4.

            x_data_rbf_ptr[col] = out_val

    def transform_and_multiply_mat(self, dataset, coef, Y):
        n_samples = dataset.n_samples
        (n_classes, coef_n_cols) = coef.shape
        (y_num_rows, y_num_cols) = Y.shape

        assert coef_n_cols == self.n_components, 'Invalid coef # of cols'
        assert y_num_rows == n_samples, 'Invalid Y # of rows'
        assert y_num_cols == n_classes, 'Invalid Y # of classes'

        self._transform_and_multiply_mat(dataset, coef, Y)

    cdef _transform_and_multiply_mat(self,
        SequentialDataset dataset,
        np.ndarray[double, ndim = 2, mode = "c"] coef,
        np.ndarray[double, ndim = 2, mode = "c"] Y):
        """
        transforms dataset row by row and then multiplies each row with coef,
        which is expected to be of shape (n_components, n_classes), i.e.
        transformed stores the result in Y
        """
        ## Declarations ####

        cdef Py_ssize_t n_samples
        cdef Py_ssize_t n_classes

        cdef double y  # unused
        cdef double sample_weight  # unused

        # holds current row information *before* transformation
        cdef double* x_row_ptr
        cdef int* x_row_ind_ptr
        cdef int xnnz

        # where the current row is stored *after* transformation
        cdef np.ndarray[double, ndim=1, mode='c'] _x_row_rbf
        cdef double* x_row_rbf_ptr

        # current value of the next output, built up during matrix
        # multiplication
        cdef double out_val

        # indices
        cdef int sample_idx, class_idx, i

        ## Assigment ####

        n_samples = dataset.n_samples
        n_classes = coef.shape[0]

        _x_row_rbf = np.zeros(self.n_components, dtype=np.double)
        x_row_rbf_ptr = <double*>_x_row_rbf.data

        for sample_idx in range(n_samples):
            dataset.next(&x_row_ptr, &x_row_ind_ptr, &xnnz,
                         &y, &sample_weight)

            self.transform(x_row_ptr, x_row_ind_ptr, xnnz, x_row_rbf_ptr)

            # compute matrix product: x_row_rbf_ptr * coef.T
            for class_idx in range(n_classes):
                out_val = 0
                for i in range(self.n_components):
                    out_val += x_row_rbf_ptr[i] * coef[class_idx, i]
                Y[sample_idx, class_idx] = out_val


cdef bint any_nonfinite(double *w, int n) nogil:
    for i in range(n):
        if not skl_isfinite(w[i]):
            return True
    return 0


cdef double sqnorm(double * x_data_ptr, int * x_ind_ptr, int xnnz) nogil:
    cdef double x_norm = 0.0
    cdef int j
    cdef double z
    for j in range(xnnz):
        z = x_data_ptr[j]
        x_norm += z * z
    return x_norm


cdef void l1penalty(WeightVector w, double * q_data_ptr,
                    int *x_ind_ptr, int xnnz, double u) nogil:
    """Apply the L1 penalty to each updated feature

    This implements the truncated gradient approach by
    [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009].
    """
    cdef double z = 0.0
    cdef int j = 0
    cdef int idx = 0
    cdef double wscale = w.wscale
    cdef double *w_data_ptr = w.w_data_ptr
    for j in range(xnnz):
        idx = x_ind_ptr[j]
        z = w_data_ptr[idx]
        if wscale * w_data_ptr[idx] > 0.0:
            w_data_ptr[idx] = max(
                0.0, w_data_ptr[idx] - ((u + q_data_ptr[idx]) / wscale))

        elif wscale * w_data_ptr[idx] < 0.0:
            w_data_ptr[idx] = min(
                0.0, w_data_ptr[idx] + ((u - q_data_ptr[idx]) / wscale))

        q_data_ptr[idx] += wscale * (w_data_ptr[idx] - z)
