""" Legendre Decomposition
"""
import numpy as np
import itertools
# from scipy import linalg
from enum import Enum
#import numba
#from sklearn.base import TransformerMixin, BaseEstimator

# TODO: Support scipy sparse matrix.
# TODO: Improve performance using cython
# TODO: Check scope of class variables.


class Constants(Enum):
    EPSILON = 1e-100


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


#class LegendreDecomposition(TransformerMixin, BaseEstimator):
class LegendreDecomposition:
    """Legendre Decomposition

    Find non-negative decomposable tensor Q whose multiplicative
    combination of parameters approximates the non-
    negative input tensor P, solving as a convex optimization
    problem thanks to information geometric formulation.

    Parameters
    ----------
    core_size : integer, default: 2
        The parameter for a decomposition basis.

    solver : 'ng' | 'gd', default: 'ng'
        Type of solver.

        - 'ng': natural gradient method.
        - 'gd': gradient descent method.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 10
        Maximum number of iterations before timing out.

    learning_rate : float, default: 0.1
        The learning rate used in gradient descent method.

    random_state : int, RandomState instance, default=None
        Used to randomize selection of a decomposition basis, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : boolean, default: False
        If true, randomize selection of a decomposition basis.

    verbose : integer, default: 0
        The verbosity level.

    Attributes
    ----------

    Examples
    --------
    >>> import numpy as np
    >>> P = np.random.rand(8, 5, 3)
    >>> from ld import LegendreDecomposition
    >>> ld = LegendreDecomposition(solver='ng', max_iter=5)
    >>> Q = ld.fit_transform(P)

    References
    ----------
    M. Sugiyama, H. Nakahara, K. Tsuda.
    "Legendre Decomposition for Tensors".
    Advances in Neural Information Processing Systems 31(NeurIPS2018),
    pages 8825-8835, 2018.
    https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors
    """

    def __init__(self, core_size=2, solver='ng',
                 tol=1e-4, max_iter=10, learning_rate=0.1,
                 random_state=None, shuffle=False, verbose=0):
        self.core_size = core_size
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(threshold=200)

    def fit_transform(self, P, y=None):
        r"""Learn a Legendre Decomposition model for the data P and
        returns the transformed data.
        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        Q : array
            second/third-order tensor.
            Transformed data reconstructed by parameter \theta.
        """
        self.theta = self._legendre_decomposition(P)
        Q = self._compute_Q(self.theta, self.beta) * P.sum()
        self.reconstruction_err_ = self._calc_rmse(P, Q)

        return Q

    def fit(self, P, y=None, **params):
        """Learn a Legendre Decomposition model for the data.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        self
        """
        self.fit_transform(P, **params)

        return self

    def backward_transform(self, theta):
        Q = self._compute_Q(theta, self.beta) * self.P.sum()
        return Q

    def transform(self):
        r"""Transform the data P according to the fitted Legendre Decomposition model.

        Parameters
        ----------
        None

        Returns
        -------
        Q : array
            second/third-order tensor.
            Transformed data reconstructed by parameter \theta.
        """
        self._check_is_fitted(self)

        return self._compute_Q(self.theta, self.beta)

    def _compute_Q_(self, theta, beta):
        r"""Compute decomposable tensor Q from parameter \theta.

        Parameters
        ----------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        Q : array
            second/third-order tensor.
            Decomposable tensor.
        """
        idx = theta.shape
        order = len(theta.shape)
        theta_sum = np.zeros(theta.shape)

        if order == 2:
            for i, j in itertools.product(range(idx[0]), range(idx[1])):
                for v in beta:
                    if (v[0] <= i) and (v[1] <= j):
                        theta_sum[i ,j] += theta[v]
        elif order == 3:
            for i, j, k in itertools.product(range(idx[0]), range(idx[1]), range(idx[2])):
                for v in beta:
                    if (v[0] <= i) and (v[1] <= j) and (v[2] <= k):
                        theta_sum[i ,j, k] += theta[v]
        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        Q = np.exp(theta_sum)
        psi = Q.sum()
        Q /= psi

        return Q

    # Using DP must help faster.
    def _compute_Q(self, theta, beta=None):
        r"""Compute decomposable tensor Q from parameter \theta using Dynamic Programming.

        Parameters
        ----------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        Q : array
            second/third-order tensor.
            Decomposable tensor.
        """
        idx = theta.shape
        order = len(theta.shape)
        theta_sum = np.zeros(theta.shape)

        if order == 2:
            theta_sum[0, 0] = theta[0, 0]

            # update outside eta.
            for i in range(1, idx[0]):
                theta_sum[i, 0] = theta[i, 0] + theta_sum[i-1, 0]
            for j in range(1, idx[1]):
                theta_sum[0, j] = theta[0, j] + theta_sum[0, j-1]

            # update internal eta.
            for i in range(1, idx[0]):
                for j in range(1, idx[1]):
                    theta_sum[i, j] = theta[i, j] + theta_sum[i-1, j] \
                                        + theta_sum[i, j-1] - theta_sum[i-1, j-1]

        elif order == 3:
            theta_sum[0, 0, 0] = theta[0, 0, 0]

            # update outside eta.
            for i in range(1, idx[0]):
                theta_sum[i, 0, 0] = theta[i, 0, 0] + theta_sum[i-1, 0, 0]
            for j in range(1, idx[1]):
                theta_sum[0, j, 0] = theta[0, j, 0] + theta_sum[0, j-1, 0]
            for k in range(1, idx[2]):
                theta_sum[0, 0, k] = theta[0, 0, k] + theta_sum[0, 0, k-1]

            # update internal eta.
            for i, j in itertools.product(range(1, idx[0]), range(1, idx[1])):
                theta_sum[i, j, 0] = theta[i, j, 0] + theta_sum[i-1, j, 0] \
                                        + theta_sum[i, j-1, 0] - theta_sum[i-1, j-1, 0]
            for j, k in itertools.product(range(1, idx[1]), range(1, idx[2])):
                theta_sum[0, j, k] = theta[0, j, k] + theta_sum[0, j-1, k] \
                                        + theta_sum[0, j, k-1] - theta_sum[0, j-1, k-1]
            for i, k in itertools.product(range(1, idx[0]), range(1, idx[2])):
                theta_sum[i, 0, k] = theta[i, 0, k] + theta_sum[i-1, 0, k] \
                                        + theta_sum[i, 0, k-1] - theta_sum[i-1, 0, k-1]

            for i, j, k in itertools.product(range(1, idx[0]), range(1, idx[1]), range(1, idx[2])):
                theta_sum[i, j, k] = theta[i, j, k] + theta_sum[i-1, j, k] + theta_sum[i, j-1, k] \
                                    + theta_sum[i, j, k-1] - theta_sum[i-1, j-1, k] - theta_sum[i-1, j, k-1] \
                                    - theta_sum[i, j-1, k-1] + theta_sum[i-1, j-1, k-1]

        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        Q = np.exp(theta_sum)
        psi = Q.sum()
        Q /= psi

        return Q

    def _compute_eta_(self, Q):
        r"""Compute parameters \eta from decomposable tensor Q.

        Parameters
        ----------
        Q : array
            second/third-order tensor.
            Decomposable tensor.

        Returns
        -------
        eta : array
            second/third-order tensor.
            parameter \eta.
            Same shapes as input tensor P.
        """
        shape = Q.shape
        order = len(Q.shape)
        eta = np.zeros(Q.shape)

        if order == 2:
            for i, j in itertools.product(range(shape[0]), range(shape[1])):
                eta[i, j] = Q[np.arange(i, shape[0])][:, np.arange(j, shape[1])].sum()
        elif order == 3:
            for i, j, k in itertools.product(range(shape[0]), range(shape[1]), range(shape[2])):
                eta[i, j, k] = Q[np.arange(i, shape[0])]\
                                [:, np.arange(j, shape[1])][:, :, np.arange(k, shape[2])].sum()
        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        return eta

    # Using DP must help faster.
    def _compute_eta(self, Q):
        r"""Compute parameters \eta from decomposable tensor Q using Dynamic Programming.

        Parameters
        ----------
        Q : array
            second/third-order tensor.
            Decomposable tensor.

        Returns
        -------
        eta : array
            second/third-order tensor.
            parameter \eta.
            Same shapes as input tensor P.
        """
        idx = [i - 1 for i in Q.shape]
        order = len(Q.shape)
        eta = np.zeros(Q.shape)

        if order == 2:
            eta[idx[0], idx[1]] = Q[idx[0], idx[1]]

            # update outside eta.
            for i in range(idx[0])[::-1]:
                eta[i, idx[1]] = Q[i, idx[1]] + eta[i+1, idx[1]]
            for j in range(idx[1])[::-1]:
                eta[idx[0], j] = Q[idx[0], j] + eta[idx[0], j+1]

            # update internal eta.
            for i in range(idx[0])[::-1]:
                for j in range(idx[1])[::-1]:
                    eta[i, j] = Q[i, j] + eta[i+1, j] + eta[i, j+1] - eta[i+1, j+1]

        elif order == 3:
            eta[idx[0], idx[1], idx[2]] = Q[idx[0], idx[1], idx[2]]

            # update outside eta.
            for i in range(idx[0])[::-1]:
                eta[i, idx[1], idx[2]] = Q[i, idx[1], idx[2]] + eta[i+1, idx[1], idx[2]]
            for j in range(idx[1])[::-1]:
                eta[idx[0], j, idx[2]] = Q[idx[0], j, idx[2]] + eta[idx[0], j+1, idx[2]]
            for k in range(idx[2])[::-1]:
                eta[idx[0], idx[1], k] = Q[idx[0], idx[1], k] + eta[idx[0], idx[1], k+1]

            # update internal eta.
            for i, j in itertools.product(range(idx[0])[::-1], range(idx[1])[::-1]):
                eta[i, j, idx[2]] = Q[i, j, idx[2]] + eta[i+1, j, idx[2]] \
                                        + eta[i, j+1, idx[2]] - eta[i+1, j+1, idx[2]]
            for j, k in itertools.product(range(idx[1])[::-1], range(idx[2])[::-1]):
                eta[idx[0], j, k] = Q[idx[0], j, k] + eta[idx[0], j+1, k] \
                                        + eta[idx[0], j, k+1] - eta[idx[0], j+1, k+1]
            for i, k in itertools.product(range(idx[0])[::-1], range(idx[2])[::-1]):
                eta[i, idx[1], k] = Q[i, idx[1], k] + eta[i+1, idx[1], k] \
                                        + eta[i, idx[1], k+1] - eta[i+1, idx[1], k+1]

            for i, j, k in itertools.product(range(idx[0])[::-1], range(idx[1])[::-1], range(idx[2])[::-1]):
                eta[i, j, k] = Q[i, j, k] + eta[i+1, j, k] + eta[i, j+1, k] + eta[i, j, k+1] \
                                - eta[i+1, j+1, k] - eta[i+1, j, k+1] - eta[i, j+1, k+1] \
                                + eta[i+1, j+1, k+1]

        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        return eta

    def _compute_jacobian(self, eta, beta):
        """Compute jacobian matrix, this is what we call Fisher information matrix.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        g : array, shape (cardinality of basis, cardinality of basis)
            Fisher information matrix.
        """
        size = len(beta)
        g = np.zeros((size, size))
        order = len(eta.shape)

        if order == 2:
            for i, j in itertools.product(range(size), range(size)):
                g[i, j] = eta[np.max((beta[i][0], beta[j][0])), \
                              np.max((beta[i][1], beta[j][1]))] \
                            - eta[beta[i]] * eta[beta[j]]
        elif order == 3:
            for i, j in itertools.product(range(size), range(size)):
                g[i, j] = eta[np.max((beta[i][0], beta[j][0])), \
                              np.max((beta[i][1], beta[j][1])), \
                              np.max((beta[i][2], beta[j][2]))] \
                            - eta[beta[i]] * eta[beta[j]]
        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        return g

    def _compute_residual(self, eta, beta):
        """Compute root mean squared error(rmse),
        which is reconstructed error between input tensor P and reconstrcted tensor Q.

        Parameters
        ----------
        eta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        Returns
        -------
        residual : float
        """
        res = np.sqrt(np.mean([(eta[v] - self.eta_hat[v])**2 for v in beta]))
        return res

    def _calc_rmse(self, P, Q):
        """Compute root mean squared error(rmse),
        which is reconstructed error between input tensor P and reconstrcted tensor Q.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Q : array
            second/third-order tensor.
            Decomposable tensor.

        Returns
        -------
        rmse : float
            rmse of reconstructed error.
        """
        return np.sqrt(np.mean(np.square(P - Q)))

    def _check_is_fitted(self, estimator, attributes=None, msg=None, all_or_any=all):
        """Perform is_fitted validation for estimator.
        Checks if the estimator is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        This utility is meant to be used internally by estimators themselves,
        typically in their own predict / transform methods.

        Parameters
        ----------
        estimator : estimator instance.
            estimator instance for which the check is performed.
        attributes : str, list or tuple of str, default=None
            Attribute name(s) given as string or a list/tuple of strings
            Eg.: ``["coef_", "estimator_", ...], "coef_"``
            If `None`, `estimator` is considered fitted if there exist an
            attribute that ends with a underscore and does not start with double
            underscore.
        msg : string
            The default error message is, "This %(name)s instance is not fitted
            yet. Call 'fit' with appropriate arguments before using this
            estimator."
            For custom messages if "%(name)s" is present in the message string,
            it is substituted for the estimator name.
            Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
        all_or_any : callable, {all, any}, default all
            Specify whether all or any of the given attributes must exist.
        Returns
        -------
        None
        Raises
        ------
        NotFittedError
            If the attributes are not found.
        """
        if msg is None:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator.")

        if not hasattr(estimator, 'fit'):
            raise TypeError("%s is not an estimator instance." % (estimator))

        if attributes is not None:
            if not isinstance(attributes, (list, tuple)):
                attributes = [attributes]
            attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
        else:
            attrs = [v for v in vars(estimator)
                    if v.endswith("_") and not v.startswith("__")]

        if not attrs:
            raise NotFittedError(msg % {'name': type(estimator).__name__})

    def _normalizer(self, P):
        """normalize input tensor P by summation of P.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Returns
        -------
        norm_P : array
            second/third-order tensor.
            Normalized data tensor to be decomposed.
        """
        # TODO: check if tensor has NaN values.
        return P / np.sum(P)

    def _initialize(self):
        r"""Initialize paramters \theta, \eta and decomposable tensor Q.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta = np.zeros(self.shape)

        return theta

    def _gen_norm(self, shape):
        """Generate set of decomposition basis B
        for Natural Gradient-based algorithm

        Parameters
        ----------
        shape : int
            shapes of the input tensor P.

        Returns
        -------
        beta : list
            set of decomposition basis vectors.
        """
        order = len(shape)
        beta = []
        temp_beta = []

        if order == 2:
            # B_1
            for i in range(shape[0]):
                if self.basis_index[i, 0] == 0:
                    temp_beta.append((i, 0))
            for j in range(shape[1]):
                if self.basis_index[0, j] == 0:
                    temp_beta.append((0, j))

        elif order == 3:
            # B_1
            for i in range(shape[0]):
                if self.basis_index[i, 0, 0] == 0:
                    temp_beta.append((i, 0, 0))
            for j in range(shape[1]):
                if self.basis_index[0, j, 0] == 0:
                    temp_beta.append((0, j, 0))
            for k in range(shape[2]):
                if self.basis_index[0, 0, k] == 0:
                    temp_beta.append((0, 0, k))

            # B_2
            if self.core_size < shape[0]:
                index_0 = [int(c * np.floor(shape[0] / self.core_size)) for c in range(self.core_size)]
            else:
                index_0 = [c for c in range(shape[0])]

            if self.core_size < shape[1]:
                index_1 = [int(c * np.floor(shape[1] / self.core_size)) for c in range(self.core_size)]
            else:
                index_1 = [c for c in range(shape[1])]

            if self.core_size < shape[2]:
                index_2 = [int(c * np.floor(shape[2] / self.core_size)) for c in range(self.core_size)]
            else:
                index_2 = [c for c in range(shape[2])]

            for i in index_0:
                for j in index_1:
                    if self.basis_index[i, j, 0] == 0:
                        temp_beta.append((i, j, 0))
                for k in index_2:
                    if self.basis_index[i, 0, k] == 0:
                            temp_beta.append((i, 0, k))

        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        for c in range(len(temp_beta)):
            if self.basis_index[temp_beta[c]] == 0:
                beta.append(temp_beta[c])
                self.basis_index[temp_beta[c]] = 1

        return beta

    def _get_P_value(self, v):
        return self.P[v]

    def _gen_core(self, shape):
        """Generate set of decomposition basis B.

        Parameters
        ----------
        shape : int
            shapes of the input tensor P.

        Returns
        -------
        beta : list
            set of decomposition basis vectors.
        """
        order = len(shape)
        beta = []
        # B_3
        for i in range(shape[0]):
            temp_beta = []
            c_size = self.core_size
            if order == 2:
                for j in range(shape[1]):
                    if self.basis_index[i, j] == 0:
                        temp_beta.append((i, j))
            elif order == 3:
                for j, k in itertools.product(range(shape[1]), range(shape[2])):
                    if self.basis_index[i, j, k] == 0:
                        temp_beta.append((i, j, k))
            else:
                raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

            if self.shuffle:
                np.random.seed(seed=self.random_state)
                np.random.shuffle(temp_beta)
            else:
                temp_beta.sort(key=self._get_P_value)

            if len(temp_beta) < c_size:
                c_size = len(temp_beta)
            for c in range(c_size):
                if self.basis_index[temp_beta[c]] == 0:
                    beta.append(temp_beta[c])
                    self.basis_index[temp_beta[c]] = 1

        return beta

    def _gen_basis(self, shape):
        """Generate set of decomposition basis B,
        which are used for reconstructing tensor Q.
        This basis are parameters how decomposed input tensor P,
        thus the basis are directory related to complexity of models.

        Parameters
        ----------
        shape : int
            shapes of the input tensor P.

        Returns
        -------
        beta : list
            set of decomposition basis vectors.
        """
        self.basis_index = np.zeros(shape)
        beta = []
        # exclude all zero basis for a technical reason.
        self.basis_index[tuple(np.zeros(len(shape)).astype(int))] = 1

        if self.solver == 'ng':
            beta += self._gen_norm(shape)
        beta += self._gen_core(shape)

        return beta

    def _fit_gradient_descent(self, P, beta):
        r"""Compute parameter \theta using Gradient Descent-based optimization algorithms.

        The objective function is KL divergence(P, Q) and is minimized with
        Gradient Descent method.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta = self._initialize()
        self.eta_hat = self._compute_eta(P)
        self.res = 0.
        if self.verbose:
            print("\n\n============= theta =============")
            print(theta)
            print("\n\n============= eta_hat =============")
            print(self.eta_hat)

        for n_iter in range(self.max_iter):
            eta = self._compute_eta(self._compute_Q(theta, beta))
            if self.verbose:
                print("\n\n============= iteration: {}, eta =============".format(n_iter))
                print(eta)

            prev_res = self.res
            self.res = self._compute_residual(eta, beta)
            if self.verbose:
                print("n_iter: {}, Residual: {}".format(n_iter, self.res))

            # check convergence
            if (self.res <= self.tol) or (prev_res <= self.res and Constants.EPSILON.value <= prev_res):
                self.converged_n_iter = n_iter
                print("Convergence of theta at iteration: {}".format(self.converged_n_iter))
                break

            for v in beta:
                # \theta_v \gets \theta_v - \epsilon \times (\eta_v - \hat{\eta_v})
                grad = self._compute_eta(self._compute_Q(theta, beta)) - self.eta_hat
                theta[v] -= self.learning_rate * grad[v]

            if self.verbose:
                print("\n\n============= iteration: {}, theta =============".format(n_iter))
                print(theta)

        return theta

    def _fit_natural_gradient(self, P, beta):
        r"""Compute parameter \theta using Natural Gradient-based optimization algorithms.

        The objective function is KL divergence(P, Q) and is minimized with
        Natural Gradient method.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta = self._initialize()
        self.eta_hat = self._compute_eta(P)
        self.res = 0.
        theta_vec = np.array([theta[v] for v in beta])
        if self.verbose:
            print("\n\n============= theta =============")
            print(theta)
            print("\n\n============= eta_hat =============")
            print(self.eta_hat)

        for n_iter in range(self.max_iter):
            eta = self._compute_eta(self._compute_Q(theta, beta))
            if self.verbose:
                print("\n\n============= iteration: {}, eta =============".format(n_iter))
                print(eta)

            prev_res = self.res
            self.res = self._compute_residual(eta, beta)
            if self.verbose:
                print("n_iter: {}, Residual: {}".format(n_iter, self.res))

            # check convergence
            if (self.res <= self.tol) or (prev_res <= self.res and Constants.EPSILON.value <= prev_res):
                self.converged_n_iter = n_iter
                print("Convergence of theta at iteration: {}".format(self.converged_n_iter))
                break

            # compute \delta\eta and Fisher information matrix.
            delta_eta = eta - self.eta_hat
            eta_vec = np.array([delta_eta[v] for v in beta])
            G = self._compute_jacobian(eta, beta)
            if self.verbose:
                print("\n\n============= iteration: {}, delta_eta =============".format(n_iter))
                print(delta_eta)
                print("\n\n============= iteration: {}, eta_vec =============".format(n_iter))
                print(eta_vec)
                print("\n\n============= iteration: {}, G =============".format(n_iter))
                print(G)

            # TODO: Algorithm 7, Information Geometric Approaches for
            # Neural Network Algorithms to compute G inverse
            try:
                theta_vec -= np.dot(np.linalg.inv(G), eta_vec)
                # theta_vec -= np.linalg.solve(G, eta_vec)
                # theta_vec -= np.dot(linalg.inv(G), eta_vec)
            except:
                theta_vec -= np.dot(np.linalg.pinv(G), eta_vec)

            if self.verbose:
                try:
                    G_inv = np.linalg.inv(G)
                except:
                    G_inv = np.linalg.pinv(G)
                print("\n\n============= iteration: {}, G_inverse =============".format(n_iter))
                print(G_inv)
                print("\n\n============= iteration: {}, theta_vec =============".format(n_iter))
                print(theta_vec)

            # Update theta
            for n, v in enumerate(beta):
                theta[v] = theta_vec[n]

            if self.verbose:
                print("\n\n============= iteration: {}, theta =============".format(n_iter))
                print(theta)

        return theta

    def fit_natural_gradient(self, Q, beta):
        #TODO
        r"""Compute parameter \theta using Natural Gradient-based optimization algorithms.

        The objective function is KL divergence(P, Q) and is minimized with
        Natural Gradient method.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta = self._initialize()
        self.eta_hat = self._compute_eta(P)
        self.res = 0.
        theta_vec = np.array([theta[v] for v in beta])
        if self.verbose:
            print("\n\n============= theta =============")
            print(theta)
            print("\n\n============= eta_hat =============")
            print(self.eta_hat)

        for n_iter in range(self.max_iter):
            eta = self._compute_eta(self._compute_Q(theta, beta))
            if self.verbose:
                print("\n\n============= iteration: {}, eta =============".format(n_iter))
                print(eta)

            prev_res = self.res
            self.res = self._compute_residual(eta, beta)
            if self.verbose:
                print("n_iter: {}, Residual: {}".format(n_iter, self.res))

            # check convergence
            if (self.res <= self.tol) or (prev_res <= self.res and Constants.EPSILON.value <= prev_res):
                self.converged_n_iter = n_iter
                print("Convergence of theta at iteration: {}".format(self.converged_n_iter))
                break

            # compute \delta\eta and Fisher information matrix.
            delta_eta = eta - self.eta_hat
            eta_vec = np.array([delta_eta[v] for v in beta])
            G = self._compute_jacobian(eta, beta)
            if self.verbose:
                print("\n\n============= iteration: {}, delta_eta =============".format(n_iter))
                print(delta_eta)
                print("\n\n============= iteration: {}, eta_vec =============".format(n_iter))
                print(eta_vec)
                print("\n\n============= iteration: {}, G =============".format(n_iter))
                print(G)

            # TODO: Algorithm 7, Information Geometric Approaches for
            # Neural Network Algorithms to compute G inverse
            try:
                theta_vec -= np.dot(np.linalg.inv(G), eta_vec)
                # theta_vec -= np.linalg.solve(G, eta_vec)
                # theta_vec -= np.dot(linalg.inv(G), eta_vec)
            except:
                theta_vec -= np.dot(np.linalg.pinv(G), eta_vec)

            if self.verbose:
                try:
                    G_inv = np.linalg.inv(G)
                except:
                    G_inv = np.linalg.pinv(G)
                print("\n\n============= iteration: {}, G_inverse =============".format(n_iter))
                print(G_inv)
                print("\n\n============= iteration: {}, theta_vec =============".format(n_iter))
                print(theta_vec)

            # Update theta
            for n, v in enumerate(beta):
                theta[v] = theta_vec[n]

            if self.verbose:
                print("\n\n============= iteration: {}, theta =============".format(n_iter))
                print(theta)

        return theta

    def _legendre_decomposition(self, P):
        """Compute Legendre decomposition.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
            Used for reconstructing decomposable tensor Q.
        """
        self.shape = P.shape
        order = len(P.shape)
        if order not in (2, 3):
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        # normalize tensor
        self.P = self._normalizer(P)
        self.beta = self._gen_basis(self.shape)
        if self.verbose:
            print("\n\n============= beta =============")
            print(self.beta)

        if self.solver == 'ng':
            theta = self._fit_natural_gradient(self.P, self.beta)
        elif self.solver == 'gd':
            theta = self._fit_gradient_descent(self.P, self.beta)
        else:
            raise ValueError("Invalid solver {}.".format(self.solver))

        return theta


def main():
    np.random.seed(2020)
    P = np.random.rand(8, 5, 3)
    ld = LegendreDecomposition(solver='ng', max_iter=5)
    reconst_tensor = ld.fit_transform(P)
    print('Reconstruction error(RMSE): {}'.format(ld.reconstruction_err_))
    np.set_printoptions(threshold=200)
    print("\n\n============= Original Tensor =============")
    print(P)
    print("\n\n============= Reconstructed Tensor =============")
    print(reconst_tensor)

if __name__ == '__main__':
    main()