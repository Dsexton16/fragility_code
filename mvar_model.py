import numpy as np
from scipy.linalg import sqrtm
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel

from sysid import SIDBase, PostHocMixin

SUPPORTED_SYSID_METHODS = ["pinv"]


def inner_transpose_multicompanion(mat, order):
    """Transpose inner matrices of a multi-companion matrix.

    ``mat`` comprises of a sequence of square matrices,
    ``A_1``, ..., ``A_order`` along with corresponding
    identity matrices where they should be. This function
    will transpose the ``A_1``, ..., ``A_order`` matrices.

    Parameters
    ----------
    mat : np.ndarray
    order : int

    Returns
    -------
    mat : np.ndarray
        Inner

    Notes
    -----
    ``mat`` has exactly (n_chs * order, n_chs * order) shape,
    so ``n_chs`` can be obtained from the shape of ``mat``.


    References
    ----------
    https://core.ac.uk/download/pdf/82792651.pdf
    """
    mat = mat.copy()  # create a copy
    n_chs = mat.shape[0] / order

    assert np.mod(mat.shape[0], order) == 0

    start_col = n_chs * (order - 1)
    start_row = 0
    for idx in range(order):
        grid = np.ix_(
            np.arange(start_row, start_row + n_chs, dtype=int),
            np.arange(start_col, start_col + n_chs, dtype=int),
        )
        # get the subset mat matrix
        submat = mat[grid]
        mat[grid] = submat.T
        start_row += n_chs
    return mat


class SystemIDModel(
    SIDBase,
    PostHocMixin,
    LinearModel,
    MultiOutputMixin,
    RegressorMixin,
):
    """Multivariate-autoregressive style algorithm used for estimating a linear system.

    The model is of the style:

        $x(t+1) = Ax(t)$

    The algorithm takes in a window of data that is CxT, where C is typically the
    number of channels and T is the number of samples for a specific window to generate
    the linear system.

    It uses either:
     1. a dynamic mode decomposition SVD based algorithm,
     2. SVD truncated algorithm
     3. Pseudoinverse with regularization

    to estimate A.

    Attributes
    ----------
    l2penalty : float
        The l2-norm regularization if 'regularize' is turned on. Only used
        if ``method_to_use`` is ``'pinv'``.
    method_to_use : str
        svd as the method to compute A matrix
    svd_rank : float
        Passed to :func:`eztrack.embed.svd.computeSVD`. If ``None``,
        will be the number of channels in the data matrix.
        Only used if ``method_to_use`` is ``'svd'``.
    tlsq_rank : int
        rank truncation computing Total Least Square. Default
        is 0, that means no truncation. See [1].
    fb : bool
        Whether to apply the Forward-Backwards algorithm. See Notes and [2].
    order : int
        The order of the model to estimate (default=1).
    solver : str
        Only used if ``method_to_use='sklearn'``. See :func:`sklearn.linear_model.Ridge` parameters.
    fit_intercept : bool
        Only used if ``method_to_use='sklearn'``. See :func:`sklearn.linear_model.Ridge` parameters.
    normalize : bool
        Only used if ``method_to_use='sklearn'``. See :func:`sklearn.linear_model.Ridge` parameters.

    Notes
    -----
    When the size of the data is too large (e.g. N > 180, W > 1000), then right now the construction of the csr
    matrix scales up. With more efficient indexing, we can perhaps decrease this.

    References
    ----------
    .. [1] De-biasing the dynamic mode decomposition for applied Koopman
        spectral analysis of noisy datasets. https://arxiv.org/pdf/1703.11004.pdf
    .. [2] Characterizing and correcting for the effect of sensor noise in the
        dynamic mode decomposition. https://arxiv.org/pdf/1507.02264.pdf

    """

    def __init__(
        self,
        l2penalty=0.0,
        method_to_use="pinv",
        svd_rank=None,
        tlsq_rank=0,
        fb: bool = False,
        solver="auto",
        fit_intercept=False,
        normalize=False,
        order=1,
        weighted: bool = False,
    ):
        super(SystemIDModel, self).__init__()
        if method_to_use not in SUPPORTED_SYSID_METHODS:
            raise AttributeError(
                f"System ID with {method_to_use} "
                f"is not supported. Please use "
                f"one of {SUPPORTED_SYSID_METHODS}."
            )

        self.l2penalty = l2penalty
        self.method_to_use = method_to_use
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.fb = fb
        self.order = order
        self.weighted = weighted

        self.solver = solver
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    @property
    def state_array(self):
        """The model tensor CxC samples by chan by chan."""
        return self.state_array_

    @property
    def eigs(self):
        return np.linalg.eigvals(self.state_array)

    def _forward_multiply(self, X, t_steps):
        # 2D array of initial conditions (C x samples)
        n_samples, n_chs = X.shape

        X_pred = []
        # generate predictions for many initial conditions
        for isamp in range(n_samples):
            x0 = X[isamp, :]
            X_hat = [x0]

            # reconstruct over possibly multiple time steps
            for it in range(t_steps - 1):
                X_hat.append(self.state_array.dot(X_hat[-1]))

            # store the predictions
            X_hat = np.array(X_hat).T
            X_pred.append(X_hat)
        # print('inside forward multiply...', np.array(X_pred).shape, n_samples, t_steps)
        return np.asarray(X_pred)

    def predict(self, X, t_steps: int = 1):
        """Predict state dynamics from initial conditions as a 2D array.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Samples of initial conditions. ``n_features`` should match the
            number of channels used in the data to fit model (in ``fit()``).
        t_steps : int
            The number of time steps to predict. Default = 1.

        Returns
        -------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        """
        check_is_fitted(self)
        X = check_array(
            X, dtype=[np.float64, np.float32], ensure_2d=True, allow_nd=False
        )

        # forward multiply linear operator
        y_pred = self._forward_multiply(X=X, t_steps=t_steps)
        return y_pred

    def score(self, X, y, sample_weight=None, t_steps=1):
        """Score fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix or a list of generic objects
             instead with shape (n_samples, n_samples_fitted),
             where n_samples_fitted is the number of samples
             used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        t_steps : int
            The number of time steps to predict. Default = 1.

        Returns
        -------
        score : float
            R^2 of self.predict(X) w.r.t. y.
        """
        from sklearn.metrics import r2_score

        check_is_fitted(self)

        X, y = self._validate_data(
            X, y, y_numeric=True, dtype=[np.float64, np.float32], multi_output=True
        )
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        n_samples, _ = X.shape

        # run predictions for possibly multiple initial conditions
        y_pred = self.predict(X, t_steps=t_steps)


        # get the state prediction and score it
        scores = []
        for idx in range(n_samples):
            X_pred = y_pred[idx, :, :]
            X_true = y[idx, :]

            score = r2_score(X_true, X_pred, sample_weight=sample_weight)
            scores.append(score)
        print("Finally")
        return np.mean(scores)

    def fit(self, X, y=None):
        """
        Generate adjacency matrix for each window for a certain winsize and stepsize.

        Sets the ``state_array`` property, which is the linear operator that
        operates over the data matrix ``X``.

        Additional hyperparameters are defined in the ``__init__`` function.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Samples of multivariate data. ``n_features`` should match the
            number of channels used in the data . ``n_samples`` should match
            the number of time points in the data window to regress.

        Notes
        -----
        This function will fit a linear operator, ``A`` on the data as such::

            X[1:, :] = X[:-1, :].dot(A)

        representing the equation ``x(t+1) = Ax(t)``.
        """
        # check data
        if y is None:
            X = self._validate_data(X)
        else:
            check_kwargs = dict(
                dtype=[np.float64, np.float32],
                y_numeric=True,
                force_all_finite=True,
                multi_output=True,
            )
            X, y = self._validate_data(X, y, **check_kwargs)

        # check condition number of the array
        cond_num = np.linalg.cond(X)

        # 1. determine shape of the window of data
        n_times, n_chs = X.shape

        # determine svd_rank if it is None
        if self.svd_rank is None:
            svd_rank = min(n_chs, n_times)
        else:
            svd_rank = self.svd_rank

        # allow for multiple order matrix estimation
        # first make sure data matrix is 2D
        # also swap channels and times to make it backwards compatible
        # with our change to sklearn API
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X.T)

        # create large snapshot with time-lags of order specified by
        # ``order`` value
        snaps = self._construct_snapshots(
            self._snapshots, order=self.order, n_times=n_times
        )
        X, Y = snaps[:, :-1], snaps[:, 1:]

        # compute total-least squares if necessary
        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        # compute inverse of variance weights
        if self.weighted:
            # compute along the samples dimension
            sample_var = np.var(X, axis=1)
            sample_weight = 1.0 / sample_var

            eps = 1e-8
            if np.any(sample_var < eps):
                sample_weight = None
        else:
            sample_weight = None

        # compute the forward linear operator from X -> Y
        if self.method_to_use == "pinv":
            A = self._pinv_solve(X, Y, self.l2penalty, solver="svd", backend="numpy")

        # if using forward-backwards algorithm, then
        # compute the backwards operator and combine the two
        if self.fb:
            # compute backward linear operator
            bA = self._pinv_solve(Y, X, l2_penalty=self.l2penalty)
            A = sqrtm(A.dot(np.linalg.inv(bA)))

        self.state_array_ = A
        return self

