import numpy as np
import scipy
from scipy.linalg import sqrtm
from scipy.sparse import linalg

SUPPORTED_HANKEL_VECTORIZATIONS = ["coo", "dok", "csr"]


class SIDBase(object):
    """System Identification base class.

    Helper functions for linear system identification and common
    statistical methods for de-biasing the estimation. For example:

    1. Forward-backwards
    2. Total least-squares
    """

    def _construct_snapshots(self, snapshots, order, n_times):
        snaps = np.concatenate(
            [snapshots[:, i : n_times - order + i + 1] for i in range(order)],
            axis=0,
        )
        return snaps

    @staticmethod
    def _col_major_2darray(X):
        """Store snapshots into a 2D matrix, by column.

        If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        Parameters
        ----------
        X : int or numpy.ndarray
            the input snapshots.

        Returns
        -------
        snapshots : np.ndarray
            the 2D matrix that contains the flatten snapshots
        snapshots_shape : tuple
            the shape of original snapshots
        """
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError("Snapshots have not the same dimension.")

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])


        return snapshots, snapshots_shape

    def _solve_regression(
        self,
        X,
        Y,
        l2_penalty,
        solver="auto",
        fit_intercept=False,
        normalize=False,
        random_state=123456,
        sample_weight=None,
        ch_weight=None,
    ):
        from sklearn.linear_model import Ridge

        clf = Ridge(
            alpha=l2_penalty,
            fit_intercept=fit_intercept,
            normalize=normalize,
            solver=solver,
            random_state=random_state,
        )

        # n_samples X n_features and n_samples X n_targets
        clf.fit(X.T, Y.T, sample_weight=sample_weight)

        # n_targets X n_features
        A = clf.coef_
        return A

    def _pinv_solve(self, X, Y, l2_penalty, solver="svd", backend="numpy"):
        """Solves linear regression using pseudoinverse.

        Can either use standard pseudoinverse solve, or
        performs l2-regularization [1].

        Parameters
        ----------
        X : np.ndarray
            the first matrix;
        Y : np.ndarray
            the second matrix;
        l2_penalty : float | None
            The l2 penalty coefficient in L2 regularization.
        solver : str
            The solver strategy for linear regression. Either uses
            ``'lstsq'`` (calls :func:`np.linalg.pinv`, or :func:`np.linalg.lstsq`),
             or ``'svd'`` (uses SVD methods).
        backend : str
            Either ``'numpy'`` (default), or ``'scipy'``.

        Returns
        -------
        adjmat : np.ndarray
            The corresponding ``A`` matrix.

        References
        ----------
        .. [1] https://www.cs.princeton.edu/courses/archive/spring07/cos424/scribe_notes/0403.pdf  # noqa
        """
        if backend not in ["scipy", "numpy"]:
            msg = f'Only "scipy", or "numpy" backend is accepted, not {backend}.'
            raise ValueError(msg)
        elif backend == "scipy":
            backend_func = scipy
        elif backend == "numpy":
            backend_func = np

        n_chs = X.shape[0]

        if l2_penalty is None or l2_penalty == 0:
            if solver == "lstsq":
                adjmat = Y.dot(scipy.linalg.pinv(X))
            elif solver == "svd":
                # or use numpy linalg
                adjmat = Y.dot(backend_func.linalg.pinv(X))
        else:  # pragma: no cover

            # apply perturbation on diagonal of X^TX
            tikhonov_regularization = l2_penalty * np.eye(n_chs)

            if solver == "lstsq":
                adjmat = backend_func.linalg.solve(
                    X.T.dot(X) + tikhonov_regularization, X.T.dot(Y)
                )
            elif solver == "svd":
                inner_regularized = X.dot(X.T) + tikhonov_regularization
                _tikhonov_pinv_mat = X.T.dot(np.linalg.inv(inner_regularized))

                # solve analytical solution for regularized L2 regression
                adjmat = Y.dot(_tikhonov_pinv_mat)
        return adjmat

    @staticmethod
    def _compute_tlsq(X, Y, tlsq_rank):
        """Compute Total Least Square.

        Assumes the model is the following:
        ::

            Y = AX

        where ``X`` and ``Y`` are passed in arrays.

        Parameters
        ----------
        X : np.ndarray
            the first matrix;
        Y : np.ndarray
            the second matrix;
        tlsq_rank : int
            the rank for the truncation; If 0, the method
            does not compute any noise reduction; if positive number, the
            method uses the argument for the SVD truncation used in the TLSQ
            method.

        Returns
        -------
        X : np.ndarray
            the denoised matrix X,
        Y : np.ndarray
            the denoised matrix Y

        References
        ----------
        .. [1] https://arxiv.org/pdf/1703.11004.pdf
        .. [2] https://arxiv.org/pdf/1502.03854.pdf
        """
        # Do not perform tlsq
        if tlsq_rank == 0:
            return X, Y

        V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
        rank = min(tlsq_rank, V.shape[0])
        VV = V[:rank, :].conj().T.dot(V[:rank, :])

        return X.dot(VV), Y.dot(VV)


class PostHocMixin:
    """Mixin class for performing post-hoc modifications to identified systems."""

    @staticmethod
    def stabilize_matrix(A, eigval):
        """
        Stabilize the matrix, A based on its eigenvalues.

        Assumes discrete-time linear system.

        Parameters
        ----------
        A : np.ndarray
             CxC matrix
        eigval : float
            the maximum eigenvalue to shift all large evals to
        Returns
        -------
        A : np.ndarray
            the stabilized matrix w/ eigenvalues <= eigval
        """
        # check if there are unstable eigenvalues for this matrix
        if np.sum(np.abs(np.linalg.eigvals(A)) > eigval) > 0:
            # perform eigenvalue decomposition
            eigA, V = scipy.linalg.eig(A)

            # get magnitude of these eigenvalues
            abs_eigA = np.abs(eigA)

            # get the indcies of the unstable eigenvalues
            unstable_inds = np.where(abs_eigA > eigval)[0]

            # move all unstable evals to magnitude eigval
            for ind in unstable_inds:
                # extract the unstable eval and magnitude
                unstable_eig = eigA[ind]
                this_abs_eig = abs_eigA[ind]

                # compute scaling factor and rescale eval
                k = eigval / this_abs_eig
                eigA[ind] = k * unstable_eig

            # recompute A
            eigA_mat = np.diag(eigA)
            Aprime = np.dot(V, eigA_mat)
            A = scipy.linalg.lstsq(V.T, Aprime.T)[0].T
        return A



