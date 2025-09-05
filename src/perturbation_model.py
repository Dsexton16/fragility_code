from typing import Union, List
import numpy as np
import scipy

import utils.dataconfig as constants

SUPPORTED_PERTURBATION_OPT_METHODS = ["dc", "grid"]
PERTURBATION_STRATEGIES = ["univariate", "bivariate"]


def ensure_list(arg):
    if not (isinstance(arg, list)):
        try:  # if iterable
            if isinstance(arg, (str, dict)):
                arg = [arg]
            elif arg is None:
                arg = []
            else:
                arg = list(arg)
        except BaseException:  # if not iterable
            arg = [arg]
    return arg



def _create_standard_basis_vector(size, index, as_vector=False):
    arr = np.zeros(size)
    arr[index] = 1.0
    if as_vector:  # turn into 1 vector in R^d, not a list
        arr = arr[:, np.newaxis]
    return arr


def compute_brauer_rank_one(
    A: np.ndarray, radius: complex, eigval_idx: int = 0, rank: int = 1
):
    """Applies Brauer's rank one perturbation algorithm.

    Perturbs the leading eigenvalue of matrix A to obtain eigenvalue
    with value at ``radius``. Will perform a perturbation on the
    eigenvalue at index ``eigval_idx``, which is the index after
    sorting the eigenvalues from largest to smallest in absolute value.

    Parameters
    ----------
    %(A)s
    %(radius)s
    eigval_idx : int
        The eigenvalue to apply perturbation to.
    rank : int
        The rank of the perturbation to apply. (Default = 1).
        Results in ``rank`` number of eigenvalues that are
        changed.

    Returns
    -------
    delta_vec : np.ndarray
        The (N, N) rank-1 matrix that when applied to ``A`` will
        result in an eigenvalue with value ``radius``.

    References
    ----------
    .. [1] A. Brauer, Limits for the characteristic roots of a matrix IV: Applications to stochastic matrices, Duke Math. J. 19
        (1952) 75–91.
    .. [2] H. Perfect, Methods of constructing certain stochastic matrices II, Duke Math. J. 22 (1955) 305–311.
    """
    # perform eigenvalue decomposition of A
    W, V = scipy.linalg.eig(A)

    # get max to min of eigenvalues
    sorted_inds = np.argsort(W)[::-1]

    # take eigenvalue as the first entry
    eigval = W[sorted_inds[eigval_idx]]
    eigvec = V[:, sorted_inds[eigval_idx]]

    # compute the perturbation of lambda
    eigval_diff = radius - eigval

    # format into 2D arrays to perform least squares
    eigvec_arr = eigvec[None, :]
    eigval_diff_arr = np.array([radius - eigval])

    # solve least squares sense
    eigval_perturbation_vec, res, rnk, s = scipy.linalg.lstsq(
        eigvec_arr, eigval_diff_arr
    )
    eigval_perturbation = eigval_perturbation_vec.conj().T.dot(eigvec)
    assert np.allclose(eigval_diff, eigval_perturbation)

    # compute the delta vector
    delta_vec = np.outer(eigvec, eigval_perturbation_vec.conj().T)

    return delta_vec


def compute_bauer_fike_bound(A, E, p=2):
    """Compute the Bauer-Fike bound.

    Computes the bound of eigenvalue movement
    based on the condition number of the
    eigenvector matrix, and the p-norm of the
    perturbation matrix.

    Parameters
    ----------
    A : np.ndarray
        The matrix. Must be diagonalizable.
    E : np.ndarray
    p : str | int
        The p-norm. Corresponds to to the ``'p'``
        in :func:`np.linalg.cond` and the
        ``'ord'`` in :func:`np.linalg.norm`.

    Returns
    -------
    bound : float
        The Bauer-Fike theorem bound.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bauer%E2%80%93Fike_theorem
    """
    # compute eigenvalue decomposition of A
    W, V = scipy.linalg.eig(A)

    # compute the condition number of V
    cond_num = np.linalg.cond(V, p=p)

    # compute the norm of the perturbation matrix
    pert_norm = np.linalg.norm(E, ord=p)

    return cond_num * pert_norm


class StructuredPerturbationModel(object):
    r"""Structured rank-1 perturbations of column/rows of a matrix.

    :math:`x(t+1) = (A + {\\Delta}) x(t)`

    The algorithm takes in a np.ndarray matrix of data that is NxN and computes the
    minimum 2-norm required to have one eigenvalue of the matrix A at "r".

    Parameters
    ----------
    %(radius)s
    %(perturb_type)s
    %(perturbation_strategy)s
    %(n_jobs)s
    %(verbose)s

    Notes
    -----
    Model can either perform grid search over all possible frequencies of perturbation (i.e. placements
    along the circle specified by "radius". Application at only the DC frequency (a+jb = radius; b = 0)
    seems to work sufficiently and is only one search.
    """

    def __init__(
        self,
        radius: Union[float, str],
        perturb_type: str,
        perturbation_strategy: str = "univariate",
        n_jobs: int = 1,
        on_error: str = "raise",
        verbose: bool = False,
    ):
        # ensure perturbation type is capitalized
        perturb_type = perturb_type.upper()

        if perturb_type not in [
            constants.PERTURBATIONTYPES.COLUMN_PERTURBATION.value,
            constants.PERTURBATIONTYPES.ROW_PERTURBATION.value,
        ]:
            msg = (
                "Perturbation type can only be {}, or {} for now. You "
                "passed in {}.".format(
                    constants.PERTURBATIONTYPES.COLUMN_PERTURBATION,
                    constants.PERTURBATIONTYPES.ROW_PERTURBATION,
                    perturb_type,
                )
            )
            raise AttributeError(msg)
        if perturbation_strategy not in PERTURBATION_STRATEGIES:
            msg = (
                f"Only perturbation strategies {PERTURBATION_STRATEGIES} "
                f"are accepted, not {perturbation_strategy}."
            )
            raise AttributeError(msg)

        self._radius = radius
        self._perturb_type = perturb_type
        self.perturbation_strategy = perturbation_strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.on_error = on_error

        self._min_frequency = None
        self._delta_vec_arr = None
        self._min_norms = None

    def __str__(self):
        """Representation of model and its parameters."""
        return (
            f"Structured Perturbation Model | "
            f"radius={self.radius}, perturb_type={self.perturb_type}"
        )

    @property
    def radius(self):
        return self._radius

    @property
    def perturb_type(self):
        return self._perturb_type

    @property
    def minimum_freqs(self) -> Union[List, int]:
        """Corresponding frequencies that minimum norm perturbations occur."""
        return self._min_frequency

    @property
    def minimum_delta_vectors(self) -> np.ndarray:
        """Corresponding perturbation vectors with minimum norm.

        The delta vectors are N x N, where N is the number of channels.
        The rows correspond to the perturbation vector. For example,
        ``self._delta_vec_arr[0, :]``
        """
        return self._delta_vec_arr

    @property
    def minimum_norms_vector(self):
        """Compute the l2 norms of the minimum-norm delta vectors.

        This will result in a N x 1 vector.
        """
        return np.linalg.norm(self.minimum_delta_vectors, ord=2, axis=1)

    def _compute_min_norm_delta_vec(
        self,
        A: np.ndarray,
        radius: complex,
        eks: List[int],
        orthogonal_constraints: List[int] = None,
    ):
        r"""Compute a $\Delta$ matrix that has minimum 2-norm.

        Based on the radius (``r = a + ib``, where ``a`` is the real
        component and ``b`` is the imaginary component) on the complex plane,
        a matrix, $\Delta$ is computed such that $A + \Delta$ has an
        eigenvalue at ``radius``.

        The $\Delta$ matrix is a rank-1 matrix that comprises of the same
        vector at the column/row indices in ``eks`` list.

        Note that the $\Delta$ matrix may be generalized to have multiple
        non-zero columns, depending on the passed in indices ``eks``.
        The perturbation is either a Row, or Column.

        Parameters
        ----------
        %(A)s
        radius : np.complex
            The perturbation radius (``np.abs(radius)``), which may have a real and
            imaginary component.
        eks : List[int]
            A list of the column/row indices to compute a perturbation on.
        %(orthogonal_constraints)s

        Returns
        -------
        delta_vec : np.ndarray
            The perturbation vector that is applied at rows/columns in ``eks``,
            which as size ``(n_chs,)``.

        Notes
        -----
        To re-compute the $\Delta$ matrix, one would add the ``delta_vec``
        vector along the rows/columns of a (n_chs X n_chs) zero matrix.
        For example: ..

            delta_mat = np.zeros((n_chs, n_chs))

            for edx in eks:
                # if we used a row perturbation
                delta_mat[edx, :] = delta_vec

                # if we used a column perturbation
                delta_mat[:, edx] = delta_vec

            # now the perturbed A matrix should have
            # eigenvalue at location radius
            print(np.linalg.eigs(A + delta_mat))

        """
        if orthogonal_constraints is None:
            orthogonal_constraints = []
        perturb_type = self._perturb_type

        # ensure all indices of the unit vector are a list
        eks = ensure_list(eks)

        if isinstance(A, np.ndarray):
            if A.shape[0] != A.shape[1]:
                msg = (
                    f"State matrix must be a square matrix. The state matrix "
                    f"passed in has shape {A.shape}."
                )
                raise AttributeError(msg)

        # initialize function parameters
        len_perturbation = A.shape[0]

        # initialize vector to desired eigenvalue
        sigma = radius.real
        omega = radius.imag
        desired_eig = sigma + 1j * omega

        # generate list of the unit vectors
        unit_vecs = [
            _create_standard_basis_vector(len_perturbation, idx, as_vector=True)
            for idx in eks
        ]

        # generate array of constraints
        constraint_vec = np.array([0, -1])
        bvec = np.tile(constraint_vec, len(unit_vecs))

        # generate the "characteristic" equation for desired eigenvalue
        characteristic_eqn = A - desired_eig * np.eye((len_perturbation))

        # generate the data matrix for running least squares
        Hmat = []  # will be 2*len(indices) X n_chs
        for idx, ek in enumerate(unit_vecs):
            # determine if to compute row, or column perturbation
            if (
                perturb_type == constants.PERTURBATIONTYPES.ROW_PERTURBATION.value
            ):  # C = inv(A)*ek
                Cmat = np.linalg.lstsq(characteristic_eqn, ek, rcond=-1)[0]
            elif perturb_type == constants.PERTURBATIONTYPES.COLUMN_PERTURBATION.value:
                Cmat = np.linalg.lstsq(characteristic_eqn.T, ek, rcond=-1)[0].T

            # extract real and imaginary components to create vector of constraints
            Cimat = np.imag(Cmat)
            Crmat = np.real(Cmat)

            # store these in vector
            Hmat.append(Cimat)
            Hmat.append(Crmat)
        Hmat = np.array(Hmat).squeeze()
        # original least squares w/ constraints
        if np.imag(desired_eig) == 0:
            Bmat = Hmat[1::2, :]
            bvec = bvec[1::2]
        else:
            Bmat = Hmat
            bvec = bvec.copy()

        # adding orthogonality constraints
        # for each orthogonal constraint will make sure the Delta vector(s) are
        # 0 at that index. For example, if Delta vector(s) are bi-columnar, then
        # orthogonal_constraints of 0 and 50 would make sure the Delta vector
        # has value 0 at indices 0 and 50 in both columns
        if len(orthogonal_constraints) > 0:
            for ekidx in orthogonal_constraints:
                ek = _create_standard_basis_vector(
                    len_perturbation, ekidx, as_vector=True
                )

                # orthogonality constraints based on row/column
                if (
                    self.perturbtype
                    == constants.PERTURBATIONTYPES.ROW_PERTURBATION.value
                ):
                    Orthmat = ek
                    Bmat = np.concatenate((Bmat.T, Orthmat), axis=1).T
                    bvec = np.hstack((bvec, [0]))
                else:
                    Orthmat = ek.T
                    Bmat = np.concatenate((Bmat, Orthmat), axis=0)
                    bvec = np.hstack((bvec, [0])).T

        # compute the delta vector that will solve the argmin problem
        # print(Bmat.shape)
        # print(np.linalg.cond(Bmat))
        # print(np.linalg.svd(Bmat))
        delvec = np.linalg.pinv(Bmat, rcond=-1).dot(bvec)

        # return a 1D vector (real component). Note the
        # imaginary component is 0
        return delvec.squeeze()

    def compute_perturbations_at_indices(
        self,
        A: np.ndarray,
        radius: Union[complex, float, str],
        perturbation_strategy: str,
        n_perturbations: int = None,
        start: int = None,
        orthogonal_constraints=None,
    ):
        """Compute single column/row perturbation of an ``A`` matrix.

        Note the ``perturb_type`` is set on instantiation of the method.

        Parameters
        ----------
        %(A)s
        radius : np.complex
            The perturbation radius (``np.abs(radius)``), which may have a real and
            imaginary component.
        %(perturbation_strategy)s
        n_perturbations : int
            The number of channels in the original dataset.
        %(orthogonal_constraints)s

        Returns
        -------
        minperturbnorm : np.ndarray
            Norms of the perturbation vectors (n_chs, 1).
        delvecs_node : np.ndarray
            Delta perturbation vectors (n_chs, n_chs). Each row
            is a new perturbation vector.

        Notes
        -----
        ``univariate`` perturbation strategy:
        Perturbs ``A`` with a delta vector at one column/row indices at a time
        to obtain an eigenvalue with value at ``radius``. This will compute
        a delta vector and corresponding norms of the delta vectors over
        all columns, or rows one at a time.

        ``bivariate`` perturbation strategy:
        Perturbs ``A`` with a delta vector at two column/row indices at a time
        to obtain an eigenvalue with value at ``radius``. This will compute
        a delta vector and corresponding norms of the delta vectors over
        all columns, or rows one at a time.
        """
        if n_perturbations is None:
            n_chs = A[0].shape[0]
            if self.perturbation_strategy == "univariate":
                n_perturbations = n_chs
            elif self.perturbation_strategy == "bivariate":
                n_perturbations = n_chs // 2
        n_perturbations = int(n_perturbations)
        # initialize array to store all the minimum euclidean norms
        minperturbnorm = np.zeros((n_perturbations,))

        # initialize NxN array to store the delta vectors with each row being
        # a delta vector computed for that specific `ek` unit vector
        delvecs_node = np.zeros((n_perturbations, A.shape[0]))


        if start is None:
            start = 0

        for idx, ek in enumerate(range(n_perturbations)):
            if perturbation_strategy == "univariate":
                ek = [ek + start]
            elif perturbation_strategy == "bivariate":
                ek = [ek, ek + (n_perturbations // 2)]  # + start

            delta_vec = self._compute_min_norm_delta_vec(
                A, radius, ek, orthogonal_constraints=orthogonal_constraints
            )

            # store the l2 norm of the perturbation vector
            min_norm = np.linalg.norm(delta_vec)
            # store the minimum norm perturbation and vector
            minperturbnorm[idx] = min_norm
            # store the vector corresponding to minimum norm perturbation
            delvecs_node[idx, :] = delta_vec

        self._delta_vec_arr = delvecs_node
        self._min_frequency = 0
        return minperturbnorm, delvecs_node

    def fit(self, A: np.ndarray, **kwargs):
        r"""Compute and setup of the least squares soln.

        Assumes, given an A matrix, |lambda| value, and unit vector ek.

        Parameters
        ----------
        %(A)s
        %(orthogonal_constraints)s

        Returns
        -------
        minperturbnorm : np.ndarray
            the l2-norm of the perturbation vector
        """
        radius = self.radius
        perturbation_strategy = self.perturbation_strategy

        if isinstance(A, np.ndarray):
            if A.shape[0] != A.shape[1]:
                msg = (
                    f"State matrix must be a square matrix. The state matrix "
                    f"passed in has shape {A.shape}."
                )
                raise AttributeError(msg)

        e_val = np.abs(np.linalg.eigvals(A)).max()
        if radius == "adaptive":
            # get the spectral radii of Amat
            radius_ = e_val + 0.1
        else:
            radius_ = radius
        if e_val >= np.abs(radius_):
            msg = (
                f"The largest eigenvalue of A has "
                f"absolute value of {e_val}, so "
                f"perturbation to {np.abs(radius)} "
                f"is ill-defined."
            )
            if self.on_error == "raise":
                raise ValueError(msg)

        # make sure A passed in is a list of A matrices
        # if passed in a list, assume it is an order p model
        if isinstance(A, np.ndarray):
            A = [A]

        # check that all A's have same shape
        assert all([x.shape == A[0].shape for x in A])

        if self.perturbation_strategy == "bivariate":
            assert np.mod(A[0].shape[0], 2) == 0

        # check the order of the model imposed
        # from state estimation
        order = len(A)

        # perturbation norms and delta vector list
        min_norms = []
        delta_vecs_list = []

        # run lti perturbation method
        for Amat in A:
            # This method, only perturbs at :math:`\omega=0`
            minperturbnorm, delta_vecs = self.compute_perturbations_at_indices(
                Amat,
                radius=radius_,
                perturbation_strategy=perturbation_strategy,
                **kwargs,
            )
            min_norms.append(minperturbnorm)
            delta_vecs_list.append(delta_vecs)
            self._min_norms = min_norms
            self._delta_per_node_arr = delta_vecs_list

        if order == 1:
            return min_norms[0]
        return min_norms


class ConjugateStructurePerturbation(StructuredPerturbationModel):
    """Structured rank-1 perturbations that occur over the entire complex disc.

    Compared to ``StructuredPerturbationModel``, this model will
    perturb over all eigenvalues with absolute value of ``radius``.
    The number of points to perturb to will be set by ``mesh_size``.

    For example, if ``radius=1.5`` and ``mesh_size=3``, then between
    ``r=1.5`` to ``r=-1.5``, there will be 51 points discretized along
    the complex disc with radius of 1.5. They will be ``r=1.5j``. We
    do not need to perturb at the conjugate pairs, since every
    real perturbation vector on a real matrix will produce conjugate
    pairs. Therefore we only apply perturbations on the top half
    of the complex disc.

    Parameters
    ----------
    %(radius)s
    %(perturb_type)s
    mesh_size : int
        Number of discrete points (default=51) on the real/imaginary
        circle w/ ``np.abs(radius)`` to search over.
    %(perturbation_strategy)s
    %(n_jobs)s
    %(verbose)s

    """

    def __init__(
        self,
        radius: float,
        perturb_type: str,
        mesh_size: int,
        perturbation_strategy: str = "univariate",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super(ConjugateStructurePerturbation, self).__init__(
            radius=radius,
            perturb_type=perturb_type,
            perturbation_strategy=perturbation_strategy,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self._mesh_size = mesh_size

    @property
    def mesh_size(self):
        return self._mesh_size

    def _compute_gridsearch_perturbation(self, A, searchnum: int = 51):
        r"""
        Generate the minimum norm perturbation model.

        This method, does a grid search over combinations of real, or imaginary eigenvalues
        that have magnitude of "radius". And perturbs at :math:`r = \sigma + \omega j`

        Reference:
        - Sritharan 2014, et al.
        - Li 2017, et al.

        Parameters
        ----------
        %(A)s
        searchnum : int
            the number of discrete points to split the circle w/ specified radius into

        Returns
        -------
        (minperturbnorm, delvecs_node) : tuple[float, np.ndarray]

        minperturbnorm : float
            the l2 norm of the perturbation vector
        delvecs_node: np.ndarray
            [c, c'] is an array of perturbation vectors that achieve minimum norm at each row

        """
        # initialize search parameters to compute minimum norm perturbation
        top_wspace = np.linspace(-self.radius, self.radius, num=searchnum)
        wspace = np.append(top_wspace, top_wspace[1 : len(top_wspace) - 1], axis=0)
        sigma_space = np.sqrt(self.radius ** 2 - top_wspace ** 2)
        sigma_space = np.append(
            -sigma_space, sigma_space[1 : len(top_wspace) - 1], axis=0
        )

        # N x N, A matrix
        numchans, _ = A.shape

        # to store all the euclidean norms of the perturbation vectors
        # frequency x channel (F x N)
        minnorm_mat = np.zeros((wspace.size, numchans))

        # to store all the euclidean norms of the perturbation vectors
        # frequency x perturbed index x channel (F x N x N)
        delvecs_mat = np.zeros((wspace.size, numchans, numchans), dtype="complex")

        # delvecs_list = []

        # to store all the frequencies at which perturbation is applied
        freqs_mat = np.zeros((wspace.size, 1), dtype="complex")

        # compute the min-norm perturbation for omega specified
        for i, (sigma, omega) in enumerate(zip(sigma_space, wspace)):
            # run lti perturbation method for all contacts
            perturbation_results = self._compute_perturbation_on_system(A, sigma, omega)
            min_norm_perturb, delvecs_node = perturbation_results

            # find index of minimum norm perturbation for channel
            minnorm_mat[i, ...] = min_norm_perturb
            # delvecs_list.append(delvecs_node)
            delvecs_mat[i, ...] = delvecs_node.squeeze()
            freqs_mat[i] = sigma + 1j * omega

        # store the minimum norms that are achievable for each contact
        # -> allows for sensitivity to different frequencies, although majority will be
        # sensitive to omega == 0
        # determine argmin of the perturbation norms over all frequencies
        min_indices = np.argmin(minnorm_mat, axis=0)

        # get the minimum perturbation norms, frequencies,
        # and delta vectors
        min_norm_perturb = np.min(minnorm_mat, axis=0)
        min_freqs = sigma_space[min_indices] + 1j * wspace[min_indices]
        # delvecs_node = [delvecs_list[i] for i in min_indices]
        print(
            f"The minimum indices of {minnorm_mat.shape} "
            f"has {len(min_indices)} indices."
        )
        print("Shape of actual minimum norm perturbation ", min_norm_perturb.shape)
        print("Shape of minimum norm perturbation freqs ", min_freqs.shape)
        print("Delta vecs node squeezed: ", delvecs_node.squeeze().shape)
        print(delvecs_mat.shape)
        delvecs_node = delvecs_mat[min_indices, ...]

        self._min_frequency = min_freqs
        self._delta_per_node_arr = delvecs_node
        self._min_norms = min_norm_perturb
        return min_norm_perturb, delvecs_node

    def fit(self, A: np.ndarray, orthogonal_constraints: List[int] = None):
        """Compute perturbation of a square matrix.

        Parameters
        ----------
        %(A)s
        %(orthogonal_constraints)s

        Returns
        -------
        minperturbnorm : np.ndarray
            the l2-norm of the perturbation vector
        """
        pass
