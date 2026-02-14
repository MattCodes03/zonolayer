import numpy as np
import torch
from .zonotope import Zonotope
import PyIPM


class Zonolayer:
    """
    Zonolayer: Last-layer uncertainty quantification via zonotopic representations.

    Learns an affine mapping from training outputs to test outputs,
    propagating interval uncertainties through this mapping to produce
    tight zonotope-based prediction intervals.

    Parameters:
    -----------
    centre_net : torch.nn.Module
        Neural network for center predictions. Must have a forward() method.
    lambda_reg : float (default=1e-6)
        Regularization parameter for the affine mapping.
    """

    def __init__(self, centre_net, lambda_reg: float = 1e-6):
        self.centre_net = centre_net
        self.lambda_reg = lambda_reg

    def _compute_zonotope_bounds(self, y_train_pred,
                                 y_lower, y_upper, y_test_pred, ipm):
        """
        Last-layer zonotopic uncertainty quantification.

        Method:
        -------
        1. Centers: Use neural network predictions (assumed accurate)
        2. Radii: Propagate training uncertainties through learned affine map

        Theoretical Guarantee:
        ---------------------
        If NN centers are exact and affine structure holds, bounds are EXACT.
        In practice, bounds are VALID (conservative) with small NN error.

        Complexity:
        ----------
        O(n_train * n_test) for affine map + O(n_test * n_train) for propagation
        Total: O(n_train * n_test), much cheaper than full GP or kernel methods.

        Returns:
        --------
        Prediction intervals [y_lower_pred, y_upper_pred] that:
        - Are centered at NN predictions
        - Have widths determined by zonotopic propagation
        - Are guaranteed to contain true outputs if assumptions hold
        """

        # Ensure correct shapes
        y_train_pred = np.atleast_1d(y_train_pred).reshape(-1)
        y_test_pred = np.atleast_1d(y_test_pred).reshape(-1)
        y_lower = np.atleast_1d(y_lower).reshape(-1)
        y_upper = np.atleast_1d(y_upper).reshape(-1)

        n_train = len(y_train_pred)

        # Zonotopic propagation of uncertainties
        Z_train = Zonotope.from_intervals(y_lower, y_upper)
        radii_train = Z_train.diagonal_generators

        # Learn affine transformation in output space
        Phi = np.column_stack([y_train_pred, np.ones(n_train)])
        Phi_test = np.column_stack([y_test_pred, np.ones(len(y_test_pred))])

        M = np.linalg.pinv(
            Phi.T @ Phi + self.lambda_reg * np.eye(Phi.shape[1])
        ) @ Phi.T
        A = Phi_test @ M

        # Exact zonotopic propagation through affine map
        radius_test = np.sum(np.abs(A) * radii_train, axis=1)

        # Final intervals
        y_lower_pred = y_test_pred - radius_test
        y_upper_pred = y_test_pred + radius_test

        if ipm:
            ipm_upper, ipm_lower = self._compute_ipm_bounds(
                y_train_pred, y_test_pred, y_lower, y_upper)
            return {
                "pred_centre": y_test_pred,
                "y_lower_pred": y_lower_pred,
                "y_upper_pred": y_upper_pred,
                "y_lower_ipm": ipm_lower,
                "y_upper_ipm": ipm_upper
            }
        else:
            return {
                "pred_centre": y_test_pred,
                "y_lower_pred": y_lower_pred,
                "y_upper_pred": y_upper_pred,
            }

    def _compute_ipm_bounds(self, x_train, x_test, y_lower, y_upper):
        ipm_model = PyIPM.IPM()

        x_train = np.asarray(x_train, dtype=np.float64).reshape(-1, 1)
        x_test = np.asarray(x_test, dtype=np.float64).reshape(-1, 1)
        y_lower = np.asarray(y_lower, dtype=np.float64).flatten()
        y_upper = np.asarray(y_upper, dtype=np.float64).flatten()

        # Combination of Endpoints
        x_train = np.vstack([x_train, x_train])
        y_train = np.concatenate([y_lower, y_upper])
        # y_train = np.column_stack([y_lower, y_upper])

        ipm_model.fit(x_train, y_train)

        return ipm_model.predict(x_test)

    def compute(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        ipm: bool = False
    ):
        """
        Compute prediction intervals for test data.

        Parameters
        ----------
        x_train : torch.Tensor
            Training feature matrix.
        x_test : torch.Tensor
            Test feature matrix.
        y_lower : np.ndarray
            Lower bounds of training target intervals.
        y_upper : np.ndarray
            Upper bounds of training target intervals.
        ipm : bool, optional (default=False)
            If True, prediction interval bounds are additionally
            computed using the IPM method (via PyIPM).

        Returns
        -------
        dict
            Dictionary containing:
                'pred_centre' : np.ndarray
                    Centers of predicted intervals (shape: n_test).
                'y_lower_pred' : np.ndarray
                    Predicted lower bounds (shape: n_test).
                'y_upper_pred' : np.ndarray
                    Predicted upper bounds (shape: n_test).

            If `ipm=True`, the dictionary additionally contains:
                'y_lower_ipm' : np.ndarray
                    IPM-computed lower bounds (shape: n_test).
                'y_upper_ipm' : np.ndarray
                    IPM-computed upper bounds (shape: n_test).
        """

        # Get predictions from the neural network
        self.centre_net.eval()
        with torch.no_grad():
            centre_train = self.centre_net(x_train).numpy().flatten()
            centre_pred = self.centre_net(x_test).numpy().flatten()

        # Ensure correct shapes for interval bounds
        y_lower = np.atleast_1d(y_lower).flatten()
        y_upper = np.atleast_1d(y_upper).flatten()

        # Compute zonotope-based intervals
        return self._compute_zonotope_bounds(
            y_train_pred=centre_train,
            y_lower=y_lower,
            y_upper=y_upper,
            y_test_pred=centre_pred,
            ipm=ipm,
        )
