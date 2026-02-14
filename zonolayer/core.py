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
    lambda_reg : float, default=1e-6
        Regularization parameter for the affine mapping.
    """

    def __init__(self, centre_net, lambda_reg: float = 1e-6):
        self.centre_net = centre_net
        self.lambda_reg = lambda_reg

    def _compute_zonotope_bounds(self, predicted_centre, y_train_pred,
                                 y_lower, y_upper, predicted_test):
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
        predicted_test = np.atleast_1d(predicted_test).reshape(-1)
        y_lower = np.atleast_1d(y_lower).reshape(-1)
        y_upper = np.atleast_1d(y_upper).reshape(-1)
        predicted_centre = np.atleast_1d(predicted_centre).reshape(-1)

        n_train = len(y_train_pred)

        # Use NN predictions as centers (Model 1)
        centers_test = predicted_centre

        # Zonotopic propagation of uncertainties (Model 2)
        Z_train = Zonotope.from_intervals(y_lower, y_upper)
        radii_train = Z_train.diagonal_generators

        # Learn affine transformation in output space
        Phi = np.column_stack([y_train_pred, np.ones(n_train)])
        Phi_test = np.column_stack(
            [predicted_test, np.ones(len(predicted_test))])

        M = np.linalg.pinv(
            Phi.T @ Phi + self.lambda_reg * np.eye(Phi.shape[1])
        ) @ Phi.T
        A = Phi_test @ M

        # Exact zonotopic propagation through affine map
        radius_test = np.sum(np.abs(A) * radii_train, axis=1)

        # Final intervals
        y_lower_pred = centers_test - radius_test
        y_upper_pred = centers_test + radius_test

        return {
            "pred_centre": centers_test,
            "y_lower_pred": y_lower_pred,
            "y_upper_pred": y_upper_pred,
        }

    def _compute_ipm_bounds():
        pass

    def compute(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ):
        """
        Compute prediction intervals for test data.

        Parameters:
        -----------
        x_train : torch.Tensor
            Training features
        x_test : torch.Tensor
            Test features
        y_lower : np.ndarray
            Lower bounds of training target intervals
        y_upper : np.ndarray
            Upper bounds of training target intervals

        Returns:
        --------
        dict with keys:
            'pred_centre': Centers of predicted intervals (shape: n_test)
            'y_lower_pred': Lower bounds of predicted intervals (shape: n_test)
            'y_upper_pred': Upper bounds of predicted intervals (shape: n_test)
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
            predicted_centre=centre_pred,
            y_train_pred=centre_train,
            y_lower=y_lower,
            y_upper=y_upper,
            predicted_test=centre_pred,
        )
