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

    def _compute_zonotope_bounds(
        self,
        predicted_centre,
        y_train_pred,
        y_lower,
        y_upper,
        predicted_test,
    ):
        """
        Compute zonotope-based prediction intervals.

        Parameters:
        -----------
        predicted_centre : array-like
            Neural network predictions on test data (used as interval centers)
        y_train_pred : array-like
            Neural network predictions on training data
        y_lower : array-like
            Lower bounds of training intervals
        y_upper : array-like
            Upper bounds of training intervals
        predicted_test : array-like
            Neural network predictions on test data (same as predicted_centre)

        Returns:
        --------
        dict with keys:
            'pred_centre': Centers of predicted intervals
            'y_lower_pred': Lower bounds of predicted intervals
            'y_upper_pred': Upper bounds of predicted intervals
        """
        # Ensure correct shapes
        y_train_pred = np.atleast_1d(y_train_pred).reshape(-1)
        predicted_test = np.atleast_1d(predicted_test).reshape(-1)
        y_lower_train = np.atleast_1d(y_lower).reshape(-1)
        y_upper_train = np.atleast_1d(y_upper).reshape(-1)
        predicted_centre = np.atleast_1d(predicted_centre).reshape(-1)

        # 1. Create training zonotope from intervals (diagonal representation)
        Z_train = Zonotope.from_intervals(y_lower_train, y_upper_train)

        # 2. Learn affine mapping from training to test outputs
        Phi = y_train_pred.reshape(-1, 1)      # (n_train, 1)
        Phi_test = predicted_test.reshape(-1, 1)  # (n_test, 1)

        # Solve regularized least squares
        M = np.linalg.pinv(Phi.T @ Phi + self.lambda_reg) @ Phi.T
        A = Phi_test @ M  # (n_test, n_train)

        # 3. Propagate training uncertainties through affine map
        radii_train = Z_train.diagonal_generators
        radius_test = np.sum(np.abs(A) * radii_train, axis=1)

        # 4. Build intervals centered at NN predictions
        y_lower_pred = predicted_centre - radius_test
        y_upper_pred = predicted_centre + radius_test

        return {
            "pred_centre": predicted_centre,
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
