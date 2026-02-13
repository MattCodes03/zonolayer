import numpy as np
import torch
from scipy.stats import t
from .zonotope import Zonotope
import PyIPM


class Zonolayer:
    """
    Zonolayer: Last-layer uncertainty modeling via zonotopic representations.

    Fits a last-layer affine transformation to interval-bounded data,
    producing tight zonotope output bounds.
    """

    def __init__(self, centre_net, lambda_reg: float = 1e-6):
        self.centre_net = centre_net
        self.lambda_reg = lambda_reg

    def _compute_zonotope_bounds(
        self,
        predicted_centre,
        latent_train,
        y_lower,
        y_upper,
        latent_test,
    ):
        Phi = np.asarray(latent_train)
        Phi_test = np.asarray(latent_test)

        y_lower = np.atleast_1d(y_lower).reshape(-1)
        y_upper = np.atleast_1d(y_upper).reshape(-1)
        predicted_centre = np.atleast_1d(predicted_centre).reshape(-1)

        n, d = Phi.shape

        # Interval centre and radii
        centre_y = (y_lower + y_upper) / 2.0
        radii_y = (y_upper - y_lower) / 2.0

        # Regularization
        lambda_reg = max(self.lambda_reg, 1e-8 * np.trace(Phi.T @ Phi) / d)

        # Linear operator from training to test
        M = np.linalg.pinv(Phi.T @ Phi + lambda_reg * np.eye(d)) @ Phi.T
        A = Phi_test @ M

        # Compute tight L1 zonotope bounds
        centre_pred = A @ centre_y
        radius = np.sum(np.abs(A) * radii_y, axis=1)

        y_lower_pred = centre_pred - radius
        y_upper_pred = centre_pred + radius

        return {
            "pred_centre": predicted_centre,
            "y_lower_pred": y_lower_pred,
            "y_upper_pred": y_upper_pred,
        }

    def _compute_ipm_bounds(self, latent_train, latent_test, y_lower, y_upper):
        latent_train = latent_train.detach().numpy().astype(np.float64)
        latent_test = latent_test.detach().numpy().astype(np.float64)

        y_lower = np.asarray(y_lower, dtype=np.float64).ravel()
        y_upper = np.asarray(y_upper, dtype=np.float64).ravel()

        # Double dataset for PyIPM point regression
        latent_train = np.vstack([latent_train, latent_train])
        y_train = np.concatenate([y_lower, y_upper])

        model = PyIPM.IPM()
        model.fit(latent_train, y_train)

        return model.predict(latent_test)

    def compute(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        ipm: bool = False,
    ):
        self.centre_net.eval()
        with torch.no_grad():
            centre_pred, latent_test = self.centre_net(
                x_test, return_latent=True)
            _, latent_train = self.centre_net(x_train, return_latent=True)

        latent_train_np = latent_train.numpy()
        latent_test_np = latent_test.numpy()
        y_lower_np = np.atleast_1d(y_lower).flatten()
        y_upper_np = np.atleast_1d(y_upper).flatten()

        zono_dict = self._compute_zonotope_bounds(
            predicted_centre=centre_pred.numpy(),
            latent_train=latent_train_np,
            y_lower=y_lower_np,
            y_upper=y_upper_np,
            latent_test=latent_test_np,
        )

        if ipm:
            ipm_upper, ipm_lower = self._compute_ipm_bounds(
                latent_train, latent_test, y_lower_np, y_upper_np
            )
            return zono_dict, ipm_upper, ipm_lower

        return zono_dict
