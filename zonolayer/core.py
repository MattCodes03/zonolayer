import numpy as np
import torch
from scipy.stats import t
from .zonotope import Zonotope
import PyIPM


class Zonolayer:
    """
    Zonolayer: Last-layer uncertainty modeling via zonotopic representations.

    This class fits a last-layer affine transformation to interval-bounded data,
    producing zonotopic output bounds and statistical prediction intervals.

    """

    def __init__(self, centre_net, lambda_reg: float = 1e-6, alpha: float = 0.05):
        """
        Parameters
        ----------
        centre_net : torch.nn.Module
            The trained neural network whose last-layer latent features are used.

        lambda_reg : float, optional
            Regularization strength for the pseudoinverse (default: 1e-6).

        alpha : float, optional
            Significance level for prediction intervals (default: 0.05 → 95% CI).
        """
        self.centre_net = centre_net
        self.lambda_reg = lambda_reg
        self.alpha = alpha

    # Core methods
    def _compute_zonotope_bounds(self, predicted_centre, latent_train, y_lower, y_upper, latent_test):
        """Internal routine to compute zonotope-based bounds and statistical intervals."""

        latent_train = np.asarray(latent_train)
        y_lower = np.asarray(y_lower).flatten()
        y_upper = np.asarray(y_upper).flatten()

        # Center and radii of output intervals
        centre_y = (y_lower + y_upper) / 2
        radii_y = (y_upper - y_lower) / 2

        # Create zonotope in output space
        Y_zono = Zonotope(centre_y, np.diag(radii_y))
        Phi = latent_train

        # Regularized pseudoinverse
        M = np.linalg.pinv(Phi.T @ Phi + self.lambda_reg *
                           np.eye(Phi.shape[1])) @ Phi.T

        # Map zonotope into parameter space
        Beta_zono = Y_zono.affine_map(M)

        # Map into test feature space
        Yhat_zono = Beta_zono.affine_map(latent_test)
        y_lower_pred, y_upper_pred = Yhat_zono.output_interval()

        # Statistical uncertainty
        predicted_centre = np.asarray(predicted_centre).flatten()
        y_mid = (y_upper + y_lower) / 2
        beta_hat = np.linalg.lstsq(Phi, y_mid, rcond=None)[0]
        residuals = y_mid - Phi @ beta_hat
        sigma2 = np.mean(residuals**2)
        sigma = np.sqrt(sigma2)

        # t-value
        n = len(latent_test)
        t_val = t.ppf(1 - self.alpha / 2, df=n - 1)
        r = 0.5 * (y_upper_pred - y_lower_pred)

        # Prediction standard error
        Phi_T_Phi_inv = np.linalg.inv(
            Phi.T @ Phi + self.lambda_reg * np.eye(Phi.shape[1]))
        SE_pred = sigma * \
            np.sqrt(1 + np.sum((latent_test @ Phi_T_Phi_inv) * latent_test, axis=1))

        # Combine statistical and zonotopic uncertainty
        y_pred_min_stat = predicted_centre - r - t_val * SE_pred
        y_pred_max_stat = predicted_centre + r + t_val * SE_pred

        return {
            "pred_centre": predicted_centre,
            "Beta_zono": Beta_zono,
            "Yhat_zono": Yhat_zono,
            "y_lower_pred": y_lower_pred,
            "y_upper_pred": y_upper_pred,
            "pi_lower": y_pred_min_stat,
            "pi_upper": y_pred_max_stat,
        }

    def _compute_ipm_bounds(self, latent_train, latent_test, y_lower, y_upper):
        """Internal routine to compute bounds using PyIPM."""

        latent_train = latent_train.detach().numpy().astype(np.float64)
        latent_test = latent_test.detach().numpy().astype(np.float64)

        y_lower = np.asarray(y_lower, dtype=np.float64).ravel()
        y_upper = np.asarray(y_upper, dtype=np.float64).ravel()

        # Double the dataset by using both endpoints, since PyIPM handles point targets
        latent_train = np.vstack([latent_train, latent_train])
        y_train = np.concatenate([y_lower, y_upper])

        model = PyIPM.IPM()  # 1st Degree Polynomial Model
        model.fit(latent_train, y_train)

        return model.predict(latent_test)

    def compute(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        y_lower: torch.Tensor,
        y_upper: torch.Tensor,
        ipm: bool = False
    ):
        """
        Compute last-layer zonotope bounds and statistical prediction intervals.

        Parameters
        ----------
        x_train : torch.Tensor
            Training inputs.
        y_lower, y_upper : numpy.ndarray
            Lower and upper interval endpoints for training targets.
        ipm : bool, optional
            Whether to use PyIPM for bound computations (default: False).

        Returns
        -------
        if ipm False: dict
            Dictionary containing:
            - y_lower_pred, y_upper_pred : zonotopic bounds
            - pi_lower, pi_upper : statistical prediction intervals
            - Beta_zono, Yhat_zono : zonotopic representations
        if ipm True: lower, upper : numpy.ndarrays
            Lower and upper bounds from PyIPM.
        """
        self.centre_net.eval()

        # Sort by input for clean visualization
        x_sorted, idx = torch.sort(x_train, dim=0)
        y_lb_sorted = y_lower[idx]
        y_ub_sorted = y_upper[idx]

        with torch.no_grad():
            # Predictions and latents for test points
            centre_pred, latent_test = self.centre_net(
                x_test, return_latent=True)

            # Latents for training points
            _, latent_train = self.centre_net(x_sorted, return_latent=True)

        # Convert to numpy
        latent_train_np = latent_train.numpy()
        latent_test_np = latent_test.numpy()
        y_lb_np = y_lb_sorted.flatten()
        y_ub_np = y_ub_sorted.flatten()

        if ipm:
            # Use PyIPM for zonotope computations
            return self._compute_ipm_bounds(latent_train, latent_test, y_lb_np, y_ub_np)
        else:

            return self._compute_zonotope_bounds(
                predicted_centre=centre_pred.cpu().numpy(),
                latent_train=latent_train_np,
                y_lower=y_lb_np,
                y_upper=y_ub_np,
                latent_test=latent_test_np,
            )
