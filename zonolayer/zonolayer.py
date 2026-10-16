import numpy as np
import torch
import torch.nn as nn
from .zonotope import Zonotope
import PyIPM


def build_numpy_feature_extractor(model):
    """Snapshot the trained network's weights into a pure-NumPy forward pass.

    Returns extract(x) -> (preds, features), where `features` are the
    last-layer latent representations (output of model.feature_extractor).
    """
    layer_ops = []

    for module in model.feature_extractor:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            layer_ops.append(("linear", W, b))
        elif isinstance(module, nn.Tanh):
            layer_ops.append(("tanh", None, None))
        elif isinstance(module, nn.ReLU):
            layer_ops.append(("relu", None, None))
        elif isinstance(module, nn.Sigmoid):
            layer_ops.append(("sigmoid", None, None))
        elif isinstance(module, (nn.Identity, nn.Dropout)):
            continue  # no-ops at inference time
        else:
            raise NotImplementedError(
                f"No NumPy equivalent registered for layer type {type(module)}"
            )

    Wr = model.regressor.weight.detach().cpu().numpy()
    br = model.regressor.bias.detach().cpu().numpy()

    def extract(x):
        x = np.asarray(x, dtype=np.float32)
        for kind, W, b in layer_ops:
            if kind == "linear":
                x = x @ W.T + b
            elif kind == "tanh":
                x = np.tanh(x)
            elif kind == "relu":
                x = np.maximum(x, 0)
            elif kind == "sigmoid":
                x = 1.0 / (1.0 + np.exp(-x))
        features = x
        preds = (features @ Wr.T + br).flatten()
        return preds, features

    return extract


class Zonolayer:
    """
    Zonolayer: Last-layer uncertainty quantification via zonotopic representations.

    Learns an affine mapping from training outputs to test outputs,
    propagating interval uncertainties through this mapping to produce
    tight zonotope-based prediction intervals.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained regression network with `.feature_extractor` and `.regressor`.
    ipm : bool
        If True, returns IPM bounds. If False, does not return IPM bounds.
    """

    def __init__(self, model, ipm=False):
        self.model = model
        self.ipm = ipm
        self.zonolayer_ = None
        self.refresh_extractor()

    def refresh_extractor(self):
        """Build the NumPy extractor from self.model's current weights.

        Call it again if the model is retrained afterwards,
        since the extractor holds a snapshot of the weights.
        """
        self.model.eval()
        self._extract = build_numpy_feature_extractor(self.model)

    def get_latent(self, x):
        """Return latent features (n, d) for raw inputs x."""
        _, features = self._extract(x)
        return features

    def fit_zonolayer(self, x_train, y_lower, y_upper):
        # Latent train comes from the trained network
        latent_train = self.get_latent(x_train)

        X = np.atleast_2d(np.asarray(latent_train, dtype=np.float64))
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        y_l = np.asarray(y_lower, dtype=np.float64).ravel()
        y_u = np.asarray(y_upper, dtype=np.float64).ravel()

        c_y = 0.5 * (y_u + y_l)
        r_y = 0.5 * (y_u - y_l)

        XtX = X.T @ X
        Beta = np.linalg.solve(XtX, X.T)  # (d+1, n_train)
        w_c = Beta @ c_y                  # (d+1,)

        # Z_y = Z(c_y, diag(r_y)); Z_beta = Z(w_c, Beta @ diag(r_y)) - represented implicitly
        self.zonolayer_ = {"latent_train": latent_train,
                           "Beta": Beta, "r_y": r_y, "w_c": w_c}
        return self.zonolayer_

    def predict_zonolayer(self, x_test, zonolayer=None):
        zonolayer = zonolayer if zonolayer is not None else self.zonolayer_
        if zonolayer is None:
            raise RuntimeError("Call fit_zonolayer before predict_zonolayer.")

        # Latent test comes from the trained network
        latent_test = self.get_latent(x_test)

        X_test = np.atleast_2d(np.asarray(latent_test, dtype=np.float64))
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        Beta, r_y, w_c = zonolayer["Beta"], zonolayer["r_y"], zonolayer["w_c"]

        H = X_test @ Beta
        c_test = X_test @ w_c   # Predicted zonotope Z_y_hat center
        G_test = H * r_y        # Predicted zonotope Z_y_hat generators
        radius = np.sum(np.abs(G_test), axis=1)

        return {
            "latent_test": latent_test,
            "midpoint": c_test,
            "generator": G_test,
            "radius": radius,
            "y_lower": c_test - radius,
            "y_upper": c_test + radius,
        }

    def compute_ipm_bounds(self, x_train, x_test, y_lower_train, y_upper_train):
        ipm_model = PyIPM.IPM()

        x_train = np.asarray(x_train, dtype=np.float64)
        x_test = np.asarray(x_test, dtype=np.float64)
        y_lower = np.asarray(y_lower_train, dtype=np.float64).flatten()
        y_upper = np.asarray(y_upper_train, dtype=np.float64).flatten()

        # Combination of endpoints
        x_train = np.vstack([x_train, x_train])
        y_train = np.concatenate([y_lower, y_upper])

        ipm_model.fit(x_train, y_train)

        return ipm_model.predict(x_test)

    def compute(self, x_train, y_lower, y_upper, x_test):
        """
        Returns a dict with the zonolayer results and, if ipm=True, "ipm".
        """

        self.fit_zonolayer(x_train, y_lower, y_upper)
        results = self.predict_zonolayer(x_test)

        if self.ipm:
            results["ipm_upper"], results["ipm_lower"] = self.compute_ipm_bounds(
                self.zonolayer_[
                    "latent_train"], results["latent_test"], y_lower, y_upper
            )

        return results
