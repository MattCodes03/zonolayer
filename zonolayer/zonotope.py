import numpy as np


class Zonotope:
    def __init__(self, centre, generators, is_diagonal=False):
        self.centre = np.array(centre).reshape(-1)         # shape (n,)
        self.is_diagonal = is_diagonal

        if is_diagonal:
            # Just store the diagonal elements - O(n) memory
            self.diagonal_generators = np.array(generators).reshape(-1)
            if self.diagonal_generators.shape[0] != self.centre.shape[0]:
                raise ValueError(
                    "Diagonal generators must have same dimension as centre")
            self.d = self.centre.shape[0]
            self.m = self.d  # diagonal has n generators
            self.generators = None
        else:
            self.generators = np.atleast_2d(generators)        # shape (n, m)
            if self.generators.shape[0] != self.centre.shape[0]:
                raise ValueError(
                    "Generators must have same dimension as centre")
            self.d = self.centre.shape[0]
            self.m = self.generators.shape[1]

    def minkowski_sum(self, other):
        new_centre = self.centre + other.centre

        # Handle diagonal cases
        if self.is_diagonal and other.is_diagonal:
            # Diagonal + Diagonal = Diagonal
            new_diag = self.diagonal_generators + other.diagonal_generators
            return Zonotope(new_centre, new_diag, is_diagonal=True)
        elif self.is_diagonal:
            self_gens = np.diag(self.diagonal_generators)
            new_generators = np.hstack((self_gens, other.generators))
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        elif other.is_diagonal:
            other_gens = np.diag(other.diagonal_generators)
            new_generators = np.hstack((self.generators, other_gens))
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        else:
            new_generators = np.hstack((self.generators, other.generators))
            return Zonotope(new_centre, new_generators, is_diagonal=False)

    def subtract(self, other):
        new_centre = self.centre - other.centre

        if self.is_diagonal and other.is_diagonal:
            new_generators = np.hstack((
                np.diag(self.diagonal_generators),
                -np.diag(other.diagonal_generators)
            ))
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        elif self.is_diagonal:
            self_gens = np.diag(self.diagonal_generators)
            new_generators = np.hstack((self_gens, -other.generators))
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        elif other.is_diagonal:
            other_gens = np.diag(other.diagonal_generators)
            new_generators = np.hstack((self.generators, -other_gens))
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        else:
            new_generators = np.hstack((self.generators, -other.generators))
            return Zonotope(new_centre, new_generators, is_diagonal=False)

    @classmethod
    def from_intervals(cls, lower, upper):
        lower = np.array(lower).reshape(-1)
        upper = np.array(upper).reshape(-1)

        if lower.shape != upper.shape:
            raise ValueError("Lower and upper must have same shape")

        centre = (lower + upper) / 2
        radii = (upper - lower) / 2

        return cls(centre, radii, is_diagonal=True)

    def output_interval(self):
        if self.is_diagonal:
            radius = np.abs(self.diagonal_generators)
        else:
            radius = np.sum(np.abs(self.generators), axis=1)

        lower = self.centre - radius
        upper = self.centre + radius
        return lower, upper

    def affine_map(self, A, b=None):
        new_centre = A @ self.centre
        if b is not None:
            new_centre = new_centre + np.array(b).reshape(-1)

        if self.is_diagonal:
            # A @ diag(d) = A with each column i scaled by d[i]
            # This is: A[:, i] * d[i] for each column
            new_generators = A * self.diagonal_generators
            return Zonotope(new_centre, new_generators, is_diagonal=False)
        else:
            new_generators = A @ self.generators
            return Zonotope(new_centre, new_generators, is_diagonal=False)

    def __repr__(self):
        if self.is_diagonal:
            return f"Zonotope(dim={self.d}, generators={self.m}, diagonal)\nCentre:\n{self.centre}\nDiagonal generators:\n{self.diagonal_generators}"
        else:
            return f"Zonotope(dim={self.d}, generators={self.m})\nCentre:\n{self.centre}\nGenerators:\n{self.generators}"
