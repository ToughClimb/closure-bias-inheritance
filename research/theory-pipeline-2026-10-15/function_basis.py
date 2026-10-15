"""Actual function coordinates: seven polynomials and two neural residuals."""
from __future__ import annotations
import numpy as np
import response_bounds as rb


class FunctionBasis:
    def __init__(self, problem):
        self.problem = problem
        self.grid = np.linspace(0, 1.2, 512)
        self.transforms, self.coefficients, self.residual_norms = {}, {}, {}
        d, r = problem.native.values(self.grid)
        for branch, degree, values in (("D", 2, d), ("R", 3, r)):
            matrix = np.polynomial.polynomial.polyvander(self.grid, degree)
            _, factor = np.linalg.qr(matrix / np.sqrt(512), mode="reduced")
            transform = np.linalg.solve(factor, np.eye(degree + 1))
            orthogonal = matrix @ transform
            coefficients = orthogonal.T @ values / 512
            residual = values - orthogonal @ coefficients
            norm = np.linalg.norm(residual) / np.sqrt(512)
            assert norm > 1e-10
            self.transforms[branch] = transform
            self.coefficients[branch] = coefficients
            self.residual_norms[branch] = float(norm)
        B_D, B_R = self.values(self.grid)
        gram = (B_D.T @ B_D + B_R.T @ B_R) / 512
        np.testing.assert_allclose(gram, np.eye(9), rtol=1e-8, atol=3e-10)
        self.gram_error = float(np.max(np.abs(gram - np.eye(9))))
        self.teacher_coordinates = np.r_[self.coefficients["D"], self.coefficients["R"],
                                         self.residual_norms["D"], self.residual_norms["R"]]
        truth = np.r_[np.linalg.solve(self.transforms["D"], [.01, .05, 0]),
                      np.linalg.solve(self.transforms["R"], [0, 1, -1, 0]), [0, 0]]
        self.truth_coordinates = truth

    def polynomials(self, branch, values, derivative=0):
        transform = self.transforms[branch]
        columns = []
        for k in range(transform.shape[1]):
            p = np.polynomial.polynomial.polyder(transform[:, k], m=derivative)
            columns.append(np.polynomial.polynomial.polyval(values, p))
        return np.stack(columns, axis=-1)

    def values(self, values, derivative=0):
        values = np.asarray(values)
        output = []
        if derivative:
            jet = self.problem.native.jet(values)
            native = (jet[2], jet[3]) if derivative == 1 else (jet[4], jet[5])
        else:
            native = self.problem.native.values(values)
        for branch, array, offset, residual_column in (("D", native[0], 0, 7), ("R", native[1], 3, 8)):
            polynomial = self.polynomials(branch, values, derivative)
            matrix = np.zeros((*values.shape, 9), dtype=values.dtype)
            matrix[..., offset:offset + polynomial.shape[-1]] = polynomial
            matrix[..., residual_column] = (array - polynomial @ self.coefficients[branch]) / self.residual_norms[branch]
            output.append(matrix)
        return tuple(output)

    def record(self):
        return dict(grid_points=512, fitting_interval=[0, 1.2],
                    inner_product="mean(D1*D2)+mean(R1*R2), absolute branch scales fixed at one",
                    dimensions=9, projection_rank=7,
                    functions="D polynomials degree<=2, R polynomials degree<=3, plus each actual frozen native residual",
                    transformations={k: v.tolist() for k, v in self.transforms.items()},
                    teacher_polynomial_coefficients={k: v.tolist() for k, v in self.coefficients.items()},
                    residual_norms=self.residual_norms, maximum_gram_identity_error=self.gram_error,
                    teacher_coordinates=self.teacher_coordinates.tolist(), truth_coordinates=self.truth_coordinates.tolist())
