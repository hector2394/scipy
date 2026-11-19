import logging
import platform
import sys

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import eig

logger = logging.getLogger(__name__)

SEED = 1234
MATRIX_SIZE = 5

type Matrix = NDArray[np.float64]
type Eigenpairs = tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]


def measure_residuals(a: Matrix, b: Matrix, eigenpairs: Eigenpairs) -> tuple[float, float]:
    values, left_vectors, right_vectors = eigenpairs
    left_rows = left_vectors.conj().T
    matrix_scale = np.linalg.norm(a, 2) + np.abs(values) * np.linalg.norm(b, 2)
    left_errors = left_rows @ a - values[:, None] * (left_rows @ b)
    right_errors = a @ right_vectors - (b @ right_vectors) * values
    left_scaled = np.linalg.norm(left_errors, axis=1) / (
        matrix_scale * np.linalg.norm(left_vectors, axis=0)
    )
    right_scaled = np.linalg.norm(right_errors, axis=0) / (
        matrix_scale * np.linalg.norm(right_vectors, axis=0)
    )
    return float(left_scaled.max()), float(right_scaled.max())


def report_case(generalized: bool) -> None:
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))
    b = rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))
    inputs = (a, b) if generalized else (a,)
    lhs_b = b if generalized else np.eye(MATRIX_SIZE)
    reference = eig(*inputs, left=True, right=True)
    fortran_inputs = tuple(np.asfortranarray(matrix) for matrix in inputs)
    overwritten = eig(*fortran_inputs, left=True, right=True, overwrite_a=True, overwrite_b=True)
    fortran_copy = eig(*(np.asfortranarray(matrix) for matrix in inputs), left=True, right=True)

    logger.info("generalized=%s", generalized)
    logger.info(
        "Scratch inputs changed (A, optional B): %s",
        tuple(
            not np.array_equal(original, scratch)
            for original, scratch in zip(inputs, fortran_inputs)
        ),
    )
    for name, eigenpairs in (
        ("C-order copy", reference),
        ("F-order copy", fortran_copy),
        ("F-order overwrite", overwritten),
    ):
        differences = tuple(
            float(np.max(np.abs(got - expected))) for got, expected in zip(eigenpairs, reference)
        )
        left_error, right_error = measure_residuals(a, lhs_b, eigenpairs)
        logger.info("  %s: max abs differences (w, vl, vr) = %s", name, differences)
        logger.info("    Max scaled residuals: left %.3e, right %.3e", left_error, right_error)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("Platform: %s", platform.platform())
    logger.info(
        "Python %s, NumPy %s, SciPy %s",
        platform.python_version(),
        np.__version__,
        scipy.__version__,
    )
    scipy.show_config()
    report_case(generalized=False)
    report_case(generalized=True)


if __name__ == "__main__":
    main()
