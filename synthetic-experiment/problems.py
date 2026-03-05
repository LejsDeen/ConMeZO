import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def objective_function(self, x: np.ndarray) -> float:
        """Evaluate the objective function at point x."""
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient at point x."""
        pass

    def generate_initial_point(self, norm: float = 10.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random initial point with specified norm."""
        if seed is not None:
            np.random.seed(seed)
        x = np.random.normal(0, 1, size=self.dimension)
        return x / np.linalg.norm(x) * norm

    def get_optimal_value(self) -> float:
        """Return the optimal function value if known."""
        return 0.0

    def get_optimal_point(self) -> Optional[np.ndarray]:
        """Return the optimal point if known."""
        return np.zeros(self.dimension)


class QuadraticProblem(OptimizationProblem):
    """Quadratic function f(x) = 0.5 * ||x||^2."""

    def __init__(self, dimension: int):
        super().__init__(dimension)

    def objective_function(self, x: np.ndarray) -> float:
        """f(x) = 0.5 * ||x||^2"""
        return 0.5 * np.dot(x, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of f(x) = 0.5 * ||x||^2 is x"""
        return x

    def get_optimal_value(self) -> float:
        return 0.0

    def get_optimal_point(self) -> np.ndarray:
        return np.zeros(self.dimension)


class GeneralQuadraticProblem(OptimizationProblem):
    """General quadratic function f(x) = 0.5 * x^T A x where A is PSD."""

    def __init__(self, dimension: int, condition_number: float = 10.0,
                 eigenvalue_decay: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialize general quadratic problem.

        Args:
            dimension: Problem dimension
            condition_number: Condition number of matrix A (>=1)
            seed: Random seed for reproducible matrix generation
        """
        super().__init__(dimension)
        self.condition_number = condition_number
        self.eigenvalue_decay = eigenvalue_decay
        self.eigenvalues = self._generate_eigenvalues(
            dimension, condition_number, eigenvalue_decay, seed
        )

    def _generate_eigenvalues(
            self, dimension: int, condition_number: float,
            eigenvalue_decay: Optional[str] = None, seed: Optional[int] = None) -> np.ndarray:
        """Generate eigenvalues for a PSD matrix with specified condition number."""
        if seed is not None:
            np.random.seed(seed)

        if eigenvalue_decay is None:
            eigenvalue_decay = 'geometric'

        if dimension == 1:
            eigenvalues = np.array([1.0])
        elif eigenvalue_decay == 'geometric':
            log_min = np.log(1.0 / condition_number)
            log_max = np.log(1.0)
            indices = np.arange(dimension)
            log_eigenvalues = log_min + (log_max - log_min) * indices / (dimension - 1)
            eigenvalues = np.exp(log_eigenvalues)
        elif eigenvalue_decay == 'linear':
            eigenvalues = np.linspace(1.0 / condition_number, 1.0, dimension)
        else:
            raise ValueError(f"Unknown eigenvalue decay type: {eigenvalue_decay}")

        return eigenvalues

    def objective_function(self, x: np.ndarray) -> float:
        """f(x) = 0.5 * x^T A x"""
        return 0.5 * (x * self.eigenvalues * x).sum()

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of f(x) = 0.5 * x^T A x is A @ x"""
        return self.eigenvalues * x

    def get_optimal_value(self) -> float:
        return 0.0

    def get_optimal_point(self) -> np.ndarray:
        return np.zeros(self.dimension)

    def get_condition_number(self) -> float:
        """Return the condition number of matrix A."""
        return self.condition_number

    def get_matrix(self) -> np.ndarray:
        """Return the eigenvalues of the diagonal matrix A."""
        return self.eigenvalues.copy()


class ProblemFactory:
    """Factory class for creating optimization problems."""

    @staticmethod
    def create_problem(problem_type: str, dimension: int, **kwargs) -> OptimizationProblem:
        problem_type = problem_type.lower()

        if problem_type == 'quadratic':
            return QuadraticProblem(dimension)
        elif problem_type == 'general_quadratic':
            condition_number = kwargs.get('condition_number', 1.0)
            eigenvalue_decay = kwargs.get('eigenvalue_decay', None)
            seed = kwargs.get('seed', None)
            return GeneralQuadraticProblem(dimension, condition_number, eigenvalue_decay, seed)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
