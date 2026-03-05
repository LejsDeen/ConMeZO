import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class ZerothOrderOptimizer(ABC):
    """Abstract base class for zeroth-order optimization algorithms."""

    def __init__(self, mu: float = 0.01, verbose: bool = True):
        self.mu = mu
        self.verbose = verbose

    @abstractmethod
    def optimize(self, x_init: np.ndarray, func: callable, T: int, eta: float,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def zo_gradient_estimator(self, x: np.ndarray, s: np.ndarray, func: callable) -> np.ndarray:
        f_plus = func(x + self.mu * s)
        f_minus = func(x - self.mu * s)
        grad_est_vector = (f_plus - f_minus) / (2 * self.mu) * s
        return grad_est_vector

    def _check_divergence(self, current_f_val: float, initial_f_val: float,
                         iteration: int, algorithm_name: str, **params) -> bool:
        if (np.isinf(current_f_val) or np.isnan(current_f_val) or
            (initial_f_val > 1e-9 and current_f_val > 1e12 * initial_f_val) or
            current_f_val > 1e10):
            if self.verbose:
                param_str = ", ".join([f"{k}={v:.7f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in params.items()])
                print(f"{algorithm_name} diverged at iteration {iteration} with {param_str}. "
                      f"f_val={current_f_val:.2e}")
            return True
        return False


class MeZOOptimizer(ZerothOrderOptimizer):
    """Memory-efficient Zeroth-Order (MeZO) optimizer."""

    def optimize(self, x_init: np.ndarray, func: callable, T: int, eta: float,
                 gradient_func: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        x = np.copy(x_init)
        d = len(x_init)
        history = np.zeros(T + 1)
        grad_norm_history = np.zeros(T + 1)

        history[0] = func(x)
        if gradient_func is not None:
            grad_norm_history[0] = np.linalg.norm(gradient_func(x))

        for t in range(T):
            s_t = np.random.normal(0, 1, size=d)
            grad_t_vec_est = self.zo_gradient_estimator(x, s_t, func)

            x = x - eta * grad_t_vec_est
            current_f_val = func(x)
            history[t + 1] = current_f_val

            if gradient_func is not None:
                grad_norm_history[t + 1] = np.linalg.norm(gradient_func(x))

            if self._check_divergence(current_f_val, history[0], t + 1, "MeZO", eta=eta):
                history[t + 1:] = np.nan
                grad_norm_history[t + 1:] = np.nan
                break

        return history, grad_norm_history


class ConMeZOOptimizer(ZerothOrderOptimizer):
    """ConMeZO optimizer with cone sampling."""

    def optimize(self, x_init: np.ndarray, func: callable, T: int, eta: float,
                 alpha_m: float, theta: float, m_init: Optional[np.ndarray] = None,
                 gradient_func: Optional[callable] = None,
                 warmup_steps: int = 0,
                 warmup_alpha_schedule: str = 'linear', warmup_theta_schedule: str = 'linear'
                 ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.copy(x_init)
        d = len(x_init)
        m = np.zeros(d) if m_init is None else np.copy(m_init)

        history = np.zeros(T + 1)
        grad_norm_history = np.zeros(T + 1)

        history[0] = func(x)
        if gradient_func is not None:
            grad_norm_history[0] = np.linalg.norm(gradient_func(x))

        for t in range(T):
            current_theta, current_alpha_m = self._get_warmup_params(
                t, theta, alpha_m, warmup_steps, warmup_alpha_schedule, warmup_theta_schedule
            )

            if t == 0 or np.linalg.norm(m) < 1e-8:
                z_t = np.random.normal(0, 1, size=d)
            else:
                z_t = self._cone_sample(m, current_theta, d)

            grad_t_vec_est = self.zo_gradient_estimator(x, z_t, func)

            x = x - eta * grad_t_vec_est
            current_f_val = func(x)
            history[t + 1] = current_f_val

            if gradient_func is not None:
                grad_norm_history[t + 1] = np.linalg.norm(gradient_func(x))

            m = (1 - current_alpha_m) * m + current_alpha_m * grad_t_vec_est

            if self._check_divergence(current_f_val, history[0], t + 1, "ConMeZO",
                                    eta=eta, alpha_m=current_alpha_m, theta=current_theta):
                history[t + 1:] = np.nan
                grad_norm_history[t + 1:] = np.nan
                break

        return history, grad_norm_history

    def _cone_sample(self, momentum: np.ndarray, theta: float, d: int) -> np.ndarray:
        m_hat = momentum / np.linalg.norm(momentum)
        u_t = np.random.normal(0, 1, size=d)
        z_t = np.cos(theta) * np.sqrt(d) * m_hat + np.sin(theta) * u_t
        return z_t

    def _get_warmup_params(self, iteration: int, target_theta: float, target_alpha_m: float,
                          warmup_steps: int,
                          warmup_alpha_schedule: str, warmup_theta_schedule: str = 'linear'
                          ) -> Tuple[float, float]:
        if warmup_steps <= 0 or iteration >= warmup_steps:
            return target_theta, target_alpha_m

        progress = iteration / warmup_steps
        start_theta = np.pi / 2
        if warmup_theta_schedule == 'linear':
            current_theta = start_theta - (start_theta - target_theta) * progress
        elif warmup_theta_schedule == 'exponential':
            current_theta = target_theta + (start_theta - target_theta) * np.exp(-3 * progress)
        elif warmup_theta_schedule == 'cosine':
            current_theta = target_theta + (start_theta - target_theta) * 0.5 * (
                1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown theta warmup schedule: {warmup_theta_schedule}")

        if warmup_alpha_schedule == 'linear':
            current_alpha_m = target_alpha_m * progress
        elif warmup_alpha_schedule == 'exponential':
            current_alpha_m = target_alpha_m * (1 - np.exp(-5 * progress))
        elif warmup_alpha_schedule == 'cosine':
            current_alpha_m = target_alpha_m * 0.5 * (1 - np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown warmup schedule: {warmup_alpha_schedule}")

        return current_theta, current_alpha_m


class OptimizerFactory:
    @staticmethod
    def create_optimizer(optimizer_type: str, **kwargs) -> ZerothOrderOptimizer:
        optimizer_type = optimizer_type.lower()
        if optimizer_type == 'mezo':
            return MeZOOptimizer(**kwargs)
        elif optimizer_type == 'conmezo':
            return ConMeZOOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
