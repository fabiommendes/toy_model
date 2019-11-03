from sympy import ln

from toy import Model, App


class LotkaVolterra(Model):
    """
    Lotka-Volterra model

    See Also:
        https://en.wikipedia.org/wiki/Lotka–Volterra_equations
    """

    # Dynamic variables initial conditions
    x = 10.0, "[1] Prey"
    y = 1.0, "[1] Predator"

    # Parameters
    alpha = 0.5, "[1] Prey growth rate"
    beta = 0.1, "[1] Prey loss per encounter"
    delta = 0.05, "[1] Predator increase per encounter"
    gamma = 0.5, "[1] Predator decay rate"

    # Equations of motion
    D_x = alpha * x - beta * x * y
    D_y = delta * x * y - gamma * y

    # Invariants
    # I_potential = delta * x + gamma * ln(y) + beta * y - alpha * ln(y)


class Logistic(Model):
    """
    Logistic model for population growth: growth start as an exponential and
    asymptotically decay to zero as population reaches environment capacity.

    See Also:
        https://en.wikipedia.org/wiki/Logistic_function
    """

    # Variables
    x = 1, "[1] Population size"

    # Parameters
    r = 1, "[1] Growth rate"
    K = 10, "[1] Carrying capacity"

    # Equations of motion
    D_x = r * x * (1 - x / K)


if __name__ == '__main__':
    App(LotkaVolterra()).run()
