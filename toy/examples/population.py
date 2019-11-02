from sympy import ln

from toy import Model, App


class LotkaVolterra(Model):
    """
    Lotka-Volterra model

    See Also:
        https://en.wikipedia.org/wiki/Lotka–Volterra_equations
    """

    # Dynamic variables initial conditions
    x = 1.0, "Prey"
    y = 1.0, "Predator"

    # Parameters
    alpha = 1, "Prey growth rate"
    beta = 1, "Prey loss per encounter"
    delta = 1, "Predator increase per encounter"
    gamma = 1, "Predator decay rate"

    # Equations of motion
    D_x = alpha * x - beta * x * y
    D_y = delta * x * y - gamma * y

    # Invariants
    I_potential = delta * x + gamma * ln(y) + beta * y - alpha * ln(y)


class Logistic:
    """
    Logistic model for population growth: growth start as an exponential and
    asymptotically decay to zero as population reaches environment capacity.

    See Also:
        https://en.wikipedia.org/wiki/Logistic_function
    """

    # Variables
    x = 1, "Population size"

    # Parameters
    r = 1, "Growth rate"
    K = 10, "Carrying capacity"

    # Equations of motion
    D_x = r * x * (1 - x / K)


if __name__ == '__main__':
    App(LotkaVolterra()).run()
