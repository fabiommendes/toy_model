from toy import Model, App


class Lorenz(Model):
    """
    Lorenz system with chaotic dynamics.
    """

    # Dynamic variables initial conditions
    x = 1.0
    y = 0.0
    z = 0.0

    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8 / 3

    # Equations of motion
    D_x = sigma * (y - x)
    D_y = x * (rho - z) - y
    D_z = x * y - beta * z


if __name__ == '__main__':
    App(Lorenz()).run()
