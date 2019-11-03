from toy import Model
from toy.app import App


class Particle(Model):
    # Variables
    x = 0, '[m] position'
    v = 0, '[m/s] speed'

    # Equations of motion
    D_x = v
    D_v = 2.0


class Exponential(Model):
    x = 1
    k = 1
    D_x = k * x


class Sin(Model):
    x = 0
    y = 1
    D_x = y
    D_y = -x


if __name__ == '__main__':
    App(Sin()).run()
