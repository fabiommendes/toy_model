from toy import Model
from toy.app import App


class Particle(Model):
    # Variables
    x = 0, '[m] position'
    v = 0, '[m/s] speed'

    # Equations of motion
    D_x = v
    D_v = 2.0


if __name__ == '__main__':
    App(Particle()).run()
