import click
import matplotlib.pyplot as plt


class App:
    """
    Convert model into a CLI application.
    """
    def __init__(self, model):
        self.model = model

    def run(self):
        """
        Run application.
        """
        runner = make_click_function(self)
        return runner()


def make_click_function(app):
    """
    The CLI app uses click to interface with the user.

    This dynamically create a Click command by inspecting properties of the
    model instance.
    """
    model = app.model

    @click.command()
    @click.option('--time', '-t', default='10', help='Simulation time')
    def main(time, **kwargs):
        print('running application')
        args = map(parse_number, time.split(','))
        result = model.run(*args)

        plt.plot(result[0])
        plt.plot(result[1])
        # plt.plot(result.x_ts)
        # plt.plot(result.y_ts)
        # plt.plot(result.z_ts)

        plt.show()

    return main


def parse_number(x):
    """
    Parse argument as int or float.
    """
    try:
        return int(x)
    except ValueError:
        return float(x)
