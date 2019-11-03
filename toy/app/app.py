import click
import matplotlib.pyplot as plt


class App:
    """
    Convert model into a CLI application.
    """

    def __init__(self, model):
        self.model = model
        self.click = make_click_function(self)

    def run(self):
        """
        Run application.
        """
        return self.click()


def make_click_function(app):
    """
    The CLI app uses click to interface with the user.

    This dynamically create a Click command by inspecting properties of the
    model instance.
    """
    model = app.model

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.option('--time', '-t', default='', help='Simulation time')
    @click.option('--legend', '-l', default=True, help='Show legend')
    @click.option('--solver', default='rk4', help='Solver algorithm')
    def series(time, legend, solver):
        run = run_times(model, time, solver=solver)
        for name in model.vars:
            value = getattr(run, name + '_ts')
            plt.plot(run.times, value, label=name)
        if legend:
            plt.legend()
        plt.show()

    @cli.command()
    @click.argument('x', default=None)
    @click.argument('y', default=None)
    @click.option('--time', '-t', default='', help='Simulation time')
    @click.option('--solver', default='rk4', help='Solver algorithm')
    def trajectory(x, y, time, solver):
        run = run_times(model, time,  solver=solver)
        X = getattr(run, x + '_ts') if x else run.values[0]
        Y = getattr(run, y + '_ts') if y else run.values[1]
        plt.plot(X, Y)
        plt.show()

    return cli


def parse_number(x):
    """
    Parse argument as int or float.
    """
    try:
        return int(x)
    except ValueError:
        return float(x)


def run_times(model, time, **kwargs):
    print('Running application')
    args = map(parse_number, time.split(','))
    return model.run(*args, **kwargs)