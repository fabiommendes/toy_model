How does Toy Model works?
=========================

Toy model is a Python library for creating complex dynamic systems modelled as a system of
ordinary differential equations. This is a very broad category which notably includes particle
systems in Physics, simplified economic, demographic and climate models, and many others.

Implementing an ODE solver in Python (specially one that uses simple methods like Euler integration)
is trivial. However, as the model grows in size, implementation often descends into giant blob of
barely maintainable (but sometimes delicious) spaguetti code.

Toy Model aims to make such models maintainable by treating each sub-system as independent units that
can be plugged together using a simple and predictable interface. This allow us, for instance, to
create an economic model, a separate climate model and a third model that connects both by
correlating emissions with economic output.

The point is that we can develop each model separately even though the final dynamics may be
be coupled. Toy Model treats each model class as an specification rather than a concrete
implementation. Implementation is created on the fly the first time the model is run and is compiled
from many bits in each sub-system.

Let us start with a simple model dynamic model describing a particle:

.. code-block:: python

    from toy import Model, var, cte


    class Particle(Model):
        """
        Simple 2D particle with an external acceleration.
        """

        # Constants
        g = cte(10, '[m/s2] acceleration of gravity')

        # Variables
        x = var(0, '[m] x position')
        y = var(0, '[m] y position')
        vx = var(0, '[m/s] x velocity')
        vy = var(0, '[m/s] y velocity')

        # Derivatives
        D_x = vx
        D_y = vy
        D_vx = 0
        D_vy = -g

The ``Particle`` class is just an specification for a dynamic model. It does not stores the
state of simulation or any dynamic variables. The actual simulation is executed by calling
the run() method with a range of times.

>>> model = Particle()
>>> run = model.run(0, 10, vx=10, vy=20)

The ``run`` object stores all intermediary steps of execution and some methods useful
to inspect the simulation result

>>> run.x_ts  # Time series for the x variable
>>> run.x     # Final value for the x variable
>>> run.run(0, 10)  # Run simulation for 10 more units of time.
<Run x=12, y=12, vx=10, vy=-10>

Toy Run object allocates an internal array that stores all state variables and a
an extension that includes constants and computed variables.