=========
Toy Model
=========

Toy model is a Python library for creating complex dynamic systems modelled as a system of
ordinary differential equations. This is a very broad category, which notably includes particle
systems in Physics, simplified economic, demographic, and climate models, and many others.

Implementing an ODE solver in Python (specially one that uses simple methods like Euler integration)
is trivial. However, as the model grows in size, implementation often descends into giant blob of
barely functional, heroically maintainable (but sometimes delicious) spaghetti code.

Toy Model aims to make such models organized by treating each sub-system as independent units that
can be plugged together using a simple and predictable interface. This allow us, for instance, to
create an economic model, a separate climate model and a third model that connects both by
correlating emissions with economic output. It also makes it easy to test and to validate each model
separately, and easily remix models in different combinations.

The point is that we can develop each model separately even though the final dynamics may be
be coupled. Toy Model treats each model class as an specification of a dynamical system
rather than a concrete implementation of an ODE solver. Implementation is created on-the-fly
the first time the model is run and is compiled from many bits collected from each sub-system.

Installing
==========

Toy Model is packaged in PyPI_ using flit_. Just ``pip install toy-model`` or use your
favorite method of `installing Python packages`_.

.. _flit: https://flit.readthedocs.io/en/latest/
.. _PyPI: https://pypi.org
.. _installing Python packages: ???


Using the model
===============

Let us start with a simple kinematic model for a particle:

.. code-block:: python

    from toy import Model


    class Particle(Model):
        """
        Simple 2D particle with external acceleration.
        """

        # Variables
        x = 0, '[m] x position',
        y = 0, '[m] y position'
        vx = 0, '[m/s] x velocity'
        vy = 0, '[m/s] y velocity'

        # Constants
        a = -2, '[m/s2] uniform acceleration'

        # Bounds
        bounds = [y >= 0]

        # Derivatives
        D_x = vx
        D_y = vy
        D_vx = 0
        D_vy = a


The ``Particle`` class is just an specification for a dynamic model. It does not store the
state of simulation or any dynamic variables. The actual simulation is executed by calling
the run() method with a range of times.

>>> model = Particle()
>>> run = model.run(0, 1, 5, vx=1, vy=2.0)

The "run" object stores all intermediary steps of execution and some methods that are
useful to inspect the simulation result. We can, for instance, fetch the final value for
a variable,

>>> run.x
1.0

and its associated time series,

>>> run.x_ts
array([0.  , 0.25, 0.5 , 0.75, 1.  ])

The run object can also be used to keep running the simulation, inspect some running
parameters, saving state, and much more. We refer to the :cls:`Run` documentation for
more details.

>>> run.run(1, steps=10)
<Run t=1.0, x=1.0, y=1.0, vx=1.0, vy=0.0>

The particular details of how we go from a model subclass (e.g., Particle) to a simulation
result involves some sophisticated processing and meta programming. The details of
this step will be discussed later and are important to understand how
Toy Model works. We will come back to this point after showing the most important bits of
the API first.

The point of this library is that users should only declare structure and the
Toy Model simply infer the a suitable implementation that may depends on the solver,
model topology and other considerations.


Structure of a model
====================

Consider the model:

.. code-block:: python

    from sympy import sin


    class ForcedOscillator(Model):
        x = 0, '[m] Position of point mass'
        v = 1, '[m/s] Velocity'
        m = 1, '[kg] Mass'
        k = 1, '[N/m] Spring constant'
        F = 1, '[N] Amplitude of forced oscillation'
        omega = 0.5, '[rad/s] Frequency of oscillation'
        gamma = 0.1, '[kg/s] Damping coefficient'

        # Force
        force = F * sin(omega * t), '[N] External force'

        # Equations
        D_x = v
        D_v = -k * x - gamma * v + force / m


Variables that form a dynamic model can be classified into any of 3 different categories.
First, and perhaps more obviously, are the dynamical variables that we want to solve for,
in this case ``x`` and ``v``. In Toy model, those variables are referred as
"dynamic variables" or simply as "vars". *Vars* are exposed as a dictionary that maps
variable names to their corresponding :cls:`Value` declarations:

>>> m = ForcedOscillator()
>>> m.vars
{'x': Value('x', 0), 'v': Value('v', 1)}

The :cls:`Value` objects store information such as name, bound symbol, units,
description, etc.

The second group of variables is what we call "parameters", or "params". Those are values
that do not change during simulation, such as mass, the spring constant, etc. All *params*
must be reduced to numbers when model is initialized. They don't change.

>>> m.params   # doctest: +ELLIPSIS
{'m': Value('m', 1), 'k': Value('k', 1), ...}

If you only need the initial values, use

>>> m.param_values()
{'m': 1, 'k': 1, 'F': 1, 'omega': 0.5, 'gamma': 0.1}

This distinction is important, because parameters cannot be changed once the
model is initialized, but the initial values for vars can. That is, the run()
method can override vars, but not params.

For instance, that's ok:

>>> m.run(0, 10, v=2)  # doctest: +SKIP

That is an error:

>>> m.run(0, 10, k=2)  # doctest: +SKIP

We can, however, override parameters during model initialization, by creating
different instances of a model class

>>> m1 = ForcedOscillator(k=2)
>>> m2 = ForcedOscillator(k=1)

Some auxiliary variables must be computed at every step of the simulation,
usually because they depend on time or the other dynamic variables. This is
what the "force" term is in the oscillator model.
We refer to those terms as "auxiliary terms" or simply as "aux",

>>> m.aux
{'force': Value('force', sin(0.5*t))}

They are subject to similar restriction as parameters, in that it is not possible
to change computed terms in the run() method, but we can do it during model
initialization. In fact, since we can override expressions to constant numerical
values and vice-versa, the distinction between parameters and computed terms
is only possible after model initialization.

>>> m3 = ForcedOscillator(force=0)
>>> m3.aux
{}
>>> m3.params  # doctest: +ELLIPSIS
{'m': Value('m', 1), ..., 'force': Value('force', 0)}


Composing models
================



Topics
======

* Variables, Parameters, and Computed
* Algebraic expressions and parameter overriding
* Units
* Model fusion and sub-classing
* Model composition and mounting
* Model vectorization
* Sensors, validation, and control
* AOT and JIT compilation
* Plotting
* CLI Apps
* Jupyter widgets and apps
* API reference
* Examples


Class creation and interpretation
---------------------------------

Python has powerful meta programming capabilities and allow us to customize many
aspects of class creation. Toy model uses a somewhat obscure feature of metaclass
programming, which is the ability to change the type of the scope dictionary used
internally during class declaration.

This allow us to turn a class declaration into very simple embedded
interpreter to a domain specific language (eDSL). Python
commands inside Model declaration are reinterpreted as mathematical expressions and
**do not** obey standard Python semantics. This embedded language is largely powered
by Sympy_, which is a Python computer algebra system.

.. _Sympy: https://sympy.pydata.org

