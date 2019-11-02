from toy import Model


class Particle1D(Model):
    """
    A simple particle in a single dimension.
    """

    x = 0, "[m] Position"
    v = 0, "[m/s] Velocity"
    a = 0, "[m/s2] Acceleration"

    # Equations of motion
    D_x = v
    D_v = a


class Particle2D(Model):
    """
    A simple particle.
    """

    x = 0, "[m] x coordinate for position"
    vx = 0, "[m/s] x coordinate for velocity"
    ax = 0, "[m/s2] y acceleration"

    y = 0, "[m] y coordinate for position"
    vy = 0, "[m/s] y coordinate for velocity"
    ay = 0, "[m/s2] y acceleration"

    # Equations of motion
    D_x = vx
    D_y = vy
    D_vx = ax
    D_vy = ay


class Particle3D(Particle2D):
    """
    A simple particle in 3 dimensions
    """

    z = 0, "[m] x coordinate for position"
    vz = 0, "[m/s] x coordinate for velocity"
    az = 0, "[m/s2] y acceleration"

    # Equations of motion
    D_z = vz
    D_vz = az

