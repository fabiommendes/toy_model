import sympy as sp

from toy import Model


class Carbon(Model):
    """
    A simple carbon model with 3 reservoirs: atmosphere, shallow oceans and
    deep oceans.

    Carbon simply flows between reservoirs and is conserved. In larger time
    scales, we would expect net carbon sinks due to rock weathering, build up
    of fossil fuels, and to have occasional carbon sources due to volcanic
    activity.

    Perhaps a better model (or a minimally good model) should include soil.

    Initial values are from 2015.
    """
    co2_atm = 851, '[GtC] Total carbon in the atmosphere'
    co2_shallow = 460, '[GtC] Total carbon in the shallow oceans'
    co2_deep = 1740, '[GtC] Total carbon in the deep oceans'

    # Flux rates and emissions
    f_AO = 0.024, '[GtC/yr] atmosphere to shallow ocean'
    f_OA = 0.0392, '[GtC/yr] shallow ocean to atmosphere'
    f_OU = 0.00029302, '[GtC/yr] deep to shallow ocean'
    f_OD = 0.0014, '[GtC/yr] shallow to deep ocean'
    emissions = 0.0, '[GtC/yr] emissions to the atmosphere'

    # Dynamic equations
    D_co2_atm = f_OA * co2_shallow - f_AO * co2_atm + emissions
    D_co2_shallow = f_AO * co2_atm + f_OU * co2_deep - (f_OA + f_OD) * co2_shallow
    D_co2_deep = f_OD * co2_shallow - f_OU * co2_deep


class Temperature(Model):
    """
    Temperature model based on the known logarithmic relation between
    CO2 concentration and radiative forcing.

    It simulates the temperature anomaly, i.e., the difference in temperature
    from pre-industrial times. It tracks atmosphere independently from the
    oceans. Atmosphere temperature is driven by radiative forcings and heat
    escaping to space and ocean temperature is driven by heat exchange with
    the atmosphere.

    It may include external radiative forcings that can be tested for different
    scenarios or to model the influence of other GHGs.
    """
    T_atm = 0.85, '[K] Atmospheric temperature'
    T_ocean = 0.0068, '[K] Oceanic temperature'

    # Historic and simulated values
    RF_double = 3.6813, '[W/m2] Estimated forcing for doubling CO2'
    RF_extra = 0.5, '[W/m2] Forcing due to external factors (other GHGs, aerosols, etc)'
    T_double = 3.1, '[K] Anomaly for doubling CO2'
    co2_atm_PI = 588.0, '[GtC] Pre-industrial carbon stock'
    co2_atm = 851, '[GtC] Total carbon in the atmosphere'

    # Radiative forcings
    RF = RF_double * sp.log(co2_atm / co2_atm_PI) / sp.log(2) + RF_extra

    # Temperature dynamics
    # Perhaps the escaping heat should be modelled using Stephen-Boltzmann
    # equation [(T_PI + T_atm)**4] rather than a proportional term.
    C1 = 0.1005 / 5
    C2 = 0.088 * 0.1005 / 5
    C3 = 0.025 / 5
    D_T_atm = C1 * (RF - RF_double / T_double * T_atm) - C2 * (T_atm - T_ocean)
    D_T_ocean = C3 * (T_atm - T_ocean)


class Population(Model):
    """
    Assumes an exponential growth with a growth rate that decays exponentially.
    Population dynamics do not depend on climate and economic variables, which
    is not very realistic.

    Assumes values (total population and growth) from 2015.
    """

    pop = 7403, '[1] Million people'
    g_pop = 0.0268, '[yr-1] Growth per year'
    f_pop = 0.08, '[1] Calibrated to reach a desired maximum population of 11,5bi'
    D_pop = g_pop * pop
    D_g_pop = -f_pop * g_pop


class TotalFactorOfProductivity(Model):
    """
    In neoliberal economics, Total Factor of Productivity (TFP) measures
    efficiency in which the factors of productivity such as capital and labor
    are converted to production.

    It appears in DICE in an equation like:

        Q_raw = TFP * K^g * L^(1 - g),

    in which Q_raw represents raw production, K is capital and L is labor.

    It models technological, educational and institutional improvements.
    """

    TFP = 5.115, '[1] Rate between capital and production'
    g_TFP = 0.0152, '[1] Growth parameter for TFP'

    f_TFP = 0.005, '[1] Exponential decay for TFP growth'
    D_TFP = g_TFP * TFP
    D_g_TFP = -f_TFP * g_TFP


class Production(Model):
    """
    Main economic model relating production to capital, labor, savings rate,
    and other variables.
    """

    # External constants
    TFP = 5.115, '[1] Rate between capital and production'
    pop = 7403, '[1] Million people'
    losses = 0.0, '[U$] Losses due to climate change'
    abatement_cost = 0.0, '[U$] Cost to abate CO2'

    # Constants
    gamma = 0.3, '[1] Elasticity of capital'
    savings = 0.25, '[yr-1] Fraction of production reinvested in a year'
    Kdelta = 0.10, '[yr-1] Yearly depreciation of capital'

    # Definitions
    K = 223, '[U$^15] Total world capital'
    L = pop / 1000, '[U$] Fixed conversion between labor and population'
    Q_raw = TFP * K ** gamma * L ** (1 - gamma)
    Q = Q_raw * (1 - losses - abatement_cost)
    consumption_per_capta = (Q - Q * savings) * 1e6 / pop

    # Capital
    D_K = Q * savings - Kdelta * K


class Emissions(Model):
    """
    Main economy-climate emissions interaction.

    Decomposes emissions into industrial (ind) vs. land use change (LUC).

    Industrial emissions are proportional to production.

    DICE do not discriminate transportation, energy generation, farming, etc.
    """

    # External
    Q_raw = 25.833, '[U$^15] World GDP'
    abatement = 0.0, '[1] CO2 abatement ratio'

    # Carbon intensity
    g_co2i = 0.01, '[1] Yearly decay of co2 intensity'
    f_co2i = -0.0002, '[1] Yearly decay of co2 intensity decay'
    co2_intensity = 0.1098, '[tC/U$^3] Carbon emitted per 1000 U$ produced'
    D_co2_intensity = -g_co2i * co2_intensity
    D_g_co2i = -f_co2i * g_co2i

    emissions_luc = 2.6, '[GtC/yr] Yearly emissions due to LUC'
    emissions_ind = co2_intensity * (1 - abatement) * Q_raw
    emissions = emissions_ind + emissions_luc


class Costs(Model):
    """
    Main climate-economy cost interaction.

    Nordhaus assumes a quadratic relation between damages and temperature and
    fits the quadratic function using very different estimations.
    """

    # External
    Q_raw = 25.833, '[U$^15] World GDP'
    T_atm = 0.85, '[K] Atmospheric temperature'
    co2_intensity = 0.1098, '[tC/U$^3] Carbon emitted per 1000 U$ produced'

    # Constants
    loss_a = 0.0, '[1] Linear coefficient of quadratic fit'
    loss_b = 0.00236, '[1] Quadratic coefficient'
    losses = (loss_a * T_atm + loss_b * T_atm ** 2), '[1] Fraction of '

    # Backstop price
    backstop_price = 550, '[U$/tC] Value to remove 1 ton of CO2'
    g_bsp = 0.005, '[yr-1] Yearly decay on backstop price'
    D_backstop_price = -g_bsp * backstop_price

    # Custos de abatimento
    abatement = 0.0, '[1] CO2 abatement ratio'
    abatement_exp = 2.6, '[1] abatement exponent'
    abatement_eff = backstop_price * co2_intensity / abatement_exp / 1000
    abatement_cost = abatement_eff * abatement ** abatement_exp, \
                     '[1] ratio of production dedicated to abatement'


if __name__ == '__main__':
    from toy.app import App

    model = Carbon
    model = Population
    model = Production
    model = Emissions
    #model = Costs
    App(model()).run()
