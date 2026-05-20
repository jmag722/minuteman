import numpy as np
from scipy.optimize import fsolve
import minuteman.thermodynamics.caloric_perfect as calp
import minuteman.compressible.isentropic as isen


def shock_tube(t: float,
               p_driver: float, p_driven: float, rho_driver: float, rho_driven: float,
               gam_driver: float = 1.4, gam_driven: float = 1.4,
               R_driver: float = calp.gas_constant_air_si, R_driven: float = calp.gas_constant_air_si,
               left_driver: bool = True, tube_length: float = 20.0, positions: np.ndarray = None):
    """
    Computes Sod shock tube problem. Both gases initially stagnant, with the contact
    surface centered in the tube at x=0.

    Parameters
    ----------
    t : float
        time [seconds]
    p_driver : float
        driver gas pressure
    p_driven : float
        driven gas pressure
    rho_driver : float
        driver gas density
    rho_driven : float
        driven gas density
    gam_driver : float, optional
        driver gas heat capacity ratio gamma, by default 1.4
    gam_driven : float, optional
        driven gas heat capacity ratio gamma, by default 1.4
    R_driver : float, optional
        driver gas constant, by default calp.gas_constant_air_si
    R_driven : float, optional
        driven gas constant, by default calp.gas_constant_air_si
    left_driver : bool, optional
        driver gas starts on left hand side rather than the right, by default True
    tube_length : float, optional
        shock tube length, by default 20. (Sod problem #1 length)
    positions : np.ndarray, optional
        explicit positions along the shock tube to evaluate, by default None.
        This is helpful when comparing to CFD grid directly.

    Returns
    -------
    dict
        position,
        pressure,
        density,
        sound speed,
        temperature,
        entropy,
        velocity,
        gamma,
        gas constant,
        specific internal energy,
        specific enthalpy,
        total energy,
        Mach number,
        region masks

    Raises
    ------
    ValueError
        Driver gas pressure is lower than driven gas pressure
    """
    p41 = p_driver/p_driven
    if p41 < 1.:
        raise ValueError(
            "Driver gas pressure must be greater than driven pressure.")

    p4 = p_driver
    r4 = rho_driver
    gam4 = gam_driver
    a4 = isen.speed_sound(gam=gam4, p=p4, rho=r4)
    T4 = p4/(r4*R_driver)
    u4 = 0.

    p1 = p_driven
    r1 = rho_driven
    gam1 = gam_driven
    a1 = isen.speed_sound(gam=gam1, p=p1, rho=r1)
    T1 = p1/(r1*R_driven)
    u1 = 0.

    p21 = solve_p21(p41=p41, a41=a4/a1, gam4=gam4, gam1=gam1)
    p2 = p21*p1
    r2 = moving_shock_density_ratio(p21=p21, gam=gam1)*r1
    a2 = isen.speed_sound(gam=gam1, p=p2, rho=r2)
    T2 = moving_shock_temperature_ratio(p21=p21, gam=gam1)*T1
    u2 = contact_surface_speed(p21=p21, a1=a1, gam1=gam1)  # u2=u3=V=u_piston
    W = moving_shock_speed(p21=p21, a1=a1, gam=gam1)

    p34 = p21/p41  # p2/p1 = p3/p1
    expansion34 = calp.isentropic_process_from_pressure(
        pressure_ratio=p34, specific_heat_ratio=gam4)
    p3 = p34*p4
    r3 = expansion34.density_ratio[0]*r4
    a3 = expansion34.speed_of_sound_ratio[0]*a4
    T3 = expansion34.temperature_ratio[0]*T4
    u3 = u2

    # critical positions
    x45 = (u4 - a4)*t  # expansion fan head
    x53 = (u3 - a3)*t  # expansion fan tail
    x32 = u2*t  # contact surface
    x21 = W*t  # shock
    # `set` removes multiple zeros @ t=0
    crit_pts = np.array(list(set([x45, x53, x32, x21])))
    if positions is None:
        N = 500
        # adding crit pts so discont. always well resolved
        x_arr = np.zeros(N + crit_pts.size)
        x_arr[:N] = np.linspace(-0.5*tube_length, 0.5*tube_length, N)
        x_arr[-crit_pts.size:] = crit_pts
        x_arr.sort()
    else:
        x_arr = np.zeros_like(positions)
        x_arr[:] = positions[:]

    # masks for each region
    region4 = x_arr < x45
    region5 = np.logical_and(x_arr >= x45, x_arr < x53)
    region3 = np.logical_and(x_arr >= x53, x_arr < x32)
    region2 = np.logical_and(x_arr >= x32, x_arr < x21)
    region1 = x_arr >= x21

    def assign_regions(_arr, _X4, _X5, _X3, _X2, _X1):
        _arr[region4] = _X4
        _arr[region5] = _X5
        _arr[region3] = _X3
        _arr[region2] = _X2
        _arr[region1] = _X1

    s_arr = np.empty_like(x_arr)  # entropy
    T_arr = np.empty_like(x_arr)
    u_arr = np.empty_like(x_arr)
    p_arr = np.empty_like(x_arr)
    a_arr = np.empty_like(x_arr)
    r_arr = np.empty_like(x_arr)

    m_arr = np.empty_like(x_arr)
    e_arr = np.empty_like(x_arr)
    h_arr = np.empty_like(x_arr)
    Et_arr = np.empty_like(x_arr)
    gam_arr = np.empty_like(x_arr)
    Rgas_arr = np.empty_like(x_arr)

    u5 = velocity_expansion_fan(a4, x_arr[region5], t, gam4)
    a5 = sound_speed_expansion_fan(a4, u5, gam4)
    expansion54 = calp.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=a5/a4, specific_heat_ratio=gam4)
    p5 = expansion54.pressure_ratio[0] * p4
    r5 = expansion54.density_ratio[0] * r4
    T5 = expansion54.temperature_ratio[0] * T4

    assign_regions(u_arr, u4, u5, u3, u2, u1)
    assign_regions(a_arr, a4, a5, a3, a2, a1)
    assign_regions(p_arr, p4, p5, p3, p2, p1)
    assign_regions(r_arr, r4, r5, r3, r2, r1)
    assign_regions(T_arr, T4, T5, T3, T2, T1)
    assign_regions(gam_arr, gam4, gam4, gam4, gam1, gam1)
    assign_regions(Rgas_arr, R_driver, R_driver, R_driver, R_driven, R_driven)
    s_arr[:] = calp.entropy_state(pressure=p_arr, density=r_arr,
                                  specific_heat_ratio=gam_arr,
                                  gas_constant=Rgas_arr)
    m_arr[:] = isen.mach(u_arr, a_arr)
    e_arr[:] = calp.specific_heat_constant_volume(
        specific_heat_ratio=gam_arr, gas_constant=Rgas_arr) * \
        T_arr  # specific internal energy = cv*T
    h_arr[:] = calp.specific_enthalpy(
        specific_internal_energy=e_arr, pressure=p_arr, density=r_arr)
    Et_arr[:] = calp.total_energy(
        pressure=p_arr, density=r_arr, speed=u_arr, specific_heat_ratio=gam_arr)

    POS_NAME = "position"
    answer = {
        POS_NAME: x_arr,
        "temperature": T_arr,
        "pressure": p_arr,
        "density": r_arr,
        "speed": u_arr,
        "sound_speed": a_arr,
        "entropy": s_arr,
        "gas_constant": Rgas_arr,
        "gamma": gam_arr,
        "mach": m_arr,
        "spec_internal_energy": e_arr,
        "spec_enthalpy": h_arr,
        "total_energy": Et_arr,
        "region1": region1,
        "region2": region2,
        "region3": region3,
        "region4": region4,
        "region5": region5,
    }

    if not left_driver:
        for k, v in answer.items():
            answer[k] = np.flip(v) if k != POS_NAME else v

    return answer


def velocity_expansion_fan(a4, x, t, gam):
    """
    Compute velocity within expansion fan. Eq. 7.89 of Anderson

    Parameters
    ----------
    a4 : float
        speed of sound of driver gas
    x : float | ArrayLike
        position within expansion fan
    t : float | ArrayLike
        time
    gam : float | ArrayLike
        ratio of specific heats

    Returns
    -------
    u
        velocity in the expansion fan bounds
    """
    return 2/(gam+1) * (a4 + x/t)


def sound_speed_expansion_fan(a4, u, gam):
    """
    Compute speed of sound within expansion fan - Eq. 7.84 of Anderson.

    Parameters
    ----------
    a4 : float
        speed of sound of driver gas
    u : float | ArrayLike
        velocity within expansion fan
    gam : float | ArrayLike
        ratio of specific heats

    Returns
    -------
    a
        speed of sound within expansion fan
    """
    return a4 - (gam-1)/2 * (u)


def moving_shock_density_ratio(p21: float, gam: float):
    """
    Compute the density ratio across a moving normal
    shock as a function of pressure ratio. See Eq. 7.11 in
    Anderson.

    Parameters
    ----------
    p21 : float
        pressure ratio across shock
    gam : float
        ratio of specific heats

    Returns
    -------
    rho2/rho1
        density ratio across shock
    """
    return (
        (1 + (gam+1)/(gam-1) * p21) / ((gam+1)/(gam-1) + p21)
    )


def moving_shock_temperature_ratio(p21: float, gam: float):
    """
    Compute the temperature ratio across a moving normal
    shock as a function of pressure ratio. See Eq. 7.10 in
    Anderson.

    Parameters
    ----------
    p21 : float
        pressure ratio across shock
    gam : float
        ratio of specific heats

    Returns
    -------
    T2/T1
        temperature ratio across shock
    """
    return (
        p21 * ((gam+1)/(gam-1) + p21) / (1 + (gam+1)/(gam-1) * p21)
    )


def moving_shock_speed(p21: float, a1: float, gam: float):
    """
    Compute the shock speed of the moving shock as a function
    of pressure ratio and the speed of sound of the driven gas.
    See Eq. 7.14 of Anderson.

    Parameters
    ----------
    p21 : float
        pressure ratio across shock
    a1 : float
        speed of sound of driven gas
    gam : float
        ratio of specific heat for the gas

    Returns
    -------
    W
        wave velocity of the moving shock wave
    """
    return (
        a1 * ((gam+1)/(2*gam) * (p21-1) + 1)**0.5
    )


def contact_surface_speed(p21: float, a1: float, gam1: float):
    """
    Compute the speed of the contact surface/piston in a shock tube,
    or the speed of the mass motion induced by the incident shock.
    See Eq. 7.16 of Anderson.

    As p21->infinity, for gam1=1.4, Mach number->1.89

    Parameters
    ----------
    p21 : float
        pressure ratio across shock
    a1 : float
        speed of sound in driven gas
    gam1 : float
        gamma of driven gas

    Returns
    -------
    up
        contact surface or piston speed
    """
    return (
        a1/gam1 * (p21-1) * (2*gam1/(gam1+1) / (p21 + (gam1-1)/(gam1+1)))**0.5
    )


def solve_p21(p41: float, a41: float, gam4: float, gam1: float):
    """
    Solve for the pressure ratio of the normal shock in a shock tube.
    See Eq. 7.94 of Anderson

    Parameters
    ----------
    p41 : float
        pressure ratio between driver and driven gas
    a41 : float
        speed of sound ratio between driver and driven gas
    gam4 : float
        gamma of driver gas
    gam1 : float
        gamma of driven gas

    Returns
    -------
    p21
        pressure ratio of the normal shock in shock tube
    """
    return fsolve(_p21_func, 0.5*p41, (p41, a41, gam4, gam1))[0]


def _p21_func(p21: float, p41: float, a41: float, gam4: float, gam1: float):
    """
    Eq. 7.94 of anderson
    """
    return (
        p41 - p21*(
            1 - ((gam4-1)/a41*(p21-1))
            / (2*gam1*(2*gam1+(gam1+1)*(p21-1)))**0.5
        )**((-2*gam4)/(gam4-1))
    )
