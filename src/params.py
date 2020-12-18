from math import exp
import numpy as np

mu = 1.0 * pow(10, -3)  # mutation rate per site per Myr
tau1 = 4
tau2 = 1.5
thtHC = 3
thtHCG = 3
div_time = 18
term = 1 / exp(tau2 / thtHC)


""" Helpers """


def phc1(tau2, thtHC):
    return 1 - exp(-tau2 / thtHC)


def phc2(tau2, thtHC):
    return exp(-tau2 / thtHC) / 3


def phg(tau2, thtHC):
    return exp(-tau2 / thtHC) / 3


def pcg(tau2, thtHC):
    return exp(-tau2 / thtHC) / 3


def expRho(rho):
    return exp(rho)


def expTau2overThetaHC(tau2, thtHC):
    return exp(tau2 / thtHC)


def expRhoGTau1PlusTau2(rho, tau1, tau2):
    return exp(rho * (tau1 + tau2))


def expRhoHTau1PlusTau2(rho, tau1, tau2):
    return exp(rho * (tau1 + tau2))


def expRhoMTau1PlusTau2(rho, tau1, tau2):
    return exp(rho * (tau1 + tau2))


def expRhoHTau1(rho, tau1):
    return exp(rho * tau1)


def expRhoHTau2(rho, tau2):
    return exp(rho * tau2)


def expRhoMTau1(rho, tau1):
    return exp(rho * tau1)


def expRhoMTau2(rho, tau2):
    return exp(rho * tau2)


def expRhoGTau1(rho, tau1):
    return exp(rho * tau1)


def expRhoGTau2(rho, tau2):
    return exp(rho * tau2)


def e001a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = (
        (expTau2overThetaHC(tau2, thtHC) ** -1)
        * (expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1)
        * (expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1)
    )
    y = 1 - (expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1)
    z = 3 / (2 * (3 + thtHCG * (rho_h + rho_m)))
    w = 1 / (2 * (5 + thtHCG * (rho_h + rho_m)))
    return (1 / 3) * x * y * (z - w)


def e001b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = thtHCG * rho_g
    u = 6 + thtHCG * (rho_h + rho_m)

    v1 = 3 + thtHCG * (rho_h + rho_m)
    v2 = 5 + thtHCG * (rho_h + rho_m)
    v3 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    nr = x * y1 * y2 * y3 * z * u
    dr = 3 * v1 * v2 * v3

    return nr / dr


def e001(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e001a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e001b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -2
    y = 1 - expRhoHTau1(rho_h, tau1) ** -1
    z = (expRhoHTau1(rho_h, tau1) ** -1) * (expTau2overThetaHC(tau2, thtHC) ** -2)
    w = (expTau2overThetaHC(tau2, thtHC) * expRhoHTau2(rho_h, tau2) ** -1) - 1
    u = thtHC * rho_h
    return x * y - (z * w * u) / (u - 1)


def mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -2
    y = 1 - expRhoMTau1(rho_m, tau1) ** -1
    z = (expRhoMTau1(rho_m, tau1) ** -1) * (expTau2overThetaHC(tau2, thtHC) ** -2)
    w = (expTau2overThetaHC(tau2, thtHC) * expRhoMTau2(rho_m, tau2) ** -1) - 1
    u = thtHC * rho_m
    return x * y - (z * w * u) / (u - 1)


def e100x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = (expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1) * (
        expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    )
    y = 3 + thtHCG * (rho_m + rho_g)
    z = 5 + thtHCG * (rho_m + rho_g)
    return x / (y * z)


def e100a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e100x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e100b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = thtHCG * rho_h

    u1 = 3 + thtHCG * (rho_g + rho_m)
    u2 = 5 + thtHCG * (rho_g + rho_m)
    u3 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    nr = x * y1 * y2 * y3 * z
    dr = u1 * u2 * u3

    return nr / dr


def e100(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e100a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e100b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def e010x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y = 5 + thtHCG * (rho_h + rho_g)
    return (x1 * x2) / (3 * y)


def e010a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e010x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e010b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = thtHCG * rho_m

    u1 = 5 + thtHCG * (rho_g + rho_h)
    u2 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    nr = x * y1 * y2 * y3 * z
    dr = 3 * u1 * u2

    return nr / dr


def e010(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e010a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e010b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def hcrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x1 = expTau2overThetaHC(tau2, thtHC) ** -2

    y1 = (1 - expRhoHTau1(rho_h, tau1) ** -1) * (1 - expRhoMTau1(rho_m, tau1) ** -1)
    y2 = (
        (expRhoHTau1(rho_h, tau1) ** -1)
        * (1 - expRhoHTau2(rho_h, tau2) ** -1)
        * (1 - expRhoMTau1(rho_m, tau1) ** -1)
    )
    y3 = (
        (expRhoMTau1(rho_h, tau1) ** -1)
        * (1 - expRhoHTau1(rho_h, tau1) ** -1)
        * (1 - expRhoMTau2(rho_m, tau2) ** -1)
    )

    z1 = (
        (expRhoMTau2(rho_m, tau2) ** -1)
        * (expTau2overThetaHC(tau2, thtHC) ** -2)
        * thtHC
    )
    z2 = (expTau2overThetaHC(tau2, thtHC) * expRhoHTau2(rho_h, tau2) ** -1) - 1

    w1 = (expTau2overThetaHC(tau2, thtHC) ** -2) * thtHC
    w2 = 1 - (
        expTau2overThetaHC(tau2, thtHC)
        * (expRhoHTau2(rho_h, tau2) ** -1)
        * (expRhoMTau2(rho_m, tau2) ** -1)
    )

    u1 = (
        (expRhoHTau2(rho_h, tau2) ** -1)
        * (expTau2overThetaHC(tau2, thtHC) ** -2)
        * thtHC
    )
    u2 = (expTau2overThetaHC(tau2, thtHC) * expRhoMTau2(rho_m, tau2) ** -1) - 1

    d1 = thtHC * rho_h - 1
    d2 = thtHC * (rho_h + rho_m) - 1

    t1 = x1 * (y1 + y2 + y3)
    t2 = rho_h * ((z1 * z2) / d1 + (w1 * w2) / d2)
    t3 = rho_m * ((u1 * u2) / d1 + (w1 * w2) / d2)

    return t1 + t2 + t3


def e110x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y = 3 + thtHCG * rho_g
    return x / (3 * y)


def e110a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hcrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e110x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e110y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y = thtHCG * rho_m
    z = 6 + thtHCG * (rho_g + rho_m)

    u1 = 3 + thtHCG * rho_g
    u2 = 3 + thtHCG * (rho_g + rho_m)
    u3 = 5 + thtHCG * (rho_g + rho_m)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def e110b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e110y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e110z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1

    y = thtHCG * rho_h

    z1 = 3 + thtHCG * rho_g
    z2 = 5 + thtHCG * (rho_g + rho_h)

    nr = x1 * x2 * y
    dr = 3 * z1 * z2

    return nr / dr


def e110c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e110z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e110d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m

    w1 = 6 + thtHCG * rho_m
    w2 = 13 + thtHCG * rho_m
    w3 = 19 + thtHCG * rho_h + 3 * thtHCG * rho_m

    nr = (
        x
        * y1
        * y2
        * y3
        * z1
        * z2
        * (
            45
            + thtHCG
            * (2 * thtHCG * (rho_g ** 2) + rho_h * w1 + rho_m * w2 + rho_g * w3)
        )
    )

    u1 = 3 + thtHCG * rho_g
    u2 = 5 + thtHCG * (rho_g + rho_h)
    u3 = 3 + thtHCG * (rho_g + rho_m)
    u4 = 5 + thtHCG * (rho_g + rho_m)
    u5 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    dr = 3 * u1 * u2 * u3 * u4 * u5

    return nr / dr


def e110(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e110a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e110b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = e110c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = e110d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def e101x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = 3 + thtHCG * rho_m
    return (x * y) / (3 * z)


def e101a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e101x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e101y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y = thtHCG * rho_g
    z = 6 + thtHCG * (rho_g + rho_m)

    u1 = 3 + thtHCG * rho_m
    u2 = 3 + thtHCG * (rho_g + rho_m)
    u3 = 5 + thtHCG * (rho_g + rho_m)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def e101b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e101y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e101c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    w = thtHCG * rho_h
    u = 6 + thtHCG * (rho_h + rho_m)

    v1 = 3 + thtHCG * rho_m
    v2 = 3 + thtHCG * (rho_h + rho_m)
    v3 = 5 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w * u
    dr = 3 * v1 * v2 * v3

    return nr / dr


def e101d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m
    z3 = thtHCG * rho_g

    w1 = (z1 ** 2) * (6 + z2)
    w2 = (z3 ** 2) * (6 + z1 + z2)
    w3 = z1 * (63 + 28 * z2 + 3 * (z2 ** 2))
    w4 = 2 * (90 + 63 * z2 + 14 * (z2 ** 2) + z2 ** 3)
    w5 = z3 * (63 + z1 ** 2 + 28 * z2 + 3 * (z2 ** 2) + (4 * z1) * (4 + z2))

    nr = x * y1 * y2 * y3 * z1 * z3 * (w1 + w2 + w3 + w4 + w5)

    u1 = 3 + z2
    u2 = 3 + z1 + z2 + z3
    u3 = 15 + z3 ** 2 + 8 * z2 + z2 ** 2 + 2 * z3 * (4 + z2)
    u4 = 15 + z3 ** 2 + 8 * z2 + z2 ** 2 + 2 * z1 * (4 + z2)

    dr = 3 * u1 * u2 * u3 * u4

    return nr / dr


def e101(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e101a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e101b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = e101c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = e101d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def e011x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = 3 + thtHCG * rho_h
    return (x * y) / (3 * z)


def e011a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e011x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e011y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1

    y = thtHCG * rho_g

    z1 = 3 + thtHCG * rho_h
    z2 = 5 + thtHCG * rho_g + thtHCG * rho_h

    nr = x1 * x2 * y
    dr = 3 * z1 * z2

    return nr / dr


def e011b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e011y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e011c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    w = thtHCG * rho_m
    u = 6 + thtHCG * (rho_h + rho_m)

    v1 = 3 + thtHCG * rho_h
    v2 = 3 + thtHCG * (rho_h + rho_m)
    v3 = 5 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w * u
    dr = 3 * v1 * v2 * v3

    return nr / dr


def e011d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y3 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m
    z3 = thtHCG * rho_g

    w = 45 + thtHCG * (
        19 * rho_h
        + 13 * rho_m
        + thtHCG * (rho_h + rho_m) * (2 * rho_h + rho_m)
        + rho_g * (6 + thtHCG * (rho_h + rho_m))
    )

    nr = x * y1 * y2 * y3 * z2 * z3 * w

    u1 = 3 + z1
    u2 = 5 + thtHCG * (rho_h + rho_g)
    u3 = 3 + thtHCG * (rho_h + rho_m)
    u4 = 5 + thtHCG * (rho_h + rho_m)
    u5 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * u1 * u2 * u3 * u4 * u5

    return nr / dr


def e011(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e011a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e011b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = e011c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = e011d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def e111x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y = 3 + thtHCG * rho_g
    return (1 / 9) * (1 - (3 * x) / y)


def e111a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hcrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e111y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = thtHCG * rho_m
    w = 3 + thtHCG * rho_m
    return (x * y * z) / (9 * w)


def e111b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e111z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = thtHCG * rho_h
    w = 3 + thtHCG * rho_h
    return (x * y * z) / (9 * w)


def e111c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e111u(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y1 = thtHCG * rho_g
    y2 = thtHCG * rho_m

    z = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * y1 * y2 * (z ** 2)

    w1 = 3 + thtHCG * rho_g
    w2 = 3 + thtHCG * rho_m
    w3 = 3 + thtHCG * (rho_m + rho_g)
    w4 = 5 + thtHCG * (rho_m + rho_g)

    dr = 9 * w1 * w2 * w3 * w4

    return nr / dr


def e111d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111u(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e111v(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1

    y1 = thtHCG * rho_g
    y2 = thtHCG * rho_h

    z = 6 + thtHCG * (rho_h + rho_g)

    nr = x1 * x2 * y1 * y2 * z

    w1 = 3 + thtHCG * rho_g
    w2 = 3 + thtHCG * rho_h
    w3 = 5 + thtHCG * (rho_h + rho_g)

    dr = 9 * w1 * w2 * w3
    return nr / dr


def e111e(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111v(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def e111f(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    w1 = thtHCG * rho_h
    w2 = thtHCG * rho_m

    u = 6 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w1 * w2 * (u ** 2)

    v1 = 3 + thtHCG * rho_h
    v2 = 3 + thtHCG * rho_m
    v3 = 3 + thtHCG * (rho_h + rho_m)
    v4 = 5 + thtHCG * (rho_h + rho_m)

    dr = 9 * v1 * v2 * v3 * v4

    return nr / dr


def e111g1(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_g
    z2 = 3 + thtHCG * (rho_m + rho_g)
    z3 = 5 + thtHCG * (rho_m + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def e111g2(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_m
    z2 = 3 + thtHCG * (rho_m + rho_g)
    z3 = 5 + thtHCG * (rho_m + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def e111g3(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_h + rho_m)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_m
    z2 = 3 + thtHCG * (rho_h + rho_m)
    z3 = 5 + thtHCG * (rho_h + rho_m)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def e111g4(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_h + rho_m)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_h
    z2 = 3 + thtHCG * (rho_h + rho_m)
    z3 = 5 + thtHCG * (rho_h + rho_m)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def e111g5(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    nr = x1 * x2 * x3

    z1 = 3 + thtHCG * rho_h
    z2 = 5 + thtHCG * (rho_h + rho_g)
    z3 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3

    return nr / dr


def e111g6(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    nr = x1 * x2 * x3

    z1 = 3 + thtHCG * rho_g
    z2 = 5 + thtHCG * (rho_h + rho_g)
    z3 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3

    return nr / dr


def e111w(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y3 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    return (1 / 3) * x * y1 * y2 * y3


def e111g(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    w = e111w(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)

    x = e111g1(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111g2(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = e111g3(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    t = e111g4(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    u = e111g5(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    v = e111g6(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return w * (x + y + z + t + u + v)


def e111(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = e111a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = e111c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    t = e111d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    u = e111e(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    v = e111f(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = e111g(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + t + u + v + w


def k001a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = (
        (expTau2overThetaHC(tau2, thtHC) ** -1)
        * (expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1)
        * (expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1)
    )
    y = 1 - (expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1)
    z = 5 + thtHCG * (rho_h + rho_m)
    return (x * y) / (3 * z)


def k001b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y3 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    z = thtHCG * rho_g

    w1 = 5 + thtHCG * (rho_h + rho_m)
    w2 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    nr = x * y1 * y2 * y3 * z
    dr = 3 * w1 * w2

    return nr / dr


def k001(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k001a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k001b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def k100x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = (expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1) * (
        expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    )
    y = 3 + thtHCG * (rho_m + rho_g)
    z = 5 + thtHCG * (rho_m + rho_g)
    return x / (y * z)


def k100a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k100x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k100b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = thtHCG * rho_h

    u1 = 3 + thtHCG * (rho_g + rho_m)
    u2 = 5 + thtHCG * (rho_g + rho_m)
    u3 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    nr = x * y1 * y2 * y3 * z
    dr = u1 * u2 * u3

    return nr / dr


def k100(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k100a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k100b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def k010x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = (expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1) * (
        expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    )
    y = 3 / (2 * (3 + thtHCG * (rho_h + rho_g)))
    z = 1 / (2 * (5 + thtHCG * (rho_h + rho_g)))
    return (1 / 3) * x * (y - z)


def k010a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k010x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k010b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = thtHCG * rho_m
    w = 6 + thtHCG * (rho_h + rho_g)

    u1 = 3 + thtHCG * (rho_g + rho_h)
    u2 = 5 + thtHCG * (rho_g + rho_h)
    u3 = 3 + thtHCG * (rho_g + rho_h + rho_m)

    nr = x * y1 * y2 * y3 * z * w
    dr = 3 * u1 * u2 * u3

    return nr / dr


def k010(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k010a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k010b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y


def k110x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y = 3 + thtHCG * rho_g
    return x / (3 * y)


def k110a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hcrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k110x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k110y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y = thtHCG * rho_m
    z = 6 + thtHCG * (rho_g + rho_m)

    u1 = 3 + thtHCG * rho_g
    u2 = 3 + thtHCG * (rho_g + rho_m)
    u3 = 5 + thtHCG * (rho_g + rho_m)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def k110b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k110y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k110z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    x2 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    y = thtHCG * rho_h
    z = 6 + thtHCG * (rho_g + rho_h)

    u1 = 3 + thtHCG * rho_g
    u2 = 3 + thtHCG * (rho_g + rho_h)
    u3 = 5 + thtHCG * (rho_g + rho_h)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def k110c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k110z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k110d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m
    z3 = thtHCG * rho_g

    w1 = 2 * (z3 ** 3)
    w2 = (z1 ** 2) * (6 + z2)
    w3 = z1 * (7 + z2) * (9 + z2)
    w4 = (3 * z2) * (21 + 2 * z2)
    w5 = (z3 ** 2) * (28 + 3 * z1 + 3 * z2)
    w6 = z3 * (126 + z1 ** 2 + 4 * z1 * (7 + z2) + z2 * (28 + z2))

    nr = x * y1 * y2 * y3 * z1 * z2 * (180 + w1 + w2 + w3 + w4 + w5 + w6)

    u1 = 3 + z3
    u2 = 3 + z1 + z3
    u3 = 5 + z1 + z3
    u4 = 3 + z2 + z3
    u5 = 5 + z2 + z3
    u6 = 3 + z1 + z2 + z3

    dr = 3 * u1 * u2 * u3 * u4 * u5 * u6

    return nr / dr


def k110(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k110a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k110b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = k110c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = k110d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def k101x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = 3 + thtHCG * rho_m
    return (x * y) / (3 * z)


def k101a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k101x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k101y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y = thtHCG * rho_g
    z = 6 + thtHCG * (rho_g + rho_m)

    u1 = 3 + thtHCG * rho_m
    u2 = 3 + thtHCG * (rho_g + rho_m)
    u3 = 5 + thtHCG * (rho_g + rho_m)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def k101b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k101y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k101c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    w = thtHCG * rho_h

    u1 = 3 + thtHCG * rho_m
    u2 = 5 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w
    dr = 3 * u1 * u2

    return nr / dr


def k101d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y3 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m
    z3 = thtHCG * rho_g

    w1 = z3 ** 2
    w2 = z1 * (6 + z2)
    w3 = z2 * (19 + 2 * z2)
    w4 = z3 * (13 + z1 + 3 * z2)

    nr = x * y1 * y2 * y3 * z1 * z3 * (45 + w1 + w2 + w3 + w4)

    u1 = 3 + z2
    u2 = 3 + z2 + z3
    u3 = 5 + z2 + z3
    u4 = 5 + z1 + z2
    u5 = 3 + z1 + z2 + z3

    dr = 3 * u1 * u2 * u3 * u4 * u5

    return nr / dr


def k101(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k101a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k101b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = k101c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = k101d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def k011x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = 3 + thtHCG * rho_h
    return (x * y) / (3 * z)


def k011a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k011x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k011y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1

    y = thtHCG * rho_g
    z = 6 + thtHCG * (rho_g + rho_h)

    u1 = 3 + thtHCG * rho_h
    u2 = 3 + thtHCG * (rho_g + rho_h)
    u3 = 5 + thtHCG * (rho_g + rho_h)

    nr = x1 * x2 * y * z
    dr = 3 * u1 * u2 * u3

    return nr / dr


def k011b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k011y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k011c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    w = thtHCG * rho_m

    u1 = 3 + thtHCG * rho_h
    u2 = 5 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w
    dr = 3 * u1 * u2

    return nr / dr


def k011d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y3 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    z1 = thtHCG * rho_h
    z2 = thtHCG * rho_m
    z3 = thtHCG * rho_g

    w1 = z3 ** 3
    w2 = 2 * (z3 ** 2) * (9 + 2 * z1 + z2)
    w3 = z3 * (5 * (5 + z1) ** 2 + z2 * (27 + 5 * z1) + z2 ** 2)
    w4 = (54 + 2 * z1 * (11 + z1) + z2 * (6 + z1)) * (5 + z1 + z2)

    nr = x * y1 * y2 * y3 * z2 * z3 * (w1 + w2 + w3 + w4)

    u1 = 3 + z1
    u2 = 3 + z1 + z3
    u3 = 5 + z1 + z3
    u4 = 5 + z1 + z2
    u5 = 3 + z1 + z2 + z3
    u6 = 5 + z1 + z2 + z3

    dr = 3 * u1 * u2 * u3 * u4 * u5 * u6

    return nr / dr


def k011(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k011a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k011b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = k011c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = k011d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + w


def k111x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    y = 3 + thtHCG * rho_g
    return (1 / 9) * (1 - (3 * x) / y)


def k111a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hcrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111x(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k111y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = thtHCG * rho_m
    w = 3 + thtHCG * rho_m
    return (x * y * z) / (9 * w)


def k111b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111y(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k111z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    z = thtHCG * rho_h
    w = 3 + thtHCG * rho_h
    return (x * y * z) / (9 * w)


def k111c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = e111z(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k111u(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    y1 = thtHCG * rho_g
    y2 = thtHCG * rho_m

    z = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * y1 * y2 * (z ** 2)

    w1 = 3 + thtHCG * rho_g
    w2 = 3 + thtHCG * rho_m
    w3 = 3 + thtHCG * (rho_m + rho_g)
    w4 = 5 + thtHCG * (rho_m + rho_g)

    dr = 9 * w1 * w2 * w3 * w4

    return nr / dr


def k111d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = hrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111u(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k111v(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1
    x2 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1

    y1 = thtHCG * rho_g
    y2 = thtHCG * rho_h

    z = 6 + thtHCG * (rho_h + rho_g)

    nr = x1 * x2 * y1 * y2 * (z ** 2)

    w1 = 3 + thtHCG * rho_g
    w2 = 3 + thtHCG * rho_h
    w3 = 3 + thtHCG * (rho_h + rho_g)
    w4 = 5 + thtHCG * (rho_h + rho_g)

    dr = 9 * w1 * w2 * w3 * w4
    return nr / dr


def k111e(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = mrec(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111v(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x * y


def k111f(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1

    z = 1 - expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    w1 = thtHCG * rho_h
    w2 = thtHCG * rho_m

    u = 6 + thtHCG * (rho_h + rho_m)

    nr = x * y1 * y2 * z * w1 * w2 * u

    v1 = 3 + thtHCG * rho_h
    v2 = 3 + thtHCG * rho_m
    v3 = 5 + thtHCG * (rho_h + rho_m)

    dr = 9 * v1 * v2 * v3

    return nr / dr


def k111g1(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_g
    z2 = 3 + thtHCG * (rho_m + rho_g)
    z3 = 5 + thtHCG * (rho_m + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def k111g2(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_m + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_m
    z2 = 3 + thtHCG * (rho_m + rho_g)
    z3 = 5 + thtHCG * (rho_m + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def k111g3(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_h + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_g
    z2 = 3 + thtHCG * (rho_h + rho_g)
    z3 = 5 + thtHCG * (rho_h + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def k111g4(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    y = 6 + thtHCG * (rho_h + rho_g)

    nr = x1 * x2 * x3 * y

    z1 = 3 + thtHCG * rho_h
    z2 = 3 + thtHCG * (rho_h + rho_g)
    z3 = 5 + thtHCG * (rho_h + rho_g)
    z4 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3 * z4

    return nr / dr


def k111g5(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    nr = x1 * x2 * x3

    z1 = 3 + thtHCG * rho_h
    z2 = 5 + thtHCG * (rho_h + rho_m)
    z3 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3

    return nr / dr


def k111g6(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    x1 = thtHCG * rho_h
    x2 = thtHCG * rho_m
    x3 = thtHCG * rho_g

    nr = x1 * x2 * x3

    z1 = 3 + thtHCG * rho_m
    z2 = 5 + thtHCG * (rho_h + rho_m)
    z3 = 3 + thtHCG * (rho_h + rho_m + rho_g)

    dr = 3 * z1 * z2 * z3

    return nr / dr


def k111w(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = expTau2overThetaHC(tau2, thtHC) ** -1

    y1 = expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1
    y2 = expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    y3 = expRhoGTau1PlusTau2(rho_g, tau1, tau2) ** -1

    return (1 / 3) * x * y1 * y2 * y3


def k111g(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):

    w = k111w(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)

    x = k111g1(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111g2(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = k111g3(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    t = k111g4(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    u = k111g5(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    v = k111g6(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return w * (x + y + z + t + u + v)


def k111(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG):
    x = k111a(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    y = k111b(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    z = k111c(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    t = k111d(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    u = k111e(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    v = k111f(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    w = k111g(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return x + y + z + t + u + v + w


""" Return transition rates """


def get_s(rho_h, rho_m, rho_g):
    x = expTau2overThetaHC(tau2, thtHC) ** -1
    y = (expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1) * (
        expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    )
    z = expTau2overThetaHC(tau2, thtHC)
    w = expRhoHTau2(rho_h, tau2) * expRhoMTau2(rho_m, tau2)
    u = expTau2overThetaHC(tau2, thtHC) - 1
    v = thtHC * (rho_h + rho_m) - 1
    return x * (1 + (y * (z - w)) / (u * v)) / 3


def get_u(rho_h, rho_m, rho_g):
    x = expTau2overThetaHC(tau2, thtHC) ** -1
    y = (expRhoHTau1PlusTau2(rho_h, tau1, tau2) ** -1) * (
        expRhoMTau1PlusTau2(rho_m, tau1, tau2) ** -1
    )
    z = (
        (expTau2overThetaHC(tau2, thtHC) ** -1)
        * (expRhoHTau1(rho_h, tau1) ** -1)
        * (expRhoMTau1(rho_m, tau1) ** -1)
    )
    u = thtHC * (rho_h + rho_m) - 1
    return 1 - x + y / u - z / u


def get_v1(rho_h, rho_m, rho_g):
    dr = phg(tau2, thtHC)
    x001 = e001(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x100 = e100(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x010 = e010(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x110 = e110(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x101 = e101(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x011 = e011(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x111 = e111(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    return (x001 + x100 + x010 + x110 + x101 + x011 + x111) / dr


def get_v2(rho_h, rho_m, rho_g):
    dr = phc2(tau2, thtHC)

    x001 = k001(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x100 = k100(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x010 = k010(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x110 = k110(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x101 = k101(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x011 = k011(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)
    x111 = k111(rho_h, rho_m, rho_g, tau1, tau2, thtHC, thtHCG)

    return (x001 + x100 + x010 + x110 + x101 + x011 + x111) / dr


""" Return initial probabilities (stationary distribution). """


def get_psi():
    psi = 1 - term
    other = (1 - psi) / 3
    return np.log(psi), np.log(other)


""" Return parameters of Coal-HMM model given length of the alignment sequence. """


def get_params(length):
    rho = 0.015 * 10 ** -6 * length
    s = get_s(rho, rho, rho)
    u = get_u(rho, rho, rho)
    v1 = get_v1(rho, rho, rho)
    v2 = get_v2(rho, rho, rho)

    # s = 0.002897638
    # u = 0.005639278
    # v1 = 0.006007024
    # v2 = 0.00656248

    psi, other = get_psi()
    a = tau1 + thtHC - tau2 * term / (1 - term)
    b = tau2 + tau2 * term / (1 - term)
    c = div_time * 2 - a - b
    a_t = tau1 + tau2 + thtHCG / 3
    b_t = thtHCG
    c_t = div_time * 2 - a_t - b_t
    return (s, u, v1, v2), (psi, other, other, other), (a, b, c, a_t, b_t, c_t), mu
