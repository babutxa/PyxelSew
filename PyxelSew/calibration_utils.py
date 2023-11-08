import numpy as np

def fromCartesianToCylindric(pCamera):
    phi = np.atan2(pCamera[0], pCamera[2])
    return np.array([phi, pCamera[1] / pCamera[2] * np.cos(phi)])

def fromCylindricToCartesianNormalized(pCylindric):
    phi = pCylindric[0]
    h = pCylindric[1]
    return np.array([np.sin(phi), h, np.cos(phi)])

def fromCartesianToEspheric(pCamera):
    r = np.linalg.norm(pCamera)
    theta = np.atan2(pCamera[0], pCamera[2])
    phi = np.arcsin(pCamera[1] / r)
    return np.array([theta, phi])

def fromEsphericToCartesianNormalized(pEspheric):
    theta = pEspheric[0] # longitude
    phi = pEspheric[1]   # latitude
    x = np.cos(phi) * np.cos(theta); # -- > zCamera
    y = np.cos(phi) * np.sin(theta); # -- > xCamera
    z = np.sin(phi);              # -- > yCamera
    return np.array([y, z, x])

