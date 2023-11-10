from enum import Enum
import numpy as np
import math
import cv2

from . import calibration_utils as cu

class iParam(Enum):
    FC = 0
    AR = 1
    CX = 2
    CY = 3
    SKEW = 4
    I_NUM_PARAMS = 5

class eParam(Enum):
    PAN = 0
    TILT = 1
    ROLL = 2
    TX = 3
    TY = 4
    TZ = 5
    E_NUM_PARAMS = 6

class dParam(Enum):
    K1 = 0
    K2 = 1
    P1 = 2
    P2 = 3
    K3 = 4
    K4 = 5
    D_NUM_PARAMS = 6

class ProjectionDistortionType(Enum):
    perspectiveNone = 0
    perspectiveRadial = 1
    fisheyeRadial = 2
    cylindricNone = 3
    esphericNone = 4

class CameraModel:
    def __init__(self, intrinsics=[0]*iParam.I_NUM_PARAMS.value, extrinsics=[0]*eParam.E_NUM_PARAMS.value,
                 distortion=[0]*dParam.D_NUM_PARAMS.value, pd_type=ProjectionDistortionType.perspectiveNone):
        self.m_intrinsics = np.array(intrinsics)
        self.m_extrinsics = np.array(extrinsics)
        self.m_distortion = np.array(distortion)
        self.m_projAndDistType = pd_type

    def getK(self, out_size=(1., 1.)):
        fc = self.m_intrinsics(iParam.FC.value)
        ar = self.m_intrinsics(iParam.AR.value)
        cx = self.m_intrinsics(iParam.CX.value)
        cy = self.m_intrinsics(iParam.CY.value)
        sk = self.m_intrinsics(iParam.SKEW.value)
        width = out_size[0]
        height = out_size[1]    
        return np.array([[fc * width, sk, cx * width],
                         [0, fc * ar * height, cy * height],
                         [0, 0, 1]])
    
    def getR(self):
        pan = self.m_intrinsics[iParam.PAN.value]
        tilt = self.m_intrinsics[iParam.TILT.value]
        roll = self.m_intrinsics[iParam.ROLL.value]
        Rpan = cv2.Rodrigues(np.array([0., 0., pan * math.pi / 180.]))[0]  # Pan around Z-axis
        Rtilt = cv2.Rodrigues(np.array([tilt * math.pi / 180, 0., 0.]))[0]  # Tilt around X-axis
        Rroll = cv2.Rodrigues(np.array([0., 0., roll * math.pi / 180]))[0]  # Roll around Z-axis again (???)
        return np.dot(np.dot(Rroll, Rtilt), Rpan)

    def getT(self):
        return np.array(self.m_extrinsics[eParam.TX.value:eParam.E_NUM_PARAMS.value])

    def setT(self, t):
        self.m_extrinsics[eParam.TX.value:eParam.E_NUM_PARAMS.value] = t

    def getC(self):
        R = self.getR()
        Rinv = np.transpose(R)
        return -Rinv.dot(self.getT())

    def setC(self, c):
        R = self.getR()
        t = -R.dot(c)
        self.setT(t)  # already resets extrinsics dependent things

    
    #######################
    ### transformations ###
    #######################

    def wordlPoint_2_tex(self, pWorld):
        bCamera = self._worldPoint_2_camera(pWorld) # 3D world -> 3D camera
        pNorm = self._camera_2_normalized(bCamera)
        pTex = self._normalized_2_tex(pNorm)
        return pTex

    
    def worldBearing_2_tex(self, bWorld):
        bCamera = self._worldBearing_2_camera(bWorld)
        pNorm = self._camera_2_normalized(bCamera)
        pTex = self._normalized_2_tex(pNorm)
        return pTex

    
    def field_2_tex(self, pField):
        # use Homography to transform field to normalized undistorted coordinates
        homoField2NormUndist = self._getHomoField2NormUndist()
        bCamera = homoField2NormUndist.dot(np.array([pField[0], pField[1], 1.0]))
        pNorm = self._camera_2_normalized(bCamera)
        pTex = self._normalized_2_tex(pNorm)
        return pTex

    
    def tex_2_wordlBearing(self, pTex):
        bCamera = self._tex_2_camera(pTex)
        bWorld = self._camera_2_world(bCamera)
        return bWorld

    
    def tex_2_field(self, pTex):
        bCamera = self._tex_2_camera(pTex)
        pField = self._camera_2_field(bCamera)
        return pField

    
    def worldPoint_2_worldBearing(self, pWorld):
        return pWorld - self.getC()


    # internal stuff - do not use this _functions 
    def __worldPoint_2_camera(self, pWorld):
        return self.getR().dot(pWorld) + self.getT()

    
    def __worldBearing_2_camera(self, bWorld):
        return self.getR().dot(bWorld)

    
    def __camera_2_normalized(self, bCamera):
        # project and distort
        k1 = self.m_distortion[dParam.K1.value]
        k2 = self.m_distortion[dParam.K2.value]
        p1 = self.m_distortion[dParam.P1.value]
        p2 = self.m_distortion[dParam.P2.value]
        k3 = self.m_distortion[dParam.K3.value]
        k4 = self.m_distortion[dParam.K4.value]

        # perspective projection
        if self.m_projAndDistType == ProjectionDistortionType.perspectiveNone:
            return np.array([bCamera[0] / bCamera[2], bCamera[1] / bCamera[2]])

        # cylindric projection
        elif self.m_projAndDistType == ProjectionDistortionType.cylindricNone:
            return cu.fromCartesianToCylindric(bCamera)
            
        # espheric projection
        elif self.m_projAndDistType == ProjectionDistortionType.esphericNone:
            return cu.fromCartesianToEspheric(bCamera)

        # perspective projection + distortion
        elif self.m_projAndDistType == ProjectionDistortionType.perspectiveRadial:     
            pNorm = np.array([bCamera[0] / bCamera[2], bCamera[1] / bCamera[2]])
            r2 = pNorm @ pNorm
            if r2 > k4:
                r2 = k4
            r4 = r2 * r2
            r6 = r4 * r2
            s = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            xDist = pNorm[0] * s + p2 * (r2 + 2.0 * pNorm[0] * pNorm[0]) + p1 * (2.0 * pNorm[0] * pNorm[0])
            yDist = pNorm[1] * s + p1 * (r2 + 2.0 * pNorm[1] * pNorm[1]) + p2 * (2.0 * pNorm[1] * pNorm[1])
            return np.array([xDist, yDist])

        # fisheye projection + distortion
        elif self.m_projAndDistType == ProjectionDistortionType.fisheyeRadial:
            r = np.sqrt(bCamera[0] * bCamera[0]  + bCamera[1]  * bCamera[1])
            theta = np.atan2(r, bCamera[2])
            theta2 = theta * theta
            if(theta2 > p1):
                theta2 = p1
            theta3 = theta2 * theta
            theta5 = theta3 * theta2
            theta7 = theta5 * theta2
            theta9 = theta7 * theta2
            theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9
            cdist = theta_d / r if r > 1e-8 else 1
            return np.array([cdist * bCamera[0], cdist * bCamera[1]])

        return np.array([]) # error return - empty array
    
    
    def __normalized_2_tex(self, pNorm):
        texP = self.getK().dot(np.array([pNorm[0], pNorm[1], 1.0]))
        return np.array([texP[0] / texP[2], texP[1] / texP[2]])

    
    def __tex_2_camera(self, pTex):
        # note: this function returns a bearing in camera coordinates (bCamera)

        # perspective projection
        if self.m_projAndDistType == ProjectionDistortionType.perspectiveNone:
            return self._tex_2_rawNorm(pTex)

        # cylindric projection
        elif self.m_projAndDistType == ProjectionDistortionType.cylindricNone:
            cylindric = self._tex_2_rawNorm(pTex)
            return cu.fromCylindricToCartesianNormalized(cylindric)
            
        # espheric projection
        elif self.m_projAndDistType == ProjectionDistortionType.esphericNone:
            espheric = self._tex_2_rawNorm(pTex)
            return cu.fromEsphericToCartesianNormalized(espheric)

        # perspective projection + distortion (radial)
        elif self.m_projAndDistType == ProjectionDistortionType.perspectiveRadial:
            pRadial = self._undistortPixelPointRadial(pTex)
            return np.array([pRadial[0], pRadial[1], 1.0])

        # fisheye projection + distortion (fisheye)
        elif self.m_projAndDistType == ProjectionDistortionType.fisheyeRadial:
            pFisheye = self._undistortPixelPointFisheye(pTex)
            return np.array([pFisheye[0], pFisheye[1], 1.0])

        return np.array([]) # error return - empty array


    def __camera_2_world(self, bCamera):
        return self.getR().inv().dot(bCamera)


    def __camera_2_field(self, bCamera):
        homoNormUndist2Field = self._getHomoField2NormUndist().inv()
        pField = homoNormUndist2Field.dot(bCamera)
        return np.array([pField[0] / pField[2], pField[1] / pField[2]])


    def __tex_2_rawNorm(self, pTex):
        Kinv = self.getK().inv()
        return Kinv.dot(np.array([pTex[0], pTex[1], 1.0]))


    def __undistortPixelPointRadial(self, pTexDist):
        k1 = self.m_distortion[dParam.K1.value]
        k2 = self.m_distortion[dParam.K2.value]
        p1 = self.m_distortion[dParam.P1.value]
        p2 = self.m_distortion[dParam.P2.value]
        k3 = self.m_distortion[dParam.K3.value]
        distCoeffs = np.array([k1, k2, p1,  p2, k3], dtype=np.float64)
        distorted_points = np.array([[[pTexDist[0], pTexDist[1]]]])
        undistorted_points = cv2.undistortPoints(distorted_points, self.getK(), distCoeffs, None, np.eye(3))
        return np.array([undistorted_points[0,0], undistorted_points[0,1]])


    def __undistortPixelPointFisheye(self, pTexDist):
        k1 = self.m_distortion[dParam.K1.value]
        k2 = self.m_distortion[dParam.K2.value]
        k3 = self.m_distortion[dParam.K3.value]
        k4 = self.m_distortion[dParam.K4.value]
        distCoeffs = np.array([k1, k2, k3, k4], dtype=np.float64)               
        distorted_points = np.array([[[pTexDist[0], pTexDist[1]]]])
        undistorted_points = cv2.fisheye.undistortPoints(distorted_points, self.getK(), distCoeffs, None, np.eye(3))
        return np.array([undistorted_points[0,0], undistorted_points[0,1]])
    

    def __getHomoField2NormUndist(self):
        # Homography to transform field to normalized 'undistorted' coordinates.
        # This homography is valid for all types of cameras, not only the pinhole ones.
        homoField2NormUndist = self.getR()
        homoField2NormUndist[:, -1] = self.getT()
        return homoField2NormUndist

    



    

       



