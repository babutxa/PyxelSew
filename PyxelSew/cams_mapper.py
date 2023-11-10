from . import camera_model as cam

class CamsMapper:
    def __init__(self, fromCamera: cam.CameraModel, toCamera: cam.CameraModel):
        self.m_fromCamera = fromCamera
        self.m_toCamera = toCamera
        self.m_homoNormUndistFromTo = toCamera._getHomoField2NormUndist().dot(fromCamera._getHomoField2NormUndist().inv())

    def  map(self, pTex):
        if self.m_fromCamera == self.m_toCamera:
            return pTex
        
        bFromCamera = self.m_fromCamera._tex_2_camera(pTex)
        bToCamera = self.m_homoNormUndistFromTo.dot(bFromCamera)
        pNorm = self.m_toCamera._camera_2_normalized(bToCamera)
        return self.m_toCamera._normalized_2_tex(pNorm)

    def map_inf(self, pTex):
        bWorld = self.m_fromCamera.tex_2_wordlBearing(pTex)
        return self.m_toCamera.worldBearing_2_tex(bWorld)
    
