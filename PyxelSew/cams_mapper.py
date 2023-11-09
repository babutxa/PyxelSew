from . import camera_model as cam

class CamsMapper:
    def __init__(self, fromCamera: cam.CameraModel, toCamera: cam.CameraModel):
        self.m_fromCamera = fromCamera
        self.m_toCamera = toCamera

    def  map(self, pTex):
        pField = self.m_fromCamera.tex_2_field(pTex)
        return self.m_toCamera.field_2_tex(pField)

    def map_inf(self, pTex):
        bWorld = self.m_fromCamera.tex_2_wordlBearing(pTex)
        return self.m_toCamera.worldBearing_2_tex(bWorld)
    

