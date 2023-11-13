import unittest
import numpy as np
import xml.etree.ElementTree as ET

from PyxelSew import camera_model as cam

def read_camera_model_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    cx = float(root.find('cx').text)
    cy = float(root.find('cy').text)
    fc = float(root.find('Zoom').text)
    ar = float(root.find('AspectRatio').text)
    skew = float(root.find('Skew').text)
    d_type = int(root.find('dType').text)
    k1 = float(root.find('k1').text)
    k2 = float(root.find('k2').text)
    p1 = float(root.find('p1').text)
    p2 = float(root.find('p2').text)
    k3 = float(root.find('k3').text)
    k4 = float(root.find('k4').text)
    pan = float(root.find('Pan').text)
    tilt = float(root.find('Tilt').text)
    roll = float(root.find('Roll').text)
    tx = float(root.find('Tx').text)
    ty = float(root.find('Ty').text)
    tz = float(root.find('Tz').text)
    return cam.CameraModel([fc, ar, cx, cy, skew], [pan, tilt, roll, tx, ty, tz], [k1, k2, p1, p2, k3, k4], d_type)

class TestCameraModel(unittest.TestCase):
    def setUp(self):
        # create a test camera model
        xml_string = '<ModelCamera computed="true"><cx>0.4999983311</cx><cy>0.500000258</cy><Zoom>0.5694238773</Zoom><AspectRatio>1.77777778</AspectRatio><Skew>0</Skew><dType>1</dType><k1>-0.004746594806</k1><k2>0.003078619124</k2><p1>0.004553997797</p1><p2>0.01030821877</p2><k3>-0.02222773463</k3><k4>1.8</k4><Pan>-37.57640603</Pan><Tilt>109.1734819</Tilt><Roll>-12.66977705</Roll><Tx>26.56035287</Tx><Ty>-9.775630904</Ty><Tz>37.23183347</Tz></ModelCamera>'
        self.camera_model = read_camera_model_from_xml(xml_string)

    def test_getK(self):
        # test the getK method
        expected_K = np.array([[0.5694238773, 0.0, 0.4999983311],
                               [0.0, 0.5694238773 * 1.77777778, 0.500000258],
                               [0.0, 0.0, 1.0]])
        self.assertTrue(np.allclose(self.camera_model.getK(), expected_K))

    def test_getR(self):
        # test the getR method
        expected_R = np.array([[9.99825857e-01, -1.86596548e-02,  2.70723255e-04],
                               [1.86590715e-02,  9.99344525e-01, -3.10219159e-02],
                               [3.08312438e-04, 3.10215651e-02,  9.99518668e-01]])
        self.assertTrue(np.allclose(self.camera_model.getR(), expected_R))

    def test_getT(self):
        # test the getT method
        expected_T = np.array([26.56035287, -9.775630904, 37.23183347])
        self.assertTrue(np.allclose(self.camera_model.getT(), expected_T))

    def test_setT(self):
        # test the setT method
        new_T = np.array([1.1, 2.2, 3.3])
        self.camera_model.setT(new_T)
        self.assertTrue(np.allclose(self.camera_model.getT(), new_T))

    def test_getC(self):
        # test the getC method
        expected_C = np.array([-26.38480241, 9.10984049, -37.5243619])
        self.assertTrue(np.allclose(self.camera_model.getC(), expected_C))

    def test_setC(self):
        # test the setC method
        new_C = np.array([1.1, 2.2, 3.3])
        self.camera_model.setC(new_C)
        self.assertTrue(np.allclose(self.camera_model.getC(), new_C))

    def test_getHomoField2NormUndist(self):
        # test the getK method
        expected_H = np.array([[9.99825857e-01, -1.86596548e-02,  2.65603529e+01],
                               [1.86590715e-02,  9.99344525e-01, -9.77563090e+00],
                               [3.08312438e-04,  3.10215651e-02,  3.72318335e+01]])
        self.assertTrue(np.allclose(self.camera_model.getHomoField2NormUndist(), expected_H))

    def test_field_2_tex_2_field(self):
        # test the field_2_tex and tex_2_field methods
        pField = np.array([5.1, 3.2])
        pTex = self.camera_model.field_2_tex(pField)
        pField2 = self.camera_model.tex_2_field(pTex)
        print("pField: ", pField)
        print("pField2: ", pField2)
        self.assertTrue(np.allclose(pField, pField2))
    
    
    def test_tex_2_field_2_tex(self):
        # test the field_2_tex and tex_2_field methods
        pTex = np.array([0.1, 0.2])
        pField = self.camera_model.tex_2_field(pTex)
        pTex2 = self.camera_model.field_2_tex(pField) 
        print("pTex: ", pTex)
        print("pTex2: ", pTex2)
        self.assertTrue(np.allclose(pTex, pTex2))


    def test_wordlPoint_2_tex(self):
        # TODOalba test the wordlPoint_2_tex method
        return True
    

    def test_worldBearing_2_tex(self):
        # TODOalba test the worldBearing_2_tex method
        return True
    

    def test_tex_2_wordlBearing(self): 
        # TODOalba test the tex_2_wordlBearing method
        return True
    

    def test_worldPoint_2_worldBearing(self):
        # test the worldPoint_2_worldBearing method
        pWorld0 = np.array([0.0, 0.0, 0.0])
        expected_bWorld0 = -self.camera_model.getC()
        self.assertTrue(np.allclose(self.camera_model.worldPoint_2_worldBearing(pWorld0), expected_bWorld0))

        pWorld = np.array([1.1, 2.2, 3.3])
        expected_bWorld = np.array([27.48480241, -6.90984049, 40.8243619])
        self.assertTrue(np.allclose(self.camera_model.worldPoint_2_worldBearing(pWorld), expected_bWorld))

if __name__ == '__main__':
    unittest.main()