import os
import shutil
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from kaggle.rle import decode, encode, encode_folder


class RLETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = "./images_to_encode"
        os.makedirs(folder)
        cls.folder = folder
        img = (np.random.random((32, 32, 3)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder, "img.png"), img)
        cls.img = img
        cls.folder = folder

    def test_encode_decode(self):
        encoded_img = encode(self.img)
        assert isinstance(encoded_img, bytes)
        decoded_img = decode(encoded_img)
        assert isinstance(decoded_img, np.ndarray)
        assert np.allclose(self.img, decoded_img.reshape(*self.img.shape))

    def test_encode_folder(self):
        f = os.path.join(self.folder, "solution.csv")
        encode_folder(self.folder, f)
        df = pd.read_csv(f)
        assert df.shape[0] == 1
        decoded_img = decode(eval(df.iloc[0]["rle"]))
        assert isinstance(decoded_img, np.ndarray)
        assert np.allclose(self.img, decoded_img.reshape(*self.img.shape))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)
