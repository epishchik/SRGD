import io
import shutil
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.main import app


class APITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path_to_extra = Path(__file__).parents[1] / "api/extra"
        shutil.copytree(path_to_extra, "./extra")
        cls.api_client = TestClient(app)

    def test_get_info(self):
        response = self.api_client.get("/info")
        assert response.status_code == 200

    def test_configurate_model_by_name(self):
        response_correct = self.api_client.post(
            "/configure_model/name", params={"config_name": "pretrained/EMT_x4"}
        )
        response_incorrect = self.api_client.post(
            "/configure_model/name", params={"config_name": "EMT_x4"}
        )
        assert response_correct.status_code == 200
        assert response_incorrect.status_code == 400

    def test_upscale_example(self):
        response = self.api_client.post("/upscale/example")
        upscaled_bytes = io.BytesIO(response.content)
        upscaled_image = Image.open(upscaled_bytes).convert("RGB")
        assert response.status_code == 200
        assert upscaled_image.size == (2560, 1440)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./extra")
