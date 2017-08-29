import unittest
import digit_reco
from json import dumps, loads

from base64 import b64encode

CONTENT_TYPE_JSON = "application/json"

class TestDigitReco(unittest.TestCase):

    def setUp(self):
        digit_reco.app.testing = True
        self.app = digit_reco.app.test_client()
        encoded_image = b64encode(open("9.png", "rb").read()).decode()
        self.request_data = dumps(dict(image=encoded_image))

    def tearDown(self):
        pass

    def test_helloworld(self):
        resp = self.app.get("/")
        self.assertEqual("200 OK", resp.status)
        self.assertEqual(b"Hello World!", resp.data)

    def test_recognize(self):
        resp = self.app.post("/recognize", data=self.request_data, content_type=CONTENT_TYPE_JSON)
        expected_resp_body = dict(label='9', status=200)
        actual_resp_body = loads(resp.data.decode())

        self.assertEqual("200 OK", resp.status)
        self.assertEqual(expected_resp_body, actual_resp_body)

    def test_invalid_request_type(self):
        resp = self.app.post("/recognize", data=self.request_data)
        actual_resp_body = loads(resp.data.decode())
        expected_resp_body = dict(description="only content-type: application/json is accepted", status=400)

        self.assertEqual("400 BAD REQUEST", resp.status)
        self.assertEqual(expected_resp_body, actual_resp_body)
        print(resp.data)

    def test_invalid_image(self):
        resp = self.app.post("/recognize", data=self.request_data, content_type=CONTENT_TYPE_JSON)
        actual_resp_body = loads(resp.data.decode())
        expected_resp_body = dict(description="field 'image' is missing or has empty content", status=400)

        self.assertEqual("400 BAD REQUEST", resp.status)
        self.assertEqual(expected_resp_body, actual_resp_body)

    def clean_json(self, resp_json):
        """
        remove \n, whitespace from response json string, making unit test easier
        :return:
        """
        return resp_json.decode().replace("\n", "").replace(" ", "").replace(":", ": ")


if __name__ == '__main__':
    unittest.main()
