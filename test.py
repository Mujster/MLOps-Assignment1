import unittest
from app import app  # Import your Flask app from app.py
import json

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b'<!DOCTYPE html>' in response.data)

    def test_prediction_api(self):
        data = {
            'area': 100,
            'bedrooms': 2,
            'bathrooms': 1,
            'stories': 1,
            'mainroad': 'yes',
            'guestroom': 'no',
            'basement': 'no',
            'hotwaterheating': 'yes',
            'airconditioning': 'no',
            'parking': 1,
            'prefarea': 'yes',
            'furnishingstatus': 'furnished',
            'buildingage': 10
        }
        response = self.app.post(
            '/predict',
            data=data,
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        response_json = json.loads(response.data)
        self.assertIn('predicted_price', response_json)
        self.assertIsInstance(response_json['predicted_price'], float)


if __name__ == '__main__':
    unittest.main()
