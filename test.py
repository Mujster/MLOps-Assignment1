import unittest
from app import app
import json

class TestApp(unittest.TestCase):
    def test_home_page(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b'<!DOCTYPE html>' in response.data) 

    def test_prediction_api(self):
        tester = app.test_client(self)
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
        response = tester.post(
            '/predict',
            data=json.dumps(data),
            content_type='application/json',
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        predicted_price = json.loads(response.data)['predicted_price']
        self.assertIsInstance(predicted_price, float)


if __name__ == '__main__':
    unittest.main()
