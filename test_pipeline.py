import os
import pickle
import unittest
import pandas as pd

class TestCKDPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.scaler_path = os.path.join("models", "scaler.pkl")
        cls.model_path = os.path.join("models", "model_gbc.pkl")
        
        # Load artifacts
        cls.assertTrue(cls, os.path.exists(cls.scaler_path), "scaler.pkl missing")
        cls.assertTrue(cls, os.path.exists(cls.model_path), "model_gbc.pkl missing")
        
        with open(cls.scaler_path, "rb") as f:
            cls.scaler = pickle.load(f)
        with open(cls.model_path, "rb") as f:
            cls.model = pickle.load(f)

    def test_artifacts_loaded(self):
        """Verify scaler and model instances are non-null."""
        self.assertIsNotNone(self.scaler)
        self.assertIsNotNone(self.model)

    def test_healthy_patient_prediction(self):
        """Test prediction for a healthy patient profile -> Expect class 0 (Not CKD)."""
        df_healthy = pd.DataFrame([{
            'age': 30, 'bp': 80, 'sg': 1.020, 'al': 0.0, 'hemo': 15.4, 'sc': 1.2,
            'htn': 0, 'dm': 0, 'cad': 0, 'appet': 1, 'pc': 1
        }])
        
        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        df_healthy[numeric_cols] = self.scaler.transform(df_healthy[numeric_cols])
        
        prediction = self.model.predict(df_healthy)[0]
        self.assertEqual(prediction, 0, "Healthy patient should be classified as 0 (Not CKD)")

    def test_high_risk_patient_prediction(self):
        """Test prediction for a high-risk patient profile -> Expect class 1 (CKD)."""
        df_risk = pd.DataFrame([{
            'age': 65, 'bp': 160, 'sg': 1.010, 'al': 3.0, 'hemo': 8.5, 'sc': 4.5,
            'htn': 1, 'dm': 1, 'cad': 1, 'appet': 0, 'pc': 0
        }])
        
        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        df_risk[numeric_cols] = self.scaler.transform(df_risk[numeric_cols])
        
        prediction = self.model.predict(df_risk)[0]
        self.assertEqual(prediction, 1, "High risk patient should be classified as 1 (CKD)")

if __name__ == "__main__":
    unittest.main()
