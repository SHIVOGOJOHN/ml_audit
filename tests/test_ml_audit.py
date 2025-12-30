import unittest
import pandas as pd
import numpy as np
import os
import shutil
import json
from ml_audit import AuditTrialRecorder

class TestMLAudit(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.data = {
            'A': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'C': ['cat', 'dog', 'cat', 'bird', 'dog', 'cat', 'bird', 'dog', 'cat', 'cat'],
            'D': pd.date_range(start='1/1/2022', periods=10),
            'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
        }
        self.df = pd.DataFrame(self.data)
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_initialization(self):
        auditor = AuditTrialRecorder(self.df)
        self.assertEqual(auditor.current_df.shape, (10, 5))
        self.assertEqual(len(auditor.operations), 1) # LoadData op
        self.assertIn("load_data", auditor.operations[0].name)

    def test_impute_single(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.impute("A", strategy="mean")
        self.assertFalse(auditor.current_df['A'].isnull().any())
        self.assertAlmostEqual(auditor.current_df['A'].iloc[2], 5.777, places=1) # Mean of rest

    def test_impute_multi(self):
        df_nan = self.df.copy()
        df_nan.loc[0, 'B'] = np.nan
        auditor = AuditTrialRecorder(df_nan)
        
        auditor.impute(["A", "B"], strategy="constant", fill_value=-1)
        self.assertEqual(auditor.current_df['A'].iloc[2], -1)
        self.assertEqual(auditor.current_df['B'].iloc[0], -1)

    def test_scale(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.scale("B", method="minmax")
        self.assertEqual(auditor.current_df['B'].min(), 0.0)
        self.assertEqual(auditor.current_df['B'].max(), 1.0)

    def test_encode_onehot(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.encode("C", method="onehot")
        self.assertIn("C_cat", auditor.current_df.columns)
        self.assertIn("C_dog", auditor.current_df.columns)
        self.assertNotIn("C", auditor.current_df.columns) # get_dummies drops original

    def test_encode_label(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.encode("C", method="label")
        self.assertTrue(pd.api.types.is_numeric_dtype(auditor.current_df['C']))

    def test_transform_log(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.transform("B", func="log")
        self.assertAlmostEqual(auditor.current_df['B'].iloc[0], np.log1p(10))

    def test_discretize_bins(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.bin_numeric("B", bins=3, strategy="uniform", labels=["L", "M", "H"])
        self.assertTrue(pd.api.types.is_categorical_dtype(auditor.current_df['B']) or pd.api.types.is_object_dtype(auditor.current_df['B']))
        self.assertIn("L", auditor.current_df['B'].values)

    def test_date_extract(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.extract_date_features("D", features=["year", "month"])
        self.assertIn("D_year", auditor.current_df.columns)
        self.assertIn("D_month", auditor.current_df.columns)

    def test_balance_oversample(self):
        # Target has 7 zeros and 3 ones
        auditor = AuditTrialRecorder(self.df)
        auditor.balance_classes("Target", strategy="oversample")
        counts = auditor.current_df['Target'].value_counts()
        self.assertEqual(counts[0], counts[1]) # Should be balanced
        self.assertEqual(counts[0], 6)

    def test_generic_pandas(self):
        auditor = AuditTrialRecorder(self.df)
        # track generic usage of rename
        auditor.track_pandas("rename", columns={'A': 'Alpha'})
        self.assertIn("Alpha", auditor.current_df.columns)
        self.assertNotIn("A", auditor.current_df.columns)

    def test_reproducibility(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.impute("A", strategy="mean").scale("B", method="standard")
        
        is_reproducible = auditor.verify_reproducibility()
        self.assertTrue(is_reproducible)

    def test_export_and_visualize(self):
        auditor = AuditTrialRecorder(self.df)
        auditor.impute("A", strategy="mean")
        
        # Test export with custom dir
        auditor.export_audit_trail("test_audit.json", output_dir=self.output_dir, visualize=True)
        
        json_path = os.path.join(self.output_dir, "test_audit.json")
        html_path = os.path.join(self.output_dir, "test_audit.html")
        
        self.assertTrue(os.path.exists(json_path), "JSON file not created")
        self.assertTrue(os.path.exists(html_path), "HTML file not created")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data['operations']), 2) # Load + Impute

if __name__ == '__main__':
    unittest.main()
