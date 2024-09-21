import unittest
from HW01.ex1.ex1 import cal_p_and_r, task1_3, task4
import io
import sys
import io
import sys
import io
import sys


class TestEx1(unittest.TestCase):

    def test_cal_p_and_r(self):
        y_true = [0, 1, 2, 2, 1, 0]
        y_pred = [0, 2, 1, 2, 0, 1]
        expected_micro_precision = 0.3333333333333333
        expected_macro_precision = 0.3333333333333333
        expected_micro_recall = 0.3333333333333333
        expected_macro_recall = 0.3333333333333333
        expected_micro_f1 = 0.3333333333333333
        expected_macro_f1 = 0.3333333333333333

        # Capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        cal_p_and_r(y_true, y_pred)
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue().strip().split("\n")
        self.assertIn(f"Micro Precision: {expected_micro_precision}", output)
        self.assertIn(f"Macro Precision: {expected_macro_precision}", output)
        self.assertIn(f"Micro Recall: {expected_micro_recall}", output)
        self.assertIn(f"Macro Recall: {expected_macro_recall}", output)
        self.assertIn(f"Micro F1: {expected_micro_f1}", output)
        self.assertIn(f"Macro F1: {expected_macro_f1}", output)

    def test_task1_3(self):
        # Capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        task1_3()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue().strip().split("\n")
        self.assertIn("Micro Precision: 0.5833333333333334", output)
        self.assertIn("Macro Precision: 0.601010101010101", output)
        self.assertIn("Micro Recall: 0.5833333333333334", output)
        self.assertIn("Macro Recall: 0.5833333333333334", output)
        self.assertIn("Micro F1: 0.5833333333333334", output)
        self.assertIn("Macro F1: 0.592039800995025", output)

    def test_task4(self):
        # Capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        task4()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue().strip().split("\n")
        self.assertIn("Micro Precision: 0.75", output)
        self.assertIn("Macro Precision: 0.7777777777777777", output)
        self.assertIn("Micro Recall: 0.6666666666666666", output)
        self.assertIn("Macro Recall: 0.6666666666666666", output)
        self.assertIn("Micro F1: 0.7058823529411765", output)
        self.assertIn("Macro F1: 0.7179487179487178", output)


if __name__ == "__main__":
    unittest.main()
