import unittest
from unittest.mock import patch
from zeroshot_engine import display_label_flowchart

class TestFlowchart(unittest.TestCase):
    def setUp(self):
        self.label_codes = {
            "present": 1,
            "absent": 0,
            "non-coded": 8,
            "empty-list": [],
        }

        self.valid_keys = [
            "issue_env",
            "issue_clim",
            "clim_weather",
            "clim_pos",
            "clim_neg",
            "clim_pol",
            "gov_clim_response",
            "clim_skep",
        ]

        self.stop_conditions = {
            1: {
                "condition": 0,
                "blocked_keys": [
                    "clim_weather",
                    "clim_pos",
                    "clim_neg",
                    "clim_pol",
                    "gov_clim_response",
                    "clim_skep"
                ],
            },
            2: {
                "condition": 0,
                "blocked_keys": ["gov_clim_response"],
            },
        }

    def test_display_label_flowchart_runs_without_errors(self):
        # This test simply verifies the function runs without raising exceptions
        try:
            display_label_flowchart(self.valid_keys, self.stop_conditions, self.label_codes, graphical=True)
            test_passed = True
        except Exception as e:
            test_passed = False
            self.fail(f"display_label_flowchart raised an exception: {e}")

        self.assertTrue(test_passed)

    @patch('zeroshot_engine.display_label_flowchart')
    def test_display_label_flowchart_called_with_correct_args(self, mock_display):
        # Test that the function gets called with the expected arguments
        display_label_flowchart(self.valid_keys, self.stop_conditions, self.label_codes, graphical=True)
        mock_display.assert_called_once_with(self.valid_keys, self.stop_conditions, self.label_codes)

if __name__ == '__main__':
    unittest.main()