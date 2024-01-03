import unittest

class TestSplitSMILES(unittest.TestCase):
    test_cases = [
            ("CClCBr", "C Cl C Br"),
            ("CClCCl", "C Cl C Cl"),
            ("NaBr", "Na Br"),
            ("OCl", "O Cl"),
            ("BrNCCl", "Br N C C Cl")
            # Add more test cases here if needed
        ]
    
    def test_original_split_function(self):
        from UniKP.utils import split_ori as original_split
        # Test cases for the original split function

        for sm, expected_result in self.test_cases:
            with self.subTest(sm=sm):
                result = original_split(sm)
                self.assertEqual(result, expected_result)
    
    def test_refactored_split_function(self):
        from UniKP.utils import split as refactored_split
        # Test cases for the refactored split function
        
        for sm, expected_result in self.test_cases:
            with self.subTest(sm=sm):
                result = refactored_split(sm)
                self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
