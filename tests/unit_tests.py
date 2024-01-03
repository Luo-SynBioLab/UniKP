import os
import unittest
import pandas as pd
from UniKP.utils import split,split_refactored

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir=os.path.join(dir_path,'..','datasets')

class TestSplitFunction(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_excel(os.path.join(dataset_dir,'Generated_pH_unified_smiles_636.xlsx'))

    def test_split_comparison(self):
        for index, row in self.data.iterrows():
            # if 'Na' not in row['smiles']:
            #     continue
            original_result = split(row['smiles'])
            refactored_result = split_refactored(row['smiles'])
            # print('-='*60)
            # print(f'| {row["smiles"]}')
            # print(f'- {original_result}\n+ {refactored_result}')
            # print('-='*60)
            self.assertEqual(original_result, refactored_result, f'Failed at row {index + 2}: {row["smiles"]}')


if __name__ == '__main__':
    unittest.main()
