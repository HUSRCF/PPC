import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/preparations/m18_union"))
from build_exact_sequence_union import label_sequence, merge_labels, validate_partition


class UnionTests(unittest.TestCase):
    def test_complementary_contexts(self):
        target, pos, obs = merge_labels([np.array([1, 1, 0]), np.array([0, 1, 1])])
        np.testing.assert_array_equal(target, [1, 1, 1])
        np.testing.assert_array_equal(pos, [1, 2, 1])
        np.testing.assert_array_equal(obs, [2, 2, 2])

    def test_missing_is_not_negative(self):
        target, pos, obs = merge_labels([np.array([-100, 0, -100]), np.array([-100, -100, 1])])
        np.testing.assert_array_equal(target, [-100, 0, 1])
        np.testing.assert_array_equal(obs, [0, 1, 1])

    def test_duplicate_does_not_expand_union(self):
        y = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(merge_labels([y, y, y])[0], y)

    def test_residue_metadata_conversion(self):
        self.assertEqual(label_sequence(['THR', 'THR', 'LYS', 'ARG', 'X']), 'TTKRX')

    def test_incompatible_positions_fail(self):
        with self.assertRaises(ValueError):
            merge_labels([np.array([0, 1]), np.array([1])])

    def test_component_cannot_cross_split(self):
        rows = [{'seq_sha1': 'a', 'component_id': 'same_component', 'pdb_id': 'one', 'split': 'train'},
                {'seq_sha1': 'b', 'component_id': 'same_component', 'pdb_id': 'two', 'split': 'test'}]
        with self.assertRaisesRegex(ValueError, 'component_id crosses'):
            validate_partition(rows)

    def test_exact_sequence_cannot_cross_split(self):
        rows = [{'seq_sha1': 'a', 'component_id': 'one', 'pdb_id': 'one', 'split': 'train'},
                {'seq_sha1': 'a', 'component_id': 'two', 'pdb_id': 'two', 'split': 'test'}]
        with self.assertRaisesRegex(ValueError, 'seq_sha1 crosses'):
            validate_partition(rows)


if __name__ == '__main__':
    unittest.main()
