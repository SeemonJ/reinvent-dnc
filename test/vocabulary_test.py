import unittest
import test.fixtures.vocabulary as tfv

import numpy as np
import numpy.testing as npt

import models.vocabulary as mv


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.voc = tfv.simple()

    def test_add(self):
        idx = self.voc.add("#")
        self._check("#", idx)

    def test_update(self):
        idxs = self.voc.update(["5", "#"])
        self._check("5", idxs[0])
        self._check("#", idxs[1])

    def test_del(self):
        idx = self.voc["1"]
        del self.voc["1"]
        self.assertFalse("1" in self.voc)
        self.assertFalse(idx in self.voc)

    def test_equal(self):
        self.assertEqual(self.voc, tfv.simple())
        self.voc.add("#")
        self.assertNotEqual(self.voc, tfv.simple())

    def test_includes(self):
        self.assertTrue(2 in self.voc)
        self.assertTrue("1" in self.voc)
        self.assertFalse(21 in self.voc)
        self.assertFalse("6" in self.voc)

    def test_len(self):
        self.assertEqual(len(self.voc), 14)
        self.assertEqual(len(mv.Vocabulary()), 0)

    def test_encode(self):
        npt.assert_almost_equal(
            self.voc.encode(["^", "C", "C", "$"]),
            np.array([1, 9, 9, 2])
        )

    def test_decode(self):
        self.assertEqual(
            self.voc.decode(np.array([1, 9, 10, 9, 2])),
            ["^", "C", "N", "C", "$"]
        )

    # helpers
    def _check(self, token, idx):
        self.assertEqual(self.voc[token], idx)
        self.assertEqual(self.voc[idx], token)
        self.assertTrue(token in self.voc)
        self.assertTrue(idx in self.voc)


class TestSMILESTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = mv.SMILESTokenizer()

    def test_tokenize(self):
        self.assertListEqual(
            self.tokenizer.tokenize("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
            ["^", "C", "C", "(", "C", ")", "C", "c", "1", "c", "c", "c", "(", "c", "c", "1", ")",
             "[C@@H]", "(", "C", ")", "C", "(", "=", "O", ")", "O", "$"]
        )

        self.assertListEqual(
            self.tokenizer.tokenize("C%12CC(Br)C1CC%121[ClH]", with_begin_and_end=False),
            ["C", "%12", "C", "C", "(", "Br", ")", "C", "1", "C", "C", "%12", "1", "[ClH]"]
        )

    def test_untokenize(self):
        self.assertEqual(
            self.tokenizer.untokenize(["^", "C", "C", "(", "C", ")", "C", "c", "1", "c", "c", "c", "(", "c", "c", "1", ")",
                                       "[C@@H]", "(", "C", ")", "C", "(", "=", "O", ")", "O", "$"]),
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
        )

        self.assertEqual(
            self.tokenizer.untokenize(["C", "1", "C", "C", "(", "Br", ")", "C", "C", "C", "1", "[ClH]"]),
            "C1CC(Br)CCC1[ClH]"
        )


class TestCreateVocabulary(unittest.TestCase):

    def test_create(self):
        voc = mv.create_vocabulary(smiles_list=tfv.SMILES_LIST, tokenizer=mv.SMILESTokenizer())
        self.assertEqual(voc, tfv.simple())
