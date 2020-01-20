# python3
import unittest

import diff


class DiffTest(unittest.TestCase):

  def testBasic(self):
    diff.main(['diff.py', 'removes.txt', 'adds.txt'])
