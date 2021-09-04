#!/usr/bin/env python
# Copyright 2021 Reda Vince

import os
import sys
import unittest
import rospkg
import rosunit

__author__ = "Reda Vince"
__copyright__ = "Copyright 2021 Reda Vince"


class TestWrapper(unittest.TestCase):
    pass

if __name__ == '__main__':
    rosunit.unitrun(pkg, 'test_masac', TestWrapper)
