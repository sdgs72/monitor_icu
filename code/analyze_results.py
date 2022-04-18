from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys

#from sklearn.externals import joblib

import joblib

if len(sys.argv) != 2:
  print("Usage: python %s <file_path>" % sys.argv[0])
  sys.exit(1)

data = joblib.load(sys.argv[1])

for k, v in data.items():
  #print("hadm_id: %s" % k)
  print(f"hadm_id: {k}")
  print(v)
  break
