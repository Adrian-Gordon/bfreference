import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest

import mock

from preprocess_bf_data import GenerateData


def test_constructor():
  generate = GenerateData('./tests/testdata/testdata.csv', './tests/testdata/scaler.save',  30, 30, './tests/testdata/preprocessed_testdata.csv','./tests/testdata/raw_testdata.csv')
  assert generate.data.shape[0] == 61
  assert generate.data.shape[1] == 41
  assert generate.data['layprice1'][60] == 1000
  assert generate.data['laydepth1'][60] == 1
  assert generate.nsequences == 1


def test_preprocess_includes_valid():
  generate = GenerateData('./tests/testdata/testdata.csv', './tests/testdata/scaler.save',  30, 30, './tests/testdata/preprocessed_testdata.csv','./tests/testdata/raw_testdata.csv')
  generate.preprocess(15.0, 20.0, 30, 30)
  assert generate.included_data.shape[0] == 30

def test_preprocess_excludes_invalid():
  generate = GenerateData('./tests/testdata/testdata.csv', './tests/testdata/scaler.save',  30, 30, './tests/testdata/preprocessed_testdata.csv','./tests/testdata/raw_testdata.csv')
  generate.preprocess(4.0, 5.0, 30, 30)
  assert generate.included_data.shape[0] == 0