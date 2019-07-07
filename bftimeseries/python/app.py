#
#	Usage: python app.py <configfilename>
#	<configfilename> should be a json structure containing:
#	{

#	"input_datafile_path": <string>, -- path to input csv file
#	"processed_output_datafile_path": <string>, -- path to output csv datafile for processed data
#	"raw_output_datafile_path": <string>, -- path to output csv datafile for raw data (unprocessed prices for data included in the dataset)
#	"scaler_file_path":<string>, -- path to the output file for saving the scaler
#	"sequence_length": <integer>, -- length of the output sequence to generate
#	"offset": <integer>, -- offset within the input data to process for output 
#	"max_starting_lay_price": <integer>, -- the maximum starting lay price for the timeseries to be considered
#	"max_starting_back_price": <integer>, -- the maximum starting back price for the timeseries to be considered

#	sequence_length: 30, offset: 30 would output a sequence of 30, starting from the 30th record of an input timeseries (assumed to be 60 in length)
#}
#

import os

import sys

import json

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + './')

from preprocess_bf_data import GenerateData



configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)

datagenerator = GenerateData(config['input_datafile_path'], config['scaler_file_path'], config['sequence_length'],config['offset'],config['processed_output_datafile_path'],config['raw_output_datafile_path'])

datagenerator.preprocess(config['max_starting_lay_price'], config['max_starting_back_price'], config['sequence_length'], config['offset'])

