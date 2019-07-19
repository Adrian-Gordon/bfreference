#basic seqtoseq inference model for time series
#usage: python basicmodelinference.py <configfilename>

import tensorflow as tf

import os

import sys

import json

from matplotlib import pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import pandas as pd
import numpy as np


from seq2seqmodel import *

configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)

generate_data_library = config["data_modulename"]

cmd = 'from ' + generate_data_library + ' import *'

exec(cmd)

 #now do some inference

#create a seq2seq instance
seq2seqInference  = Seq2Seq(config)
init = tf.global_variables_initializer()

gd=GenerateData(config["datafilename"])
gd_raw = GenerateData(config["rawdatafilename"])

n_rows = len(gd.processed_data) / 30


column_increment = 0

limit = 0.01

scaler = joblib.load(config["scaler_location"])

with tf.Session() as sess:
  sess.run(init)
  saver = tf.train.Saver
  saver().restore(sess, os.path.join('./',config["savefilename"]))

  for row in range(0,n_rows):
    print("NEXT ROW", row)
    for column in range(1):
      print("n_rows:", n_rows,"ROW:", row, "NEXT COLUMN: ", column)
      test_sequence_input_raw = gd_raw.getTestSample(30, 10, row, column)
      lay_price_start = test_sequence_input_raw[0][0]
      back_price_start = test_sequence_input_raw[0][2]
      #print("lay_price_start %s", lay_price_start)
     # print("back_price_start %s", back_price_start)

      test_sequence_input = gd.getTestSample(30, 10, row, column)
      test_sequence_input_1 = gd.getTestSample(30, 15, row, column)



      #print(test_sequence_input.transpose())

      feed_dict={seq2seqInference.encoder_inputs[t]: test_sequence_input[t].reshape(1,config["input_dim"]) for t in range(config["input_sequence_length"])}
      feed_dict.update({seq2seqInference.decoder_target_inputs[t]: np.zeros([1,config["output_dim"]]) for t in range(config["output_sequence_length"])})
      #print(feed_dict)
      test_out = np.array(sess.run(seq2seqInference.encoder_decoder_inference,feed_dict)).transpose()#.reshape(-1)[:20]
      test_lay_output = test_out[0].reshape(-1)
      #test_output = np.array(test_out).reshape(-1)
      #print(test_lay_output)
      test_back_output = test_out[1].reshape(-1)
      #print(test_back_output)
      last_observed = test_sequence_input[9][0]
      next_predicted = test_lay_output[0]

      #print(last_observed, next_predicted, last_observed - next_predicted)

      #if abs(last_observed - next_predicted) < limit :

      #transposed = test_sequence_input.transpose()
      #print(transposed)
      
      observed_ar= np.zeros((15,40))
      for row1 in range(15):
        observed_ar[row1][0] = test_sequence_input_1[row1][0] 
        observed_ar[row1][20] = test_sequence_input_1[row1][2] 
      descaled_observed_ar = scaler.inverse_transform(observed_ar)

      descaled_observed_ar[:,0] += lay_price_start
      descaled_observed_ar[:,20] += back_price_start
      #print(descaled_observed_ar[:,0])
      #print(descaled_observed_ar[:,20])

      predicted_ar = np.zeros((5, 40))
      for row2 in range(5):
        predicted_ar[row2][0]= test_lay_output[row2] 
        predicted_ar[row2][20] =test_lay_output[row2] 

      descaled_predicted_ar = scaler.inverse_transform(predicted_ar)
      descaled_predicted_ar[:,0] += lay_price_start
      descaled_predicted_ar[:,20] += back_price_start

      print(descaled_observed_ar[:,0])
      print(descaled_observed_ar[:,20])

      print(descaled_predicted_ar[:,0])
      print(descaled_predicted_ar[:,20])

      plt.figure(figsize=(15,4))
      l1, = plt.plot(descaled_observed_ar[:,0], 'b.', label = 'Actual lay')
      l2, = plt.plot(descaled_observed_ar[:,20], 'y.', label = 'Actual back')
      l3, = plt.plot(range(10,15),descaled_predicted_ar[:,0], 'r.', label = 'Predicted lay')
      l4, = plt.plot(range(10,15),descaled_predicted_ar[:,20], 'g.', label = 'Predicted back')

      plt.legend(handles=[l1, l2, l3, l4], loc='upper left')

      plt.show()
      
      i = raw_input("")

