
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os

import sys

import json


from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import pandas as pd
import numpy as np


from .seq2seqmodel import *


class BfInferer():

  def __init__(self, config_filename):

    with open(config_filename,'r') as f:
      config = json.load(f)
      self.config = config

   
      self.seq2seqInference  = Seq2Seq(config)
      init = tf.global_variables_initializer()
      self.scaler = joblib.load(config["scaler_location"])

      self.sess = tf.Session()
      self.sess.run(init)
      saver = tf.train.Saver
      saver().restore(self.sess, config["savefilename"])
      

  def doInference(self, data):
    columns = ["layprice1","laydepth1","layprice2","laydepth2","layprice3","laydepth3","layprice4","laydepth4","layprice5","laydepth5","layprice6","laydepth6","layprice7","laydepth7","layprice8","laydepth8","layprice9","laydepth9","layprice10","laydepth10","backprice1","backdepth1","backprice2","backdepth2","backprice3","backdepth3","backprice4","backdepth4","backprice5","backdepth5","backprice6","backdepth6","backprice7","backdepth7","backprice8","backdepth8","backprice9","backdepth9","backprice10","backdepth10"]
    
    sequence = pd.DataFrame(data, columns = columns)

    #prices relative to the first price

    starting_layprice1 = sequence['layprice1'][0]
    starting_layprice2 = sequence['layprice2'][0]
    starting_layprice3 = sequence['layprice3'][0]
    starting_layprice4 = sequence['layprice4'][0]
    starting_layprice5 = sequence['layprice5'][0]
    starting_layprice6 = sequence['layprice6'][0]
    starting_layprice7 = sequence['layprice7'][0]
    starting_layprice8 = sequence['layprice8'][0]
    starting_layprice9 = sequence['layprice9'][0]
    starting_layprice10 = sequence['layprice10'][0]

    starting_backprice1 = sequence['backprice1'][0]
    starting_backprice2 = sequence['backprice2'][0]
    starting_backprice3 = sequence['backprice3'][0]
    starting_backprice4 = sequence['backprice4'][0]
    starting_backprice5 = sequence['backprice5'][0]
    starting_backprice6 = sequence['backprice6'][0]
    starting_backprice7 = sequence['backprice7'][0]
    starting_backprice8 = sequence['backprice8'][0]
    starting_backprice9 = sequence['backprice9'][0]
    starting_backprice10 = sequence['backprice10'][0]

    #print(sequence)

    sequence['layprice1'] = sequence['layprice1'] - starting_layprice1
    sequence['layprice2'] = sequence['layprice2'] - starting_layprice2
    sequence['layprice3'] = sequence['layprice3'] - starting_layprice3
    sequence['layprice4'] = sequence['layprice4'] - starting_layprice4
    sequence['layprice5'] = sequence['layprice5'] - starting_layprice5
    sequence['layprice6'] = sequence['layprice6'] - starting_layprice6
    sequence['layprice7'] = sequence['layprice7'] - starting_layprice7
    sequence['layprice8'] = sequence['layprice8'] - starting_layprice8
    sequence['layprice9'] = sequence['layprice9'] - starting_layprice9
    sequence['layprice10'] = sequence['layprice10'] - starting_layprice10

    sequence['backprice1'] = sequence['backprice1'] - starting_backprice1
    sequence['backprice2'] = sequence['backprice2'] - starting_backprice2
    sequence['backprice3'] = sequence['backprice3'] - starting_backprice3
    sequence['backprice4'] = sequence['backprice4'] - starting_backprice4
    sequence['backprice5'] = sequence['backprice5'] - starting_backprice5
    sequence['backprice6'] = sequence['backprice6'] - starting_backprice6
    sequence['backprice7'] = sequence['backprice7'] - starting_backprice7
    sequence['backprice8'] = sequence['backprice8'] - starting_backprice8
    sequence['backprice9'] = sequence['backprice9'] - starting_backprice9
    sequence['backprice10'] = sequence['backprice10'] - starting_backprice10

    #print(sequence)

    #standardise

    sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=self.scaler.transform(sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])


    #print(sequence)


    input_data = sequence[['layprice1','laydepth1','backprice1','backdepth1']]

    input = np.array(input_data)

    #print(input)

    feed_dict={self.seq2seqInference.encoder_inputs[t]: input[t].reshape(1,self.config["input_dim"]) for t in range(self.config["input_sequence_length"])}
    feed_dict.update({self.seq2seqInference.decoder_target_inputs[t]: np.zeros([1,self.config["output_dim"]]) for t in range(self.config["output_sequence_length"])})
    #print(feed_dict)
    test_out = np.array(self.sess.run(self.seq2seqInference.encoder_decoder_inference,feed_dict)).transpose()#.reshape(-1)[:20]
    test_lay_output = test_out[0].reshape(-1)
    #test_output = np.array(test_out).reshape(-1)
    #print(test_lay_output)
    test_back_output = test_out[1].reshape(-1)


    #de-standardize
    predicted_ar = np.zeros((5, 40))
    for row2 in range(5):
      predicted_ar[row2][0]= test_lay_output[row2] 
      predicted_ar[row2][20] =test_back_output[row2] 

    descaled_predicted_ar = self.scaler.inverse_transform(predicted_ar)
    descaled_predicted_ar[:,0] += starting_layprice1
    descaled_predicted_ar[:,20] += starting_backprice1

    return descaled_predicted_ar[:,0], descaled_predicted_ar[:,20]



'''

raw_data1 = [[5.5,19.29,5.6,29.05,5.7,27.56,5.8,7.59,5.9,78.18,6,38.22,6.2,213.12,6.4,12.74,6.6,33.92,6.8,14.96,5.4,30.42,5.3,47.02,5.2,127.43,5.1,48.23,5,18.76,4.9,20.62,4.8,14.95,4.7,43.36,4.6,148.91,4.5,52.63],
[5.5,74.19,5.6,41.41,5.7,31.56,5.8,7.59,5.9,81.18,6,38.22,6.2,173.12,6.4,51.74,6.6,33.92,6.8,14.96,5.4,46.53,5.3,113.64,5.2,118.85,5.1,95.34,5,18.76,4.9,20.62,4.8,34.69,4.7,96.36,4.6,44.07,4.5,76.06],
[5.5,63.91,5.6,53.09,5.7,31.56,5.8,22.92,5.9,86.18,6,43.22,6.2,173.12,6.4,56.37,6.6,33.92,6.8,16.96,5.4,82.82,5.3,183.87,5.2,118.85,5.1,154.45,5,41.32,4.9,20.62,4.8,41.69,4.7,96.36,4.6,44.07,4.5,76.06],
[5.7,27.99,5.8,51.15,5.9,89.18,6,53.61,6.2,173.12,6.4,17.75,6.6,71.92,6.8,16.96,7,3.2,7.4,2,5.5,57.88,5.4,189.85,5.3,221.12,5.2,159.37,5.1,130.14,5,41.32,4.9,20.62,4.8,93.69,4.7,20.31,4.6,44.07],
[5.6,66.72,5.7,13.86,5.8,18.41,5.9,81.56,6,48.95,6.2,180.52,6.4,10.38,6.6,72.06,6.8,12,7,3.2,5.5,33.97,5.4,133.75,5.3,172.82,5.2,146.12,5.1,145.85,5,22.82,4.9,34.83,4.8,29.02,4.7,73.02,4.6,44.07],
[5.6,87.8,5.7,45.35,5.8,38.15,5.9,81.56,6,78.95,6.2,180.52,6.4,17.52,6.6,72.06,6.8,12,7,15.29,5.5,77.82,5.4,178.05,5.3,239.31,5.2,143.12,5.1,159.28,5,22.82,4.9,34.83,4.8,81.02,4.7,20.02,4.6,44.07],
[5.6,68.03,5.7,35.98,5.8,36.15,5.9,81.56,6,78.9,6.2,180.52,6.4,19.52,6.6,72.06,6.8,12,7,15.29,5.5,181.67,5.4,176.91,5.3,184.95,5.2,134.31,5.1,159.28,5,22.82,4.9,34.83,4.8,81.02,4.7,20.02,4.6,44.07],
[5.7,20.33,5.8,47.13,5.9,84.56,6,79.39,6.2,187.96,6.4,19.52,6.6,72.06,6.8,14,7,15.29,7.2,2,5.6,164.7,5.5,201.92,5.4,176.91,5.3,241.31,5.2,132.31,5.1,159.28,5,22.82,4.9,34.83,4.8,81.02,4.7,20.02],
[5.8,114.17,5.9,87.07,6,105.39,6.2,187.96,6.4,19.52,6.6,34.06,6.8,51,7,15.29,7.2,2,7.4,2,5.7,247.19,5.6,247.65,5.5,177.69,5.4,232.3,5.3,230.31,5.2,76.31,5.1,141.9,5,22.82,4.9,85.83,4.8,29.02],
[5.8,38.4,5.9,76.22,6,121.56,6.2,244.25,6.4,24.52,6.6,34.06,6.8,21.13,7,15.29,7.2,37,7.4,2,5.7,231.55,5.6,220.65,5.5,171.38,5.4,159.41,5.3,167.55,5.2,76.31,5.1,220.9,5,22.82,4.9,34.83,4.8,29.02]]


bfInferer = BfInferer("./configs/config_bf.json")

lay_predictions, back_predictions = bfInferer.doInference(raw_data1)

print(lay_predictions)
print(back_predictions)

raw_data2 = [[10.5,5,11,4,12,3.45,12.5,24.82,13,15.99,13.5,7.65,16.5,2.04,18,20.89,19,2,24,2,9.4,8.2,9.2,15.74,9,33.01,8.8,35.37,8.4,2,8,22.51,7.8,35.33,7.6,27.73,7.4,4.3,7.2,3.56],
[10.5,6.66,10.5,3,11,14.5,11.5,3,12,5.24,13,16.31,13.5,26.65,16.5,2.04,18,20.89,19,2,9.8,9.78,9.6,3,9.4,17.14,9.2,19.8,9,31.01,8.8,35.37,8.4,2,8.2,30,8,22.51,7.8,3.33],
[10.5,19.89,11,9.69,11.5,24.55,12,4.43,12.5,4.91,13,15.99,13.5,3.16,14,18,16.5,2.04,18,20.89,10,3,9.8,32.76,9.6,3,9.4,15.14,9.2,13.74,9,31.01,8.8,30.48,8.6,29,8.4,2,8,22.51],
[10.5,34.76,11,9.69,11.5,24.55,12.5,5.34,13,15.99,13.5,3.16,14,18,16.5,2.04,18,20.89,19,2,9.6,3,9.4,9.27,9.2,13.74,9,31.01,8.8,35.32,8.4,32,8,22.51,7.8,3.33,7.6,27.73,7.4,4.3],
[10,4,10.5,37.66,11,12.95,11.5,24.55,12.5,5.34,13,15.99,13.5,3.16,14,18,16.5,2.04,18,20.89,9.4,3.89,9.2,17.8,9,31.01,8.8,35.32,8.4,2,8.2,6,8,53.51,7.8,3.33,7.6,27.73,7.4,4.3],
[10.5,34.76,11,9.69,11.5,24.55,12.5,5.34,13,15.99,13.5,3.16,14,18,16.5,2.04,18,20.89,19,2,10,25.78,9.6,33.22,9.4,14.99,9.2,13.74,9,35.68,8.8,30.48,8.6,29,8.4,2,8,22.51,7.8,3.33],
[10,14.92,10.5,50.63,11,9.69,11.5,21.55,12.5,5.34,13,15.99,13.5,22.16,16.5,2.04,18,20.89,19,2,9.6,16.6,9.4,11.99,9.2,16.74,9,29.01,8.8,30.48,8.4,2,8.2,30,8,22.51,7.8,3.33,7.6,27.73],
[9.6,6.18,9.8,11.94,10,16.77,10.5,50.63,11.5,21.55,12.5,5.34,13,34.99,13.5,3.16,16.5,2.04,18,20.89,9.4,13.32,9.2,5.28,9,32.01,8.8,39.47,8.4,2,8,22.51,7.8,35.33,7.6,27.73,7.4,4.3,7.2,3.56],
[9.4,7.25,9.6,10.18,9.8,18.42,10,16.77,10.5,29.76,11.5,21.55,12,21.43,13,16.31,13.5,3.16,16.5,2.04,9,30.99,8.8,51.13,8.4,8.92,8,22.51,7.8,35.33,7.6,27.73,7.4,4.3,7.2,3.56,7,12.29,6.8,3.38],
[9.2,2.66,9.4,11.95,9.6,17.58,9.8,18.42,10,12.77,10.5,29.76,12,21.43,13,16.31,13.5,3.16,16.5,2.04,8.8,44.47,8.6,4,8.4,17.84,8,15.85,7.8,3.33,7.6,60.73,7.4,4.3,7.2,10.22,7,12.29,6.8,3.38]]


lay_predictions, back_predictions = bfInferer.doInference(raw_data2)

print(lay_predictions)
print(back_predictions)
  
'''