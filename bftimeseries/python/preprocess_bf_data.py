#A class to prreprocess bf timeseries data
# Usage python preprocess_bf_data <configfilename>
import sys
import json
import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib



class GenerateData:
  

  def __init__(self, datafile_path,scaler_path,seq_length, offset, processed_output_path, raw_output_path):
    '''
      arguments:
        datafile_path -- string, filepath for input datafile
        scaler_path -- struing, path to save scaler to
        seq_length -- integer, no of entries per timeseries
        offset -- integer, offset from start of data series to use in output. 
                  e.g. seq_length= 30, offset = 30 : skip first 30,then use the next 30
        processed_output_path -- string, filepath for processed output data
        raw_output_path -- string, filepath for raw output data (for the sequences that are to be used)

    '''
    self.data =[]
    self.scaler_path = scaler_path
    self.processed_output_path = processed_output_path
    self.raw_output_path = raw_output_path
    self.included_data = pd.DataFrame() #data to be included - i.e. those with odds in the required range, and from tghe required offset
    self.processed_data =pd.DataFrame() #the processed data
    self.data = pd.read_csv(datafile_path)
    print("Loaded")
    self.data['layprice1'] = self.data['layprice1'].replace(0,1000)
    self.data['layprice2'] = self.data['layprice2'].replace(0,1000)
    self.data['layprice3'] = self.data['layprice3'].replace(0,1000)
    self.data['layprice4'] = self.data['layprice4'].replace(0,1000)
    self.data['layprice5'] = self.data['layprice5'].replace(0,1000)
    self.data['layprice6'] = self.data['layprice6'].replace(0,1000)
    self.data['layprice7'] = self.data['layprice7'].replace(0,1000)
    self.data['layprice8'] = self.data['layprice8'].replace(0,1000)
    self.data['layprice9'] = self.data['layprice9'].replace(0,1000)
    self.data['layprice10'] = self.data['layprice10'].replace(0,1000)

    self.data['laydepth1'] = self.data['laydepth1'].replace(0,1)
    self.data['laydepth2'] = self.data['laydepth2'].replace(0,1)
    self.data['laydepth3'] = self.data['laydepth3'].replace(0,1)
    self.data['laydepth4'] = self.data['laydepth4'].replace(0,1)
    self.data['laydepth5'] = self.data['laydepth5'].replace(0,1)
    self.data['laydepth6'] = self.data['laydepth6'].replace(0,1)
    self.data['laydepth7'] = self.data['laydepth7'].replace(0,1)
    self.data['laydepth8'] = self.data['laydepth8'].replace(0,1)
    self.data['laydepth9'] = self.data['laydepth9'].replace(0,1)
    self.data['laydepth10'] = self.data['laydepth10'].replace(0,1)

    self.data['backprice1'] = self.data['backprice1'].replace(0,1000)
    self.data['backprice2'] = self.data['backprice2'].replace(0,1000)
    self.data['backprice3'] = self.data['backprice3'].replace(0,1000)
    self.data['backprice4'] = self.data['backprice4'].replace(0,1000)
    self.data['backprice5'] = self.data['backprice5'].replace(0,1000)
    self.data['backprice6'] = self.data['backprice6'].replace(0,1000)
    self.data['backprice7'] = self.data['backprice7'].replace(0,1000)
    self.data['backprice8'] = self.data['backprice8'].replace(0,1000)
    self.data['backprice9'] = self.data['backprice9'].replace(0,1000)
    self.data['backprice10'] = self.data['backprice10'].replace(0,1000)

    self.data['backdepth1'] = self.data['backdepth1'].replace(0,1)
    self.data['backdepth2'] = self.data['backdepth2'].replace(0,1)
    self.data['backdepth3'] = self.data['backdepth3'].replace(0,1)
    self.data['backdepth4'] = self.data['backdepth4'].replace(0,1)
    self.data['backdepth5'] = self.data['backdepth5'].replace(0,1)
    self.data['backdepth6'] = self.data['backdepth6'].replace(0,1)
    self.data['backdepth7'] = self.data['backdepth7'].replace(0,1)
    self.data['backdepth8'] = self.data['backdepth8'].replace(0,1)
    self.data['backdepth9'] = self.data['backdepth9'].replace(0,1)
    self.data['backdepth10'] = self.data['backdepth10'].replace(0,1)

    self.nsequences = len(self.data) / ( seq_length + offset)
    #print("nsequences: " , len(self.data), self.nsequences)

  def preprocess(self, max_starting_lay_price, max_starting_back_price, seq_length, offset):
    print("Preprocessing")
    for i in range (self.nsequences):
      asequence  = self.data[i * seq_length + offset: (i * seq_length) + seq_length + offset].copy()
      add_it = False
      for index, row in asequence.iterrows():
        if row['layprice1'] <= max_starting_lay_price and row['backprice1'] <= max_starting_back_price:
          add_it = True
          
      if add_it:
          self.included_data = self.included_data.append(asequence,ignore_index = True)
      #print("add it: ", add_it)

    n_good_sequences = len(self.included_data) / (seq_length) #the number of sequences to preprocess

    print("Preprocessing ", n_good_sequences , "sequences")
    self.included_data.to_csv(self.raw_output_path, index = False)

    for j in range(n_good_sequences):
      #print("included data: ", self.included_data)
      #print("processing: ", j)
      sequence  = self.included_data[j * seq_length: (j + 1)* seq_length].copy().reset_index(drop = True)
      #print(sequence)
     
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

      self.processed_data = self.processed_data.append(sequence)

    if n_good_sequences > 0:
     
      print("Scaling")
      scaler = StandardScaler()

      self.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=scaler.fit_transform(self.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])
        #save the scaler for later use
     
      joblib.dump(scaler, self.scaler_path)
     
      self.processed_data.to_csv(self.processed_output_path, index = False)
      print("Done")
      
    
