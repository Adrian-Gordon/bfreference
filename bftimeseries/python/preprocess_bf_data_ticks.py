#A class to preprocess bf timeseries data
# Usage python preprocess_bf_data_ticks <configfilename>
#Generate a dataset with each timeseries having prices expressed in difference in tickes relative to the start of the sequence
import sys
import json
import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from trader import Trader



class GenerateData:
  

  def __init__(self, datafile_path,scaler_path,seq_length, offset, window, processed_output_path, raw_output_path):
    '''
      arguments:
        datafile_path -- string, filepath for input datafile
        scaler_path -- struing, path to save scaler to
        seq_length -- integer, no of entries per timeseries
        offset -- integer, offset from start of data series to use in output. 
                  e.g. seq_length= 30, offset = 30 : skip first 30,then use the next 30
        window -- width of the window to output. The data will be output in sequences of this length
                  will iterate over a sequence, generating output sequences of this length
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

    self.nsequences = int(len(self.data) / ( seq_length + offset))
    #print("nsequences: " , len(self.data), self.nsequences)

  def preprocess(self, max_starting_lay_price, max_starting_back_price, seq_length, offset, window):
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

    n_good_sequences = int(len(self.included_data) / (seq_length)) #the number of sequences to preprocess

    print("Preprocessing ", n_good_sequences , "sequences")
    self.included_data.to_csv(self.raw_output_path, index = False) #output the raw data, fo use in plotting/predicting etc

    #n_good_sequences = 1 #for testing

    trader = Trader()

    for j in range(n_good_sequences):
      #print("included data: ", self.included_data)
      print("processing: ", j)
      sequence  = self.included_data[j * seq_length: (j + 1)* seq_length].copy().reset_index(drop = True)
      #print(sequence)

      #generate list of sequences

      n_window_iterations = 60 - offset - window
      #print("iterations:", n_window_iterations)
      for k in range(n_window_iterations):
        raw_output_sequence = sequence[k:k + window]
        #print("iteration:", k)
        #print(raw_output_sequence['layprice1'].iloc[0])

     
     
        starting_layprice1_ticks = trader.get_ticks(raw_output_sequence['layprice1'].iloc[0])
        starting_layprice2_ticks =  trader.get_ticks(raw_output_sequence['layprice2'].iloc[0])
        starting_layprice3_ticks =  trader.get_ticks(raw_output_sequence['layprice3'].iloc[0])
        starting_layprice4_ticks =  trader.get_ticks(raw_output_sequence['layprice4'].iloc[0])
        starting_layprice5_ticks =  trader.get_ticks(raw_output_sequence['layprice5'].iloc[0])
        starting_layprice6_ticks =  trader.get_ticks(raw_output_sequence['layprice6'].iloc[0])
        starting_layprice7_ticks =  trader.get_ticks(raw_output_sequence['layprice7'].iloc[0])
        starting_layprice8_ticks =  trader.get_ticks(raw_output_sequence['layprice8'].iloc[0])
        starting_layprice9_ticks =  trader.get_ticks(raw_output_sequence['layprice9'].iloc[0])
        starting_layprice10_ticks =  trader.get_ticks(raw_output_sequence['layprice10'].iloc[0])

        starting_backprice1_ticks =  trader.get_ticks(raw_output_sequence['backprice1'].iloc[0])
        starting_backprice2_ticks =  trader.get_ticks(raw_output_sequence['backprice2'].iloc[0])
        starting_backprice3_ticks =  trader.get_ticks(raw_output_sequence['backprice3'].iloc[0])
        starting_backprice4_ticks =  trader.get_ticks(raw_output_sequence['backprice4'].iloc[0])
        starting_backprice5_ticks =  trader.get_ticks(raw_output_sequence['backprice5'].iloc[0])
        starting_backprice6_ticks =  trader.get_ticks(raw_output_sequence['backprice6'].iloc[0])
        starting_backprice7_ticks =  trader.get_ticks(raw_output_sequence['backprice7'].iloc[0])
        starting_backprice8_ticks =  trader.get_ticks(raw_output_sequence['backprice8'].iloc[0])
        starting_backprice9_ticks =  trader.get_ticks(raw_output_sequence['backprice9'].iloc[0])
        starting_backprice10_ticks =  trader.get_ticks(raw_output_sequence['backprice10'].iloc[0])


        #print("starting_layprice1_ticks" , starting_layprice1_ticks)

        tickified_output_sequence = raw_output_sequence.copy()

        tickified_output_sequence['layprice1'] = tickified_output_sequence['layprice1'].apply(lambda x: trader.get_ticks(x) - starting_layprice1_ticks)
        tickified_output_sequence['layprice2'] = tickified_output_sequence['layprice2'].apply(lambda x: trader.get_ticks(x) - starting_layprice2_ticks)
        tickified_output_sequence['layprice3'] = tickified_output_sequence['layprice3'].apply(lambda x: trader.get_ticks(x) - starting_layprice3_ticks)
        tickified_output_sequence['layprice4'] = tickified_output_sequence['layprice4'].apply(lambda x: trader.get_ticks(x) - starting_layprice4_ticks)
        tickified_output_sequence['layprice5'] = tickified_output_sequence['layprice5'].apply(lambda x: trader.get_ticks(x) - starting_layprice5_ticks)
        tickified_output_sequence['layprice6'] = tickified_output_sequence['layprice6'].apply(lambda x: trader.get_ticks(x) - starting_layprice6_ticks)
        tickified_output_sequence['layprice7'] = tickified_output_sequence['layprice7'].apply(lambda x: trader.get_ticks(x) - starting_layprice7_ticks)
        tickified_output_sequence['layprice8'] = tickified_output_sequence['layprice8'].apply(lambda x: trader.get_ticks(x) - starting_layprice8_ticks)
        tickified_output_sequence['layprice9'] = tickified_output_sequence['layprice9'].apply(lambda x: trader.get_ticks(x) - starting_layprice9_ticks)
        tickified_output_sequence['layprice10'] = tickified_output_sequence['layprice10'].apply(lambda x: trader.get_ticks(x) - starting_layprice10_ticks)

        tickified_output_sequence['backprice1'] = tickified_output_sequence['backprice1'].apply(lambda x: trader.get_ticks(x) - starting_backprice1_ticks)
        tickified_output_sequence['backprice2'] = tickified_output_sequence['backprice2'].apply(lambda x: trader.get_ticks(x) - starting_backprice2_ticks)
        tickified_output_sequence['backprice3'] = tickified_output_sequence['backprice3'].apply(lambda x: trader.get_ticks(x) - starting_backprice3_ticks)
        tickified_output_sequence['backprice4'] = tickified_output_sequence['backprice4'].apply(lambda x: trader.get_ticks(x) - starting_backprice4_ticks)
        tickified_output_sequence['backprice5'] = tickified_output_sequence['backprice5'].apply(lambda x: trader.get_ticks(x) - starting_backprice5_ticks)
        tickified_output_sequence['backprice6'] = tickified_output_sequence['backprice6'].apply(lambda x: trader.get_ticks(x) - starting_backprice6_ticks)
        tickified_output_sequence['backprice7'] = tickified_output_sequence['backprice7'].apply(lambda x: trader.get_ticks(x) - starting_backprice7_ticks)
        tickified_output_sequence['backprice8'] = tickified_output_sequence['backprice8'].apply(lambda x: trader.get_ticks(x) - starting_backprice8_ticks)
        tickified_output_sequence['backprice9'] = tickified_output_sequence['backprice9'].apply(lambda x: trader.get_ticks(x) - starting_backprice9_ticks)
        tickified_output_sequence['backprice10'] = tickified_output_sequence['backprice10'].apply(lambda x: trader.get_ticks(x) - starting_backprice10_ticks)

       # print('raw:', raw_output_sequence['layprice1'])
      #  print("tickified: ", tickified_output_sequence['layprice1'])
        self.processed_data = self.processed_data.append(tickified_output_sequence)
  
     
    print("Scaling")
    scaler = StandardScaler()

    self.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=scaler.fit_transform(self.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])
      #save the scaler for later use
   
    joblib.dump(scaler, self.scaler_path)
   
    self.processed_data.to_csv(self.processed_output_path, index = False)
    print("Done")
      
