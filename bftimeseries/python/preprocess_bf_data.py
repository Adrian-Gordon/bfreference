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
  

  def __init__(self, datafile_path,scaler_path,seq_length, offset, output_path):
    '''
      arguments:
        datafile_path -- string, filepath for input datafile
        scaler_path -- struing, path to save scaler to
        seq_length -- integer, no of entries per timeseries
        offset -- integer, offset from start of data series to use in output. 
                  e.g. seq_length= 30, offset = 30 : skip first 30,then use the next 30


    '''
    self.data =[]
    self.included_data = pd.DataFrame() #data to be included - i.e. those with odds in the required rance
    self.processed_data =pd.DataFrame() #the processed data
    self.data = pd.read_csv(datafile_path)
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

  def preprocess(self, max_starting_lay_price, max_starting_back_price, seq_length, offset):
    for i in range (self.nsequences):
      sequence  = self.data[i * seq_length + offset: (i * seq_length) + seq_length + offset]
      add_it = False
      for index, row in sequence.iterrows():
        if row['layprice1'] <= max_starting_lay_price and row['backprice1'] <= max_starting_back_price:
          add_it = True
          
      if add_it:
          self.included_data = self.included_data.append(sequence,ignore_index = True)

    n_good_sequences = len(self.included_data) / (seq_length + offset) #the number of sequences to preprocess

    print("n_good_sequences", n_good_sequences)

    for j in range(n_good_sequences):
      sequence  = self.included_data[i * seq_length + offset: (i * seq_length) + seq_length + offset]
      starting_layprice1 = sequence['layprice1'][0]
      print(starting_layprice1)

'''
    print("Generating Sequences")
    _nsequences = len(GenerateData.data) / ( seq_length + offset)
    print(_nsequences)
    
    for i in range (_nsequences):
      _sequence = GenerateData.data[i * seq_length + offset: (i * seq_length) + seq_length + offset]
      add_it = False
      for index, row in _sequence.iterrows():
        if row['layprice1'] <= 15 and row['backprice1'] <= 20:
          add_it = True
          
      if add_it:
          GenerateData.processed_data = GenerateData.processed_data.append(_sequence,ignore_index = True)

      print("Generated Sequences")

    print("Scaling")
    scaler = StandardScaler()

    GenerateData.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=scaler.fit_transform(GenerateData.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])
      #save the scaler for later use
    print("Scaled")
    joblib.dump(scaler, scaler_path)
    print("Saved Scaler")
    print("Saving preprocessed data to: "+ output_path)
    GenerateData.processed_data.to_csv(output_path, index = False)
    print("Preprocessed data saved")
    
'''
  

#test
'''configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)
gd = GenerateData(config["input_datafile_path"],config["scaler_file_path"],config["sequence_length"], config["offset"],config["output_datafile_path"])
'''
#print("processed data: ")
#print(gd.processed_data)
#print(gd.data[0:60])
#input_batch, output_batch = gd.getTrainingSample(60,2,30,10)
#print(input_batch)
#print(output_batch)

#reshaped_input_batch = gd.reshape(np.array(input_batch),30, 40)
#print(reshaped_input_batch)

#reshaped_output_batch = gd.reshape(np.array(output_batch), 10, 2)
#print(reshaped_output_batch)