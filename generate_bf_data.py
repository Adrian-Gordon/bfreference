#Generate data samples from an input file - assumed to have been preprocessed (standardised, etc) - as is done in 'preprocess_bf_data.py'

import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class GenerateData:
  #data =[]
  #processed_data =pd.DataFrame()
  def __init__(self, datafile_path):
    self.processed_data = pd.read_csv(datafile_path)
    

  def getTrainingSample(self,seq_length,batch_size, input_seq_length, output_seq_length):
    data = self.processed_data
    _nsequences = len(data) / seq_length
    input_batches = []
    output_batches = []

    for i in range(batch_size):
      #n_sequences = len(data) / (input_seq_length + output_seq_length)
      sequence_index = np.random.randint(0,_nsequences)
      #sequence_index = 0
     # print(sequence_index)
      n_starting_indexes = seq_length - (input_seq_length + output_seq_length) -1
      #print(n_starting_indexes)
      starting_index = np.random.randint(0,n_starting_indexes)
      #starting_index = 0
      #print(starting_index)
      starting_offset = sequence_index * seq_length
      #print(starting_offset)
      an_input_sequence = data[starting_offset + starting_index: starting_offset + starting_index + input_seq_length]
      #print(an_input_sequence)

      input_data = an_input_sequence[['layprice1','laydepth1','backprice1','backdepth1']]
      input_batches.append(np.array(input_data))

      an_output_sequence = data[starting_offset + starting_index + input_seq_length:starting_offset +starting_index + input_seq_length + output_seq_length]
      #print(an_output_sequence)
      output_data = an_output_sequence[['layprice1','backprice1']]
      output_batches.append(np.array(output_data))


    return input_batches, output_batches

  def getTestSample(self, seq_length, input_seq_length, offset1, offset2):
    '''seq_length - length of the underlying sequence
        input_seq_length - length of input sequence to generate
        offset1 - offset into sequence
        offset2 - offset within sequence
    '''
    input_batches = []
    print("seq_length: ", seq_length, " input_seq_length: ", input_seq_length, " offset1: ", offset1, "offeset2: ", offset2)
    start_index = (offset1 * seq_length) + offset2
    end_index = start_index + input_seq_length
    print("start:", start_index, "end: ", end_index)
    an_input_sequence = self.processed_data[start_index:end_index]
    input_data = an_input_sequence[['layprice1','laydepth1','backprice1','backdepth1']]

    return np.array(input_data)
    #input_batches.append(np.array(input_data))
    #return input_batches


  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))

#test
#gd = GenerateData('./data/preprocessed_generate.csv')
#print("processed data: ")
#print(gd.processed_data)
#print(gd.data[0:60])
#input_batch, output_batch = gd.getTrainingSample(30,1,10,5)
#print(input_batch)
#print(output_batch)

#reshaped_input_batch = gd.reshape(np.array(input_batch),30, 40)
#print(reshaped_input_batch)

#reshaped_output_batch = gd.reshape(np.array(output_batch), 10, 2)
#print(reshaped_output_batch)