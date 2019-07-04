import tensorflow as tf
import numpy as np

class GenerateData:
  def __init__(self):
    self.x = np.linspace(0, 30, 105)

    self.train_data_x = self.x[:85]

    self.y = 2 * np.sin(self.x)

    self.train_y = self.y.copy()

    self.noise_factor = 0.5

    self.train_y += np.random.randn(105)* self.noise_factor



  def true_signal(self,x):
    y = 2 * np.sin(x)
    return(y)

  def noise_func(self,x, noise_factor = 0.5):
    return np.random.randn(len(x)) * self.noise_factor

  def generate_y_values(self,x):
    return self.true_signal(x) + self.noise_func(x)


  def getTrainingSample(self,batch_size, input_seq_length, output_seq_length):
    data = self.train_data_x
    possibleStartingPoints = len(data) - input_seq_length - output_seq_length
    #print(possibleStartingPoints)
    start_x_indexes = np.random.choice(range(possibleStartingPoints), batch_size)
    #print(start_x_indexes)

    input_seq_x =[data[i:(i+input_seq_length)] for i in start_x_indexes]
    output_seq_x = [data[(i+input_seq_length):(i+input_seq_length + output_seq_length)] for i in start_x_indexes]

    input_seq_y = [self.generate_y_values(x) for x in input_seq_x]
    output_seq_y = [self.generate_y_values(x) for x in output_seq_x]

    #print(input_seq_x)
    #print(input_seq_y)
    #print(output_seq_x)
    #print(output_seq_y)
    return np.array(input_seq_y), np.array(output_seq_y)

  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))







 