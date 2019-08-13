# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


from generate_bf_data import GenerateData

from trader import Trader

class SeqToSeqModel:
  def __init__(self, config):
    self.config = config
    model = keras.Sequential()
    model.add(keras.layers.LSTM(200, activation='relu', input_shape=(config['input_sequence_length'], config['input_dim'])))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(config['output_sequence_length']))
    model.compile(loss='mse', optimizer='adam')
    #print(model.summary())
    self.model = model
    self.scaler = joblib.load(config["scaler_location"])

  def generate_data(self):
    generateData = GenerateData(self.config['datafilename'])
    self.input_training_data, self.output_training_data = generateData.getTrainingSample(self.config['seq_length'],self.config['dataset_size'], self.config['input_sequence_length'], self.config['output_sequence_length'],self.config['input_attributes'],self.config['output_attribute'])
    #print("input shape: ",self.input_training_data.shape)
    #(self.input_training_data[0])
    #print("output shape: ",self.output_training_data.shape)
   #print(self.output_training_data[0])

  def train(self):
    self.model.fit(self.input_training_data,self.output_training_data, epochs = self.config['epochs'], batch_size = self.config['batch_size'], verbose = 1)
    self.model.save_weights(self.config['savefilename'])

  def load_weights(self):
    self.model.load_weights(self.config['savefilename'])

  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions[0]

  def doInference(self, data):
    columns = ["layprice1","laydepth1","layprice2","laydepth2","layprice3","laydepth3","layprice4","laydepth4","layprice5","laydepth5","layprice6","laydepth6","layprice7","laydepth7","layprice8","laydepth8","layprice9","laydepth9","layprice10","laydepth10","backprice1","backdepth1","backprice2","backdepth2","backprice3","backdepth3","backprice4","backdepth4","backprice5","backdepth5","backprice6","backdepth6","backprice7","backdepth7","backprice8","backdepth8","backprice9","backdepth9","backprice10","backdepth10"] 
    sequence = pd.DataFrame(data, columns = columns)
    trader = Trader()

    starting_layprice1 = sequence['layprice1'][0]
    starting_layprice1_ticks = trader.get_ticks(sequence['layprice1'][0])
    starting_layprice2_ticks = trader.get_ticks(sequence['layprice2'][0])
    starting_layprice3_ticks = trader.get_ticks(sequence['layprice3'][0])
    starting_layprice4_ticks = trader.get_ticks(sequence['layprice4'][0])
    starting_layprice5_ticks = trader.get_ticks(sequence['layprice5'][0])
    starting_layprice6_ticks = trader.get_ticks(sequence['layprice6'][0])
    starting_layprice7_ticks = trader.get_ticks(sequence['layprice7'][0])
    starting_layprice8_ticks = trader.get_ticks(sequence['layprice8'][0])
    starting_layprice9_ticks = trader.get_ticks(sequence['layprice9'][0])
    starting_layprice10_ticks = trader.get_ticks(sequence['layprice10'][0])

    starting_backprice1_ticks = trader.get_ticks(sequence['backprice1'][0])
    starting_backprice2_ticks = trader.get_ticks(sequence['backprice2'][0])
    starting_backprice3_ticks = trader.get_ticks(sequence['backprice3'][0])
    starting_backprice4_ticks = trader.get_ticks(sequence['backprice4'][0])
    starting_backprice5_ticks = trader.get_ticks(sequence['backprice5'][0])
    starting_backprice6_ticks = trader.get_ticks(sequence['backprice6'][0])
    starting_backprice7_ticks = trader.get_ticks(sequence['backprice7'][0])
    starting_backprice8_ticks = trader.get_ticks(sequence['backprice8'][0])
    starting_backprice9_ticks = trader.get_ticks(sequence['backprice9'][0])
    starting_backprice10_ticks = trader.get_ticks(sequence['backprice10'][0])

    #print("starting_layprice1_ticks: {}".format(starting_layprice1_ticks))

    #print("raw sequqnce: {}".format(sequence['layprice1']))

    #need to iterate over them

    tickified_sequence = sequence.copy()
    tickified_sequence['layprice1'] = tickified_sequence['layprice1'].apply(lambda x: trader.get_ticks(x) - starting_layprice1_ticks)
    tickified_sequence['layprice2'] = tickified_sequence['layprice2'].apply(lambda x: trader.get_ticks(x) - starting_layprice2_ticks)
    tickified_sequence['layprice3'] = tickified_sequence['layprice3'].apply(lambda x: trader.get_ticks(x) - starting_layprice3_ticks)
    tickified_sequence['layprice4'] = tickified_sequence['layprice4'].apply(lambda x: trader.get_ticks(x) - starting_layprice4_ticks)
    tickified_sequence['layprice5'] = tickified_sequence['layprice5'].apply(lambda x: trader.get_ticks(x) - starting_layprice5_ticks)
    tickified_sequence['layprice6'] = tickified_sequence['layprice6'].apply(lambda x: trader.get_ticks(x) - starting_layprice6_ticks)
    tickified_sequence['layprice7'] = tickified_sequence['layprice7'].apply(lambda x: trader.get_ticks(x) - starting_layprice7_ticks)
    tickified_sequence['layprice8'] = tickified_sequence['layprice8'].apply(lambda x: trader.get_ticks(x) - starting_layprice8_ticks)
    tickified_sequence['layprice9'] = tickified_sequence['layprice9'].apply(lambda x: trader.get_ticks(x) - starting_layprice9_ticks)
    tickified_sequence['layprice10'] = tickified_sequence['layprice10'].apply(lambda x: trader.get_ticks(x) - starting_layprice10_ticks)

    tickified_sequence['backprice1'] = tickified_sequence['backprice1'].apply(lambda x: trader.get_ticks(x) - starting_backprice1_ticks)
    tickified_sequence['backprice2'] = tickified_sequence['backprice2'].apply(lambda x: trader.get_ticks(x) - starting_backprice2_ticks)
    tickified_sequence['backprice3'] = tickified_sequence['backprice3'].apply(lambda x: trader.get_ticks(x) - starting_backprice3_ticks)
    tickified_sequence['backprice4'] = tickified_sequence['backprice4'].apply(lambda x: trader.get_ticks(x) - starting_backprice4_ticks)
    tickified_sequence['backprice5'] = tickified_sequence['backprice5'].apply(lambda x: trader.get_ticks(x) - starting_backprice5_ticks)
    tickified_sequence['backprice6'] = tickified_sequence['backprice6'].apply(lambda x: trader.get_ticks(x) - starting_backprice6_ticks)
    tickified_sequence['backprice7'] = tickified_sequence['backprice7'].apply(lambda x: trader.get_ticks(x) - starting_backprice7_ticks)
    tickified_sequence['backprice8'] = tickified_sequence['backprice8'].apply(lambda x: trader.get_ticks(x) - starting_backprice8_ticks)
    tickified_sequence['backprice9'] = tickified_sequence['backprice9'].apply(lambda x: trader.get_ticks(x) - starting_backprice9_ticks)
    tickified_sequence['backprice10'] = tickified_sequence['backprice10'].apply(lambda x: trader.get_ticks(x) - starting_backprice10_ticks)


    #print("sequence ticks diff: {}".format(tickified_sequence['layprice1']))

    #standardise
    tickified_sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=self.scaler.transform(tickified_sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])

    #print(sequence)
    input_data = tickified_sequence[self.config['input_attributes']]
    input = np.array(input_data)
    input = input.reshape(1, 10, 1)

    #print("input: {}".format(input))

    results = self.predict([input])

    #print("results: {}".format(results))
    #de-standardize
    predicted_ar = np.zeros((5, 40))
    for row2 in range(5):
      predicted_ar[row2][0]= results[row2]
      

    descaled_predicted_ar = self.scaler.inverse_transform(predicted_ar)
    #print("descaled_predicted_ar: {} ".format(descaled_predicted_ar[:,0]))
    #descaled_predicted_ar[:,0] += starting_layprice1
    descale = lambda x: trader.increment_price(starting_layprice1,x)

    descaled = map(descale,descaled_predicted_ar[:,0])
    #descaled_predicted_ar = descaled_predicted_ar.apply(lambda x: trader.increment_price(starting_layprice1,x))
    #print("descaled: {}".format(descaled))

    return descaled


