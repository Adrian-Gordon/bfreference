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
    print(model.summary())
    self.model = model
    self.scaler = joblib.load(config["scaler_location"])

  def generate_data(self):
    generateData = GenerateData(self.config['datafilename'])
    self.input_training_data, self.output_training_data = generateData.getTrainingSample(self.config['seq_length'],self.config['dataset_size'], self.config['input_sequence_length'], self.config['output_sequence_length'])
    print("input shape: ",self.input_training_data.shape)
    print(self.input_training_data[0])
    print("output shape: ",self.output_training_data.shape)
    print(self.output_training_data[0])

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

    print("starting_layprice1_ticks: ", starting_layprice1_ticks)

    print("raw sequqnce: ", sequence['layprice1'])

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


    print("sequence ticks diff", tickified_sequence['layprice1'])

    #standardise
    tickified_sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=self.scaler.transform(tickified_sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])

    #print(sequence)
    input_data = tickified_sequence[['layprice1']]
    input = np.array(input_data)
    input = input.reshape(1, 10, 1)

    print(input)

    results = self.predict([input])

    print("results", results)
    #de-standardize
    predicted_ar = np.zeros((5, 40))
    for row2 in range(5):
      predicted_ar[row2][0]= results[row2]
      

    descaled_predicted_ar = self.scaler.inverse_transform(predicted_ar)
    print(descaled_predicted_ar[:,0])
    #descaled_predicted_ar[:,0] += starting_layprice1
    descale = lambda x: trader.increment_price(starting_layprice1,x)

    descaled = descale(descaled_predicted_ar)
    #descaled_predicted_ar = descaled_predicted_ar.apply(lambda x: trader.increment_price(starting_layprice1,x))
    print(descaled)

    return descaled


#test
configs ={
"data_modulename":"generate_bf_data",
"datafilename": "../data/preprocessed_generate.csv",
"rawdatafilename": "./data/raw_generate.csv",
"savefilename":"../save/save_keras",
"seq_length": 30,
"input_sequence_length": 10,
"output_sequence_length": 5,
"dataset_size" :10000,
"batch_size" : 100,
"input_dim" : 1,
"output_dim":1,
"num_stacked_layers" :1,
"hidden_dim" :3,
"learning_rate" :0.05,
"l2_regularization_lambda":0.03,
"gradient_clipping": 2.5,
"epochs": 20,
"scaler_location":"../data/scaler.save"
}

''' Train
seq_to_seq_model = SeqToSeqModel(configs)
seq_to_seq_model.generate_data()
seq_to_seq_model.train()

'''

''' Do Inference
'''
seq_to_seq_model = SeqToSeqModel(configs)
seq_to_seq_model.load_weights()

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

result = seq_to_seq_model.doInference(raw_data1)
print(result)
'''

#predictions = seq_to_seq_model.predict(np.array([[[-0.00764403],[-0.00899999],[-0.00764403],[-0.00696605],[-0.00628807],[-0.00628807],[-0.00493212],[-0.00493212],[-0.00628807],[-0.0056101]]]))
#print(predictions)
#seq_to_seq_model.generate_data()
#print(seq_to_seq_model.input_training_data)
#seq_to_seq_model.train()
'''
