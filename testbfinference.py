from seq2seq import BfInferer
from seq2seq import GenerateData
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

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

next_observed = [[5.8,17.54,5.9,95.1,6,137.56,6.2,264.19,6.4,27.52,6.6,34.06,6.8,19.13,7,51.29,7.2,2,7.6,5,5.7,226.33,5.6,244.61,5.5,223.2,5.4,181.41,5.3,165.55,5.2,76.31,5.1,171.9,5,70.82,4.9,34.83,4.8,27.02],
[5.9,72.64,6,134.65,6.2,262.19,6.4,27.89,6.6,39.45,6.8,19.13,7,51.29,7.2,2,7.6,5,7.8,3.88,5.8,59.57,5.7,150.75,5.6,199.31,5.5,191.83,5.4,258.91,5.3,125.55,5.2,81.31,5.1,166.8,5,70.82,4.9,26.53],
[6,163.5,6.2,258.86,6.4,31.89,6.6,47.16,6.8,21.13,7,15.29,7.2,37,7.4,2,7.6,5,7.8,5.88,5.9,123.72,5.8,109.67,5.7,182.01,5.6,193.67,5.5,170.1,5.4,267,5.3,125.55,5.2,101.31,5.1,207.75,5,22.82],
[6,166.6,6.2,260.86,6.4,31.52,6.6,49.16,6.8,21.13,7,17.29,7.2,2,7.4,36,7.6,5,7.8,5.88,5.9,205,5.8,101.89,5.7,181.66,5.6,193.67,5.5,170.1,5.4,267,5.3,128.39,5.2,149.31,5.1,158.75,5,22.82],
[6.2,245.74,6.4,95.78,6.6,53.16,6.8,21.13,7,21.29,7.2,33.64,7.4,2.8,7.6,38,7.8,5.88,8,20.02,6,138.29,5.9,262.68,5.8,153.77,5.7,253.45,5.6,164.67,5.5,156.88,5.4,239.55,5.3,175.32,5.2,101.31,5.1,158.75]]

'''

bfInferer = BfInferer("./configs/config_bf.json")



gd = GenerateData('./data/generate.csv')


n_rows = len(gd.processed_data) / 60

print(n_rows)

for row in range(0,n_rows):
  print("ROW " ,row)
  for column in range(15):
    print("coulmn " , column)
    test_sample = gd.getTestSample(30, 10, (row * 2) + 1, column)
    #print(test_sample[:,0])
    test_sample_next = gd.getTestSample(30, 15, (row * 2) + 1, column)

    lay_predictions, back_predictions = bfInferer.doInference(test_sample)

    #print(lay_predictions)
    #print(back_predictions)
    plt.figure(figsize=(15,4))
    l1, = plt.plot(test_sample_next[:,0], 'b.', label = 'Actual lay')
    l2, = plt.plot(test_sample_next[:,20], 'y.', label = 'Actual back')
    l3, = plt.plot(range(10,15),lay_predictions, 'r.', label = 'Predicted lay')
    l4, = plt.plot(range(10,15),back_predictions, 'g.', label = 'Predicted back')

    plt.legend(handles=[l1, l2, l3, l4], loc='upper left')

    plt.show()
    i = raw_input("")