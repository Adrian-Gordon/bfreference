
#basic seqtoseq model for time series
#usage: python basicmodellearn.py <configfilename>

import sys
import json

configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)


import importlib

import tensorflow as tf

import os

from matplotlib import pyplot as plt




from basicmodel import *

generate_data_library = config["data_modulename"]

cmd = 'from ' + generate_data_library + ' import *'

exec(cmd)

#from generate_univariate_data import *

def main():
 
  datafilename = config["datafilename"]
  gd=GenerateData(datafilename)

  #create a seq2seq instance
  seq2seq  = Seq2Seq()

  loss_summary = tf.summary.scalar("loss", seq2seq.loss)
  merged = tf.summary.merge_all()


  init = tf.global_variables_initializer()
  
  with tf.Session() as sess:
   
    sess.run(init)

    for epoch in range(config["epochs"]):

     
      writer = tf.summary.FileWriter('./graphs/seq2seq/basic', sess.graph)

      #generate a data batch
      input_batch, output_batch = gd.getTrainingSample(seq_length=config["seq_length"],batch_size = config["batch_size"], input_seq_length =config["input_sequence_length"], output_seq_length = config["output_sequence_length"])

      #for beijing, it would be:
      # input_batch, output_batch = gd.getTrainingSample(batch_size = config["batch_size"], input_seq_length =config["input_sequence_length"], output_seq_length = config["output_sequence_length"])

      reshaped_input_batch = gd.reshape(np.array(input_batch),config["input_sequence_length"], config["input_dim"])

     
      reshaped_output_batch = gd.reshape(np.array(output_batch),config["output_sequence_length"], config["output_dim"])

      feed_dict={encoder_inputs[t]: reshaped_input_batch[t] for t in range(config["input_sequence_length"])}

      feed_dict.update({decoder_target_inputs[t]: reshaped_output_batch[t] for t in range(config["output_sequence_length"])})
     

      #the_decoder_outputs = sess.run(seq2seq.encode_decode,feed_dict)

      _, loss = sess.run([seq2seq.optimize, seq2seq.loss],feed_dict)

      summary = sess.run(merged,feed_dict)
      writer.add_summary(summary, epoch)
     
      print(epoch,loss)
     
    encoded = sess.run([seq2seq.encode_decode],feed_dict)
   
    saver = tf.train.Saver
    save_path = saver().save(sess, os.path.join('./', config["savefilename"]))
    print("Checkpoint saved at: ", save_path)

main()