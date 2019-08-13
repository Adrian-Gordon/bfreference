#Train the seq2seq keras model using config
#usage: python keraslearn <configfilepath>
import sys
import json

from kerasSeqToSeq import SeqToSeqModel

configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)


seq2seqModel = SeqToSeqModel(config)
seq2seqModel.generate_data()
seq2seqModel.train()