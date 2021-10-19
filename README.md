# SequencePredictionRNN

This code implements a multi-layer character-level RNN using LSTM's. It takes in a sequence of characters, trains on it and learns predict new characters in the sequence.

# Data

The underlying dataset includes one of the stories of Sherlock Holmes  (A Scandal in Bohemia), the text file can be found in the data folder. In order to train the model on a new sequence, you can add your own input data inside the data folder. 

# Training

Staetful RNN's are used, batches are created in such a way such that batches have in sequence in continuous of the previous batches so that output of a batch can be fed as an input to the next batch. Stateful RNN's are useful when training on one big sequences.

train.py can be used to train the model the underling dataset

# Model Architecture

model.py contains the model architecture, the current model contains an embedding layer followed by 3 LSTM layers with 256 units and eventually a TimeDistributedDense Layer

# Sampler

Once the model is trained sample.py file could be used to generate a new sequence based on the header/starter provied

# Results
