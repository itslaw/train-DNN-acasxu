Contains two acasxu DNNs.

- One rounds the floating point weights/biases so that the DNN can be implemented later on in 8-10 bits. It will automatically detect the minimum number of bits that can be used (while also not sacrificing correctness). Uses the same number of bits for all weights/biases.

- The other retrains the network and forces the weights to an 8 bit representation.

Both are extracted from the rlv file.
