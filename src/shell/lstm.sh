#! /bin/bash

# python maxlen=, n_hidden=, nepoch=, activation=, pic_path1=, pic_path2=

python src/lstm.py 5 100 1000 linear \
  result/pictures/lstm/len5_neuron100_epoch1000_linear_activtion_fitting.png \
  result/pictures/lstm/len5_neuron100_epoch1000_linear_activtion_pred.png &

python src/lstm.py 10 100 1000 linear \
  result/pictures/lstm/len10_neuron100_epoch1000_linear_activtion_fitting.png \
  result/pictures/lstm/len10_neuron100_epoch1000_linear_activtion_pred.png &

python src/lstm.py 5 200 1000 linear \
  result/pictures/lstm/len5_neuron200_epoch1000_linear_activtion_fitting.png \
  result/pictures/lstm/len5_neuron200_epoch1000_linear_activtion_pred.png &

python src/lstm.py 10 200 1000 linear \
  result/pictures/lstm/len10_neuron200_epoch1000_linear_activtion_fitting.png \
  result/pictures/lstm/len10_neuron200_epoch1000_linear_activtion_pred.png &

wait