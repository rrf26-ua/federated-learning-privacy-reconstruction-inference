# FedAvg E=1 baseline with lr=0.05

FedAvg with one local epoch, 20 rounds, 6 clients, batch size 32 and learning rate 0.05 reached 76.04% centralized test accuracy on CIFAR-10.

Compared with the previous E=1 run using learning rate 0.02, the final accuracy improved from 75.01% to 76.04%, and the final loss improved from 0.7824 to 0.7117.

This configuration is selected as the main FedAvg E=1 baseline. The gain over lr=0.02 is moderate, so further accuracy improvements should focus on increasing the number of local epochs rather than only tuning the learning rate.
