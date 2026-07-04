# Discarded server optimizer smoke tests

## Context

After observing slow convergence with strict FedSGD, server-side optimizer strategies were tested using FedSGD-style client updates.

The objective was to check whether server-side momentum or adaptive optimization could improve convergence without switching back to FedAvg local training.

## FedAvgM

FedAvgM was tested with different combinations of client learning rate, server learning rate and server momentum.

Some configurations produced NaN losses, while the stable configurations remained close to random accuracy and clearly underperformed strict FedSGD at comparable early rounds.

The best stable FedAvgM smoke result reached only 13.26% accuracy after 5 rounds, which is below the strict FedSGD baseline at the same early stage.

## FedAdam

FedAdam was tested with conservative eta values: 0.001, 0.005, 0.01 and 0.02.

None of the configurations improved over strict FedSGD. Accuracy remained close to random guessing, and the larger eta values showed increasing evaluation loss.

## Decision

FedAvgM and FedAdam were discarded for this specific setup.

The main optimization path was moved back to FedAvg with different numbers of local epochs. FedAvg E=1 and FedAvg E=2 provided substantially better accuracy and more stable training.

## Interpretation

These discarded experiments are still useful methodologically: they show that server optimizer strategies were considered and tested, but were not selected because they did not improve the training dynamics in this implementation.
