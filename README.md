# LLM-DFL
Codes for Paper “Large Language Model-Empowered Decision-focused Learning in Local Energy Communities”, Authored by Yangze Zhou, Yu Zuo, Daniel Kirschen, Yi Wang.


## Environments
The environments for the code can be installed by
```
conda env create -f environments.yml
```
## Data
The load data used for experiments can be found in ```./Data/GEF_data```.

## Code
There are four settings in our work. 
| Filefold name      | Description |
| ----------- | ----------- |
|NN+LP   |Forecasting model is a neural network and optimization problem that ignores the integer constraints. This setting can be handled by Optnet.|
|Tree+LP|Forecasting model is a Tree model and optimization problem that ignores the integer constraints in the UC problem. Due to the tree model not being trained by gradient descent, it is hard for Optnet to train a DFL model.|
|NN+MILP|Forecasting model is a Tree model, and optimization problem is a mixed integer linear problem (MILP). It is also hard for Optnet to train a DFL model because the integer variable makes the gradient of the decision-making problem hard to calculate.|
|NN+SO| Function related to online learning for parametric approach|
