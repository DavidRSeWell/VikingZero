# VikingZero

This repository is an effort to understand the AlphaGo family of algorithms. 
 

### Installation
 
 ```
 pip install vikingzero
 
 ```
Or if you are familiar with poetry (https://python-poetry.org/) you can install locally using the following

```
git clone https://github.com/befeltingu/VikingZero.git
cd VikingZero
poetry install
```

### Usage

For example of how to run go to notebooks/run_alphago.ipynb and run

This project uses yaml files for run configurations. notebooks/ has a few example files to get started

### Agents

There are a few types of agents that you can experiment with in this project. 

* MiniMax
* AlphaBeta Pruning
* AlphaBeta Pruing with depth limited search
* Monte carlo tree search - UCT
* AlphaZero - ( Look to notebooks/connect4.yaml for a connect4 configuration)


### Play

Go to https://github.com/befeltingu/VikingDashboard for a dashboard where you can play against some pre trained models






