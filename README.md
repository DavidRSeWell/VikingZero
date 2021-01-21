# VikingZero

This repository is an effort to understand the AlphaGo family of algorithms. 
 

### Installation
 
 ```
 pip install vikingzero
 ```
Or if you are familiar with poetry (https://python-poetry.org/) you can install from source using the following

```
git clone https://github.com/befeltingu/VikingZero.git
cd VikingZero
poetry install
```
You can also use the setup.py file or requriements file for installing from source. 

### Usage

For example of how to train an agent go to notebooks/run_alphago.ipynb and run the cells. 

This project uses yaml files for run configurations. notebooks/ has a few example files to get started. 

There is a notebooks/step_tutorial.ipynb file that is meant as a tutorial for better understanding of the Monte
Carlo Tree search steps. It used pydot with graphviz and requires that you have an installation of graphviz. 


### Agents

There are a few types of agents that you can experiment with in this project. 

* MiniMax
* AlphaBeta Pruning
* AlphaBeta Pruing with depth limited search
* Monte carlo tree search - UCT
* AlphaZero - ( Look to notebooks/connect4.yaml for a connect4 configuration)


### Play

Go to https://github.com/befeltingu/VikingDashboard for a dashboard where you can play against some pre trained models


![Alt text](https://github.com/befeltingu/VikingDashboard/blob/master/public/example_play.png)



