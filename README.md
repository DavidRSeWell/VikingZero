# VikingZero

Building AlphaGo Zero from scratch.
 

### Installation
Requires poetry to install currently

https://python-poetry.org/docs/

```
git clone https://github.com/befeltingu/VikingZero.git
cd VikingZero
poetry install
```

### Usage

```
cd tests
python test_sacred_file.py
```

This will kick off a run of a MCTS agent using the configuration that is specified in the 
tests/connect4.yaml file. The sacred tests make use of the python sacred package for better 
tracking and reproducibility of tests. This will automatically store certain statistics in a 
local MongoDB database. Using just a local file store is also possible. It is recommended to 
use the omniboard front end tool for viewing test output. Or check out the development branch
for a dash front end. 

https://sacred.readthedocs.io/en/stable/index.html

https://github.com/vivekratnavel/omniboard


### TODO 
In development
1. Improve Dash Front End
    1. View saved games from sacred experiment included with agent stats of the current board state
    2. General Cleanup for Dash

2. Agents
    1. Save off agents post run. 
    2. Allow for creation of "Expert" database from a minimax or other "good" agent
    3. Implement MCTS + Expert Net
    
3.  Manim - Bring in Manim support for helpful visuals 





