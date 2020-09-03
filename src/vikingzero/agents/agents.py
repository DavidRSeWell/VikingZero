"""
Agents that are not specific to a single environment
"""

class HumanAgent:

    def __init__(self,env):
        self._env = env

    def act(self,s):

        while True:
            try:
                action = int(input("Enter Action: "))

                valid_actions = self._env.actions(self._env.board)

                if action in valid_actions:
                    return action
            except:

                print(f"action is invalid valid action are")
