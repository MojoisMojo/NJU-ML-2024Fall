import numpy as np
from typing import Dict

m, n = 4, 4
gridsValue = np.zeros((m, n))
gridsNextValue = np.zeros((m, n))
gridsAction = np.zeros((m, n))
actionsDesc = ["U", "L", "D", "R"]
actionsDict = [(-1, 0), (0, -1), (1, 0), (0, 1)]
goal_reward = 0.8
idle_reward = 1.0
print("goal_reward:", goal_reward)
print("idle_reward:", idle_reward)
gamma = 0.9
war_poss = [(2, 1)]
goal_poss = [(1, 3), (2, 3)]

"""
  0 1 2 3
0 
1       v
2   x   v 
3
"""


def isLegal(pos):
    i, j = pos
    if i < 0 or i >= m or j < 0 or j >= n:
        return False
    if pos in war_poss:
        return False
    return True


def isGoal(pos):
    return pos in goal_poss


def getNextAction(action) -> Dict[int, float]:
    """
    return: next action after taking action with probability (probs sum 1)
    """
    return {action: 0.8, (action + 1) % 4: 0.1, (action + 3) % 4: 0.1}


def getNextState(pos, action) -> Dict[tuple[int,int], float]:
    """
    return: next state after taking action from pos with probability (probs sum 1)
    """
    i, j = pos
    state_dict = {}
    for nextAction, prob in getNextAction(action).items():
        di, dj = actionsDict[nextAction]
        ni, nj = i + di, j + dj
        if not isLegal((ni, nj)):
            state_dict[pos] = state_dict.get(pos, 0) + prob
        else:
            state_dict[(ni, nj)] = state_dict.get((ni, nj), 0) + prob
    assert sum(state_dict.values()) == 1
    return state_dict

def getReward(pos):
    i, j = pos
    assert isLegal(pos)
    if isGoal(pos):
        return goal_reward
    return idle_reward


def getAction(pos):
    i, j = pos
    return gridsAction[i][j]


def getValue(pos):
    i, j = pos
    return gridsValue[i][j]


def setValue(pos, value):
    i, j = pos
    gridsValue[i][j] = value


def setAction(pos, action):
    i, j = pos
    gridsAction[i][j] = action


def getBestActionAndValue(pos):
    i, j = pos
    bestAction = None
    bestValue = -1e9
    for action in range(4):
        nextValue = 0
        for nextPos, prob in getNextState(pos, action).items():
            nextValue += prob * getValue(nextPos)
        if nextValue > bestValue:
            bestValue = nextValue
            bestAction = action
    return bestAction, bestValue

if __name__ == "__main__":
    for i in range(m):
        for j in range(n):
            setAction((i, j), 0)
            setValue((i, j), 0)
    epochs = 20
    for epoch in range(1,epochs+1):
        for i in range(m):
            for j in range(n):
                if isLegal((i, j)):
                    if isGoal((i, j)):
                        gridsNextValue[i][j] = goal_reward
                    else:
                        bestAction, bestNextValue = getBestActionAndValue((i, j))
                        setAction((i, j), bestAction)
                        gridsNextValue[i][j] = getReward((i, j)) + gamma * bestNextValue
        if np.allclose(gridsValue, gridsNextValue):
            print("Converged in epoch", epoch)
            break
        gridsValue = np.copy(gridsNextValue)
        gridsNextValue = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            print(f"action: {actionsDesc[int(getAction((i, j)))]}", end=" ")
            print(f"value: {getValue((i, j)):.4f}", end=" ")
        print()
    
    print("done")