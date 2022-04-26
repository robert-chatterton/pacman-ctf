# Team we trained against that used pathfinding algorithms. 
# Full repo can be found here: https://github.com/pcgotan/PacmanCTF_Agent/blob/master/myTeam3.py

from captureAgents import CaptureAgent
import random
import time
import util
import operator
from game import Directions
import game
import distanceCalculator
import math
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent', **args):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.tocenter = True
        self.possiblePosition = [None] * 4
        self.start = gameState.getAgentPosition(self.index)
        self.datalayout = gameState.data.layout
        self.discount = 0.9
        global strategy
        strategy = [0, 0, 0, 0]
        for i in self.getTeam(gameState):
            strategy[i] = 3  # To Border
        self.toborder = True
        global targetFood
        targetFood = [[], [], [], []]
        global Capsule_list
        Capsule_list = [[], [], [], []]
        global possible_Position_List
        possible_Position_List = []
        for i in range(0, gameState.getNumAgents()):
            possible_Position_List.append(util.Counter())
        self.width, self.height = gameState.getWalls().asList()[-1]
        self.border = []
        self.repeat = self.width / 2 + self.height + 5

        if self.red:
            self.border_x = self.datalayout.width / 2 - 1
        else:
            self.border_x = self.datalayout.width / 2

        for i in range(self.height):
            if not self.datalayout.walls[self.border_x][i]:
                self.border.append(((self.border_x), i))
        the_other = self.seperate_index(gameState)

        if gameState.getAgentPosition(self.index)[1] > gameState.getAgentPosition(the_other)[1]:
            self.seperatep = self.border[int(3 * len(self.border) / 4)]
        elif gameState.getAgentPosition(self.index)[1] < gameState.getAgentPosition(the_other)[1]:
            self.seperatep = self.border[int(1 * len(self.border) / 4)]
        else:
            if self.index == self.getTeam(gameState)[0]:
                self.seperatep = self.border[int(3 * len(self.border) / 4)]
            else:
                self.seperatep = self.border[int(1 * len(self.border) / 4)]

        # All possiblePositionList begin with the agent at its inital position
        for i, val in enumerate(possible_Position_List):
            if i in self.getOpponents(gameState):
                possible_Position_List[i][gameState.getInitialAgentPosition(
                    i)] = 1.0

    def chooseAction(self, gameState):  # choice the action base one strategy
        """
        Picks among actions randomly.
        """
        '''
        You should change this in your own agent.
        '''

        start_time = time.time()
        self.reward = util.Counter()
        self.mypos = gameState.getAgentState(self.index).getPosition()
        self.border.sort(key=self.getDistance)
        targetFood[self.index].sort(key=self.getDistance)
        self.choose_strategy(gameState)
        self.get_position_pro(gameState)
        _, _, possible_action = self.getdiscountrewards(gameState, self.mypos)
        action = self.getretutnaction(gameState)
        ini_num = 0
        # for a in self.getPossibleActions(self.mypos):
        #  ini_num += self.reward[self.getNextPosition(self.mypos,a)]
        if time.time()-start_time > 0.9:
            self.repeat = self.repeat-3
        for i, v in possible_action.items():
            ini_num += self.reward[v]
        if ini_num == 0:
            if strategy[self.index] == 1:
                if len(targetFood[self.index]) > 1:
                    action = self.aStarSearch(
                        gameState, targetFood[self.index][0])
            elif strategy[self.index] == 3:

                action = self.aStarSearch(gameState, self.seperatep)
            else:
                if len(self.enemy_pacmen) > 0:
                    for i in self.getOpponents(gameState):
                        if gameState.getAgentState(i).isPacman:
                            if gameState.getAgentState(i).getPosition() == None:
                                pacmanPosition = self.possiblePosition[i]
                            else:
                                pacmanPosition = gameState.getAgentState(
                                    i).getPosition()
                            action = self.aStarSearch(
                                gameState, pacmanPosition)
        return action

    def getretutnaction(self, gameState):  # choose the action with max rewards
        x, y = self.mypos
        action_list = gameState.getLegalActions(self.index)
        action_counter = util.Counter()
        for i in action_list:
            if i == "Stop":
                action_counter[i] = self.reward[(x, y)]
            if i == "West":
                action_counter[i] = self.reward[(x-1, y)]
            if i == "East":
                action_counter[i] = self.reward[(x+1, y)]
            if i == "South":
                action_counter[i] = self.reward[(x, y-1)]
            if i == "North":
                action_counter[i] = self.reward[(x, y+1)]
        return action_counter.argMax()

    def e_posistion(self, e, i):
        if e.getPosition() == None:
            enemy_Position = self.possiblePosition[i]
        else:
            enemy_Position = e.getPosition()
        return enemy_Position

    def broderfeatures(self, gameState, pos):
        features = util.Counter()
        agentState = gameState.getAgentState(self.index)
        if self.tocenter:
            if pos == self.seperatep:
                features['br'] = 1.00
        else:
            if pos in self.border:
                features['br'] = 1.00
        for i in self.getOpponents(gameState):
            e = gameState.getAgentState(i)
            enemy_Position = self.e_posistion(e, i)
            if e.isPacman and agentState.scaredTimer > 0:
                if pos == enemy_Position:
                    features['pn'] = -1.00
            else:
                if e.getPosition() == None and e.scaredTimer < 3 and (enemy_Position == pos):
                    features['gt'] = 1.00
                elif enemy_Position == pos:
                    if e.scaredTimer < 3:
                        features['gt'] = 1.00
                    if self.getDistance(pos) == 1 and (0 < e.scaredTimer):
                        features['gt'] = -1.00
        return features

    def feature_of_ghosts(self, gameState, pos):
        f_ghost = 0
        for i in self.getOpponents(gameState):
            e = gameState.getAgentState(i)
            if not e.isPacman:
                if e.getPosition() == None and e.scaredTimer < 3:
                    if self.possiblePosition[i] == pos:
                        f_ghost = 1.00
                elif e.getPosition() == pos:
                    if e.scaredTimer < 3:
                        f_ghost = 1.00
                    if self.getDistance(pos) == 1 and e.scaredTimer > 0:
                        f_ghost = -1.00
        return f_ghost

    def defencefeatures(self, gameState, pos):
        features = util.Counter()
        for i in self.getOpponents(gameState):
            e = gameState.getAgentState(i)
            if e.isPacman:
                pacmanPosition = self.e_posistion(e, i)
                if pos == pacmanPosition:
                    features['pn'] = 1.00
            features['gt'] = self.feature_of_ghosts(gameState, pos)
        return features

    def ghosts_list(self, enemies):
        ghosts = []
        for a in enemies:
            if not a.isPacman and a.getPosition() != None and a.scaredTimer < 6:
                ghosts.append(a)
        return ghosts

    def compute_gostsDis(self, ghosts):
        ghostDist = 0
        if len(ghosts) > 0:
            ghostDists = [self.getDistance(a.getPosition()) for a in ghosts]
            ghostDist = min(ghostDists)
        return ghostDist

    def get_range(self, foodList, targetNum):
        max = int(math.ceil(min((targetNum, (len(foodList)) / 2))))
        return max

    def attackfeatures(self, gameState, pos):
        features = util.Counter()
        agentState = gameState.getAgentState(self.index)
        foods = self.getFood(gameState)
        foodList = foods.asList()
        capsuleList = self.getCapsules(gameState)
        targetNum = len(foodList)
        if targetNum < 1:
            targetNum = 1
        Agent2 = self.seperate_index(gameState)
        otherAgentPos = gameState.getAgentState(Agent2).getPosition()
        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        ghosts = self.ghosts_list(enemies)
        ghostDist = self.compute_gostsDis(ghosts)
        targetFood[self.index] = foodList
        Capsule_list[self.index] = capsuleList
        targetCapsuleNum = len(capsuleList) / 2

        if strategy[Agent2] == 1:
            targetFood[self.index] = []
            Capsule_list[self.index] = []
            foodList.sort(key=self.getDistance)
            capsuleList.sort(key=self.getDistance)
            max = self.get_range(foodList, targetNum)
            if self.index == self.getTeam(gameState)[0]:
                for i in range(max):
                    targetFood[self.index].append(foodList[i])
                if len(capsuleList) == 1:
                    targetCapsuleNum = 1
                for i in range(targetCapsuleNum):
                    Capsule_list[self.index].append(capsuleList[i])
            else:
                j = 0
                for i in range(len(foodList)):
                    if j >= math.ceil(min((targetNum, (len(foodList)) / 2))):
                        break
                    if foodList[i] not in targetFood[Agent2]:
                        targetFood[self.index].append(foodList[i])
                        j = j + 1
                    else:
                        if self.getDistance(foodList[i]) < self.getMazeDistance(otherAgentPos, foodList[i]):
                            targetFood[Agent2].remove(foodList[i])
                            targetFood[self.index].append(foodList[i])
                            j = j + 1
                k = 0
                for i in range(len(capsuleList)):
                    if k >= targetCapsuleNum:
                        break
                    if capsuleList[i] not in Capsule_list[Agent2]:
                        Capsule_list[self.index].append(capsuleList[i])
                        k = k + 1
                    else:
                        if self.getDistance(capsuleList[i]) < self.getMazeDistance(otherAgentPos, capsuleList[i]):
                            Capsule_list[Agent2].remove(capsuleList[i])
                            Capsule_list[self.index].append(capsuleList[i])
                            k = k + 1
        features['gt'] = self.feature_of_ghosts(gameState, pos)

        if 0 < ghostDist < 6:
            for capsule in capsuleList:
                if capsule == pos and self.getDistance(pos) < ghostDist / 2:
                    features['ce'] = 1.0
            if (pos) in self.border and agentState.isPacman:
                features['br'] = 1.0

        elif len(foodList) <= 2:
            if (pos) in self.border and agentState.isPacman:
                features['br'] = 1.0
        else:
            if (pos) in targetFood[self.index]:
                features['fd'] = 1.00
            notScaredGhosts = [
                a for a in enemies if not a.isPacman and a.scaredTimer == 0]
            for capsule in capsuleList:
                if capsule == (pos):
                    if ghostDist > 0:
                        if self.getDistance(pos) < ghostDist / 2:
                            features['ce'] = 1.0
                    else:
                        features['ce'] = 1.0
                    if len(notScaredGhosts) == 0:
                        features['ce'] = 0.0
        return features

    def getweight(self):
        return {'br': 1, 'pn': 1, 'gt': -1, 'fd': 1, 'ce': 1, }

    def guessposition(self, gameState):  # guess the position of enemy
        for enemy in self.getOpponents(gameState):
            for x in range(0, self.width):
                for y in range(0, self.height):
                    if not self.datalayout.walls[x][y]:
                        manhattanDistance = abs(
                            self.mypos[0] - x) + abs(self.mypos[1] - y)
                        possible_Position_List[enemy][(x, y)] *= gameState.getDistanceProb(manhattanDistance,
                                                                                           gameState.getAgentDistances()[
                                                                                               enemy])

    def getDistance(self, pos1):
        return self.getMazeDistance(self.mypos, pos1)

    # Computes a linear combination of features and feature weights
    def evaluate(self, gameState, pos):

        if strategy[self.index] == 1:  # Attack
            features = self.attackfeatures(gameState, pos)
        elif strategy[self.index] == 2:  # Defence
            features = self.defencefeatures(gameState, pos)
        elif strategy[self.index] == 3:  # To Border
            features = self.broderfeatures(gameState, pos)

        weight = self.getweight()
        return features * weight
    # computes the next state rewards when execute action

    def getdiscountrewards(self, gameState, position):
        x, y = position
        new_reward = {}
        possibleaction = {"Stop": position}
        if not gameState.data.layout.walls[int(x) - 1][int(y)]:
            possibleaction["West"] = (x - 1, y)
        if not gameState.data.layout.walls[int(x) + 1][int(y)]:
            possibleaction["East"] = (x + 1, y)
        if not gameState.data.layout.walls[int(x)][int(y) - 1]:
            possibleaction["South"] = (x, y - 1)
        if not gameState.data.layout.walls[int(x)][int(y) + 1]:
            possibleaction["North"] = (x, y + 1)
        for i, v in possibleaction.items():
            new_reward[i] = self.discount * self.reward[v]
        key = max(new_reward, key=new_reward.get)
        value = new_reward[key]
        return key, value, possibleaction

    def updateguess(self, gameState):  # update the guess position of enemy
        for i in self.getOpponents(gameState):
            tem = util.Counter()
            tem_pos = possible_Position_List[i]
            if gameState.getAgentPosition(i) != None:
                tem[gameState.getAgentPosition(i)] = 1.0
            else:
                for p in tem_pos:
                    if not gameState.hasWall(p[0], p[1]) and tem_pos[p] > 0:
                        _, _, next_action = self.getdiscountrewards(
                            gameState, p)
                        for z, v in next_action.items():
                            tem[v] += tem_pos[p]
                if len(tem) == 0:
                    if self.getPreviousObservation() != None and self.getPreviousObservation().getAgentPosition(
                            i) != None:
                        tem[gameState.getInitialAgentPosition(i)] = 1.0
                    else:
                        for x in range(self.width):
                            for y in range(self.height):
                                if not gameState.hasWall(x, y):
                                    tem[(x, y)] = 1.0
            possible_Position_List[i] = tem

    def seperate_index(self, gameState):  # find the index of each agent
        if self.index == self.getTeam(gameState)[0]:
            other_index = self.getTeam(gameState)[1]
        else:
            other_index = self.getTeam(gameState)[0]
        return other_index

    def choose_strategy(self, gameState):  # choose the strategy
        enemy = [gameState.getAgentState(i)
                 for i in self.getOpponents(gameState)]
        self.enemy_pacmen = [i for i in enemy if i.isPacman]
        other_index = self.seperate_index(gameState)
        self.border.sort(key=self.getDistance)
        my_carry = gameState.getAgentState(
            self.index).numCarrying + gameState.getAgentState(other_index).numCarrying
        enemy_carry = sum([gameState.getAgentState(
            i).numCarrying for i in self.getOpponents(gameState)])
        if self.mypos in self.border:
            self.tocenter = False
            self.toborder = False
        if (not self.toborder):
            strategy[self.index] = 1
        if len(self.enemy_pacmen) > 0:
            if not gameState.getAgentState(self.index).isPacman and not strategy[other_index] == 2:
                strategy[self.index] = 2
            elif gameState.getAgentState(self.index).scaredTimer == 0 and self.getScore(
                    gameState) + my_carry > enemy_carry:
                if gameState.getAgentState(other_index).isPacman and gameState.getAgentState(self.index).isPacman:
                    if gameState.getAgentState(self.index).numCarrying >= gameState.getAgentState(
                            other_index).numCarrying:
                        strategy[self.index] = 3
                        self.toborder = True
        elif not strategy[self.index] == 1:
            strategy[self.index] = 3
            self.toborder = True

        if self.mypos == self.start:
            strategy[self.index] = 3
            self.toborder = True

        if (len(self.getFood(gameState).asList())) <= 2:
            if gameState.getAgentState(self.index).isPacman:
                strategy[self.index] = 3
                self.toborder = True
            else:
                strategy[self.index] = 2

    # assign the inital rewards for each position
    def get_position_pro(self, gameState):

        pos_x = []
        null_position = []
        if not gameState.getAgentState(self.index).isPacman:
            if self.red:
                for i in range(0, self.border_x + 7):
                    pos_x.append(i)
            else:
                for i in range(self.border_x - 6, self.width):
                    pos_x.append(i)
        else:
            if self.red:
                for i in range(self.border_x, self.width):
                    pos_x.append(i)
            else:
                for i in range(0, self.border_x + 1):
                    pos_x.append(i)
        for i in pos_x:
            for j in range(0, gameState.data.layout.height):
                if not gameState.hasWall(i, j):
                    estimateVaule = self.evaluate(gameState, (i, j))
                    self.reward[(i, j)] = estimateVaule
                    if estimateVaule == 0:
                        null_position.append((i, j))
        self.updatereward(gameState, null_position)
        self.guessposition(gameState)
        self.choose_enemy_possible_position(gameState)
        self.updateguess(gameState)

    def updatereward(self, gameState, null_position):  # update reawrds for each position

        tem = self.reward.copy()
        for i in range(self.repeat):
            for j in null_position:
                if not gameState.hasWall(j[0], j[1]):
                    _, tem[j], _ = self.getdiscountrewards(gameState, j)
            self.reward = tem.copy()

    # find the enemy position with max probablity position
    def choose_enemy_possible_position(self, gameState):
        for i in self.getOpponents(gameState):
            possible_Position_List[i].normalize()
            self.possiblePosition[i] = possible_Position_List[i].argMax()

    def aStarSearch(self, gameState, goalPos):
        Queue = util.PriorityQueue()
        pos = gameState.getAgentPosition(self.index)
        Queue.push(pos, 0)
        actionsmap = {pos: (None, None, 0)}
        if pos == goalPos:
            return 'Stop'
        while Queue.isEmpty() == False:
            test_action = []
            now_Pos = Queue.pop()
            if now_Pos == goalPos:
                break
            _, _, testout = self.getdiscountrewards(gameState, now_Pos)
            for i, v in testout.items():
                test_action.append(i)
            for action in test_action:
                nowCost = actionsmap[now_Pos][2] + 1
                aStarCost = nowCost + self.getMazeDistance(now_Pos, goalPos)
                nextPos = testout[action]
                if nextPos in actionsmap and aStarCost < actionsmap[nextPos][2] + self.getMazeDistance(now_Pos,
                                                                                                       goalPos):
                    actionsmap[nextPos] = (action, now_Pos, nowCost)
                    Queue.update(nextPos, aStarCost)
                elif nextPos not in actionsmap:
                    actionsmap[nextPos] = (action, now_Pos, nowCost)
                    Queue.push(nextPos, aStarCost)
        path = self.findpath(actionsmap, goalPos)
        return path

    def findpath(self, actions, goalPos):
        path = []
        while actions[goalPos][0] is not None:
            path.append(actions[goalPos][0])
            actions[goalPos] = actions[actions[goalPos][1]]
        return path[-1]