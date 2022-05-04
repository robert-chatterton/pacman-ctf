# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions, AgentState
import game
from util import nearestPoint

#################
# Team creation #
#################
training = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent', **kwargs):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for functions to use for both agents (just removed registerInitialState and chooseAction)
  """

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  An approximate Q-learning agent for the offensive agent
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.featuresExtractor = OffensiveFeaturesExtractor(self)

    #self.weights = {'successorScore': -10, 'distanceToFood': -1}  # -0.027287497346388308, -2.2558226236802597
    self.weights = {'closest-food': -2.528, 'successorScore': -0.2489, 'bias': -8.0709,
                    '#-of-ghosts-1-step-away': -21.342, 'onDefense': -18.822, 'eats-food': 10.27,
                    'distance-capsule': 2.14, 'distance-to-friend': 10}
    #self.weights = util.Counter()
    self.discount = 0.8 # to what extent will a future reward be discounted compared to a current reward.
    self.alpha = 0.2 # to what extent will new information be weighted higher than old information.
    self.epsilon = 0.05 # to what extent will the agent explore actions versus exploit best-known actions.
    #self.numTraining = 30
    #self.episodesSoFar = 0

  def getWeights(self):
      return self.weights

  def getQValue(self, state, action):
      """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
      "*** YOUR CODE HERE ***"
      features = self.featuresExtractor.getFeatures(state, action)
      qValue = features * self.weights
      return qValue

  def computeValueFromQValues(self, state):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      "*** YOUR CODE HERE ***"
      legalActions = state.getLegalActions(self.index)

      if len(legalActions) == 0:
        return 0.0

      action = self.getPolicy(state)
      max_action = self.getQValue(state, action)

      return max_action

  def computeActionFromQValues(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
      "*** YOUR CODE HERE ***"
      legalActions = state.getLegalActions(self.index)

      if not legalActions:
        return None

      maxVal, bestAction = None, None

      for a in legalActions:
        #self.update(state, a)
        val = self.getQValue(state, a)
        if maxVal == None or val > maxVal:
            maxVal, bestAction = val, a
      return bestAction

  def chooseAction(self, state):
      """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
      # Pick Action
      legalActions = state.getLegalActions(self.index)
      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      #values = [self.evaluate(state, a) for a in legalActions]
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

      #maxValue = max(values)
      # best action before any additional features (could be more with same value)
      #bestActions = [a for a, v in zip(legalActions, values) if v == maxValue]

      foodLeft = len(self.getFood(state).asList())

      if foodLeft <= 2:
        bestDist = 9999
        for action in legalActions:
          successor = self.getSuccessor(state, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start,pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction

      action = None
      if not legalActions:
        return action

      if training:
        for action in legalActions:
          self.updateWeights(state, action)

      p = self.epsilon
      if util.flipCoin(p):
        action = random.choice(legalActions)
      else:
        action = self.getPolicy(state)

      return action

  def getPolicy(self, state):
      return self.computeActionFromQValues(state)

  def getValue(self, state):
      return self.computeValueFromQValues(state)

  def update(self, state, action, nextState, reward):
      """
        Should update your weights based on transition

        function from PA4
      """
      features = self.featuresExtractor.getFeatures(state, action)
      old = self.getQValue(state, action)
      future = self.getValue(nextState)
      diff = reward + self.discount * future - old

      for feat in features:
          self.weights[feat] += self.alpha * diff * features[feat]

  def updateWeights(self, state, action):
    nextState = self.getSuccessor(state, action)
    #reward = nextState.getScore() - state.getScore()
    reward = self.getReward(state, nextState)
    self.update(state, action, nextState, reward)

  def getReward(self, state, nextState):
    reward = 0
    agentPosition = state.getAgentPosition(self.index)

    # decrease reward if pacman is eaten by ghost
    enemies = [state.getAgentState(i)
               for i in self.getOpponents(state)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(
          agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward = -200

    return reward

  def final(self, state):
      "Called at the end of each game."
      # call the super-class final method
      CaptureAgent.final(self, state)
      print("Final weights for offensive agent:")
      print(self.weights)


class OffensiveFeaturesExtractor:
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def __init__(self, agent):
      self.agent = agent

    def closestFood(self, pos, food, walls):
      """
      closestFood -- this is similar to the function that we have
      worked on in the search project; here its all in one place
      """
      fringe = [(pos[0], pos[1], 0)]
      expanded = set()
      while fringe:
          pos_x, pos_y, dist = fringe.pop(0)
          if (pos_x, pos_y) in expanded:
              continue
          expanded.add((pos_x, pos_y))
          # if we find a food at this location then exit
          if food[pos_x][pos_y]:
              return dist
          # otherwise spread out from the location to its neighbours
          nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
          for nbr_x, nbr_y in nbrs:
              fringe.append((nbr_x, nbr_y, dist+1))
      # no food found
      return None

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.agent.getFood(state)
        walls = state.getWalls()
        enemies = [state.getAgentState(i) for i in self.agent.getOpponents(state)]
        ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]

        features = util.Counter()

        successor = self.agent.getSuccessor(state, action)
        foodList = self.agent.getFood(successor).asList()
        features["successorScore"] = -len(foodList)  # self.getScore(successor)

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(self.agent.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / \
                (walls.width * walls.height)

        # Computes whether we're on defense (1) or offense (0)
        myState = successor.getAgentState(self.agent.index)
        features['onDefense'] = 1
        if myState.isPacman:
          features['onDefense'] = 0

        capsules = self.agent.getCapsules(state)
        if len(capsules) > 0:
            minDistCapsule = min([self.agent.getMazeDistance((x,y), c) for c in capsules])
            if minDistCapsule < 6:
              features['distance-capsule'] = -minDistCapsule

        friends = [state.getAgentState(i) for i in self.agent.getTeam(state)]
        if len(friends) > 0:
          minDistFriend = min([self.agent.getMazeDistance(state.getAgentPosition(self.agent.index), f.getPosition()) for f in friends if f != self.agent])
          features['distance-to-friend'] = -minDistFriend

        features.divideAll(10.0)
        return features


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  An approximate Q-learning agent for the offensive agent
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.featuresExtractor = DefensiveFeaturesExtractor(self)

    self.weights = {'numInvaders': -1000, 'onDefense': 100,
                    'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    self.discount = 0.8 # to what extent will a future reward be discounted compared to a current reward.
    self.alpha = 0.2 # to what extent will new information be weighted higher than old information.
    self.epsilon = 0.05 # to what extent will the agent explore actions versus exploit best-known actions.
    #self.numTraining = 30
    #self.episodesSoFar = 0

  def getWeights(self):
      return self.weights

  def getQValue(self, state, action):
      """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
      "*** YOUR CODE HERE ***"
      features = self.featuresExtractor.getFeatures(state, action)
      qValue = features * self.weights
      return qValue

  def computeValueFromQValues(self, state):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      "*** YOUR CODE HERE ***"
      legalActions = state.getLegalActions(self.index)

      if len(legalActions) == 0:
        return 0.0

      action = self.getPolicy(state)
      max_action = self.getQValue(state, action)

      return max_action

  def computeActionFromQValues(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
      "*** YOUR CODE HERE ***"
      legalActions = state.getLegalActions(self.index)

      if not legalActions:
        return None

      maxVal, bestAction = None, None

      for a in legalActions:
        val = self.getQValue(state, a)
        if maxVal == None or val > maxVal:
            maxVal, bestAction = val, a
      return bestAction

  def chooseAction(self, state):
      """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
      # Pick Action
      legalActions = state.getLegalActions(self.index)
      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      #values = [self.evaluate(state, a) for a in legalActions]
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

      #maxValue = max(values)
      # best action before any additional features (could be more with same value)
      #bestActions = [a for a, v in zip(legalActions, values) if v == maxValue]

      foodLeft = len(self.getFood(state).asList())

      if foodLeft <= 2:
        bestDist = 9999
        for action in legalActions:
          successor = self.getSuccessor(state, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start,pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction

      action = None
      if not legalActions:
        return action

      if training:
        for action in legalActions:
          self.updateWeights(state, action)

      p = self.epsilon
      if util.flipCoin(p):
        action = random.choice(legalActions)
      else:
        action = self.getPolicy(state)

      return action

  def getPolicy(self, state):
      return self.computeActionFromQValues(state)

  def getValue(self, state):
      return self.computeValueFromQValues(state)

  def update(self, state, action, nextState, reward):
      """
        Should update your weights based on transition

        function from PA4
      """
      features = self.featuresExtractor.getFeatures(state, action)
      old = self.getQValue(state, action)
      future = self.getValue(nextState)
      diff = reward + self.discount * future - old

      for feat in features:
          self.weights[feat] += self.alpha * diff * features[feat]

  def updateWeights(self, state, action):
    nextState = self.getSuccessor(state, action)
    #reward = nextState.getScore() - state.getScore()
    reward = self.getReward(state, nextState)
    self.update(state, action, nextState, reward)

  def getReward(self, state, nextState):
    reward = 0
    agentPosition = state.getAgentPosition(self.index)

    # increase reward if pacman eats ghost
    enemies = [state.getAgentState(i)
               for i in self.getOpponents(state)]
    pacmans = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(pacmans) > 0:
      minDistPacman = min([self.getMazeDistance(
          agentPosition, p.getPosition()) for p in pacmans])
      if minDistPacman == 1:
        enemies = [nextState.getAgentState(i)
                   for i in self.getOpponents(nextState)]
        pacmansNext = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(pacmansNext) == len(pacmans)-1:
          reward = 100

    return reward

  def final(self, state):
      "Called at the end of each game."
      # call the super-class final method
      CaptureAgent.final(self, state)
      print("Final weights for defensive agent:")
      print(self.weights)


class DefensiveFeaturesExtractor:
    """
    Returns simple features for a basic reflex Pacman:
    - whether we're on defense of offense
    - how many invaders there are
    - how far away an invader is
    - distance to food (need to protect it)
    """

    def __init__(self, agent):
      self.agent = agent

    def getFeatures(self, state, action):
      features = util.Counter()
      successor = self.agent.getSuccessor(state, action)

      myState = successor.getAgentState(self.agent.index)
      myPos = myState.getPosition()

      foodList = self.agent.getFood(successor).asList()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman:
        features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i)
                for i in self.agent.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      # don't want it to stop
      if action == Directions.STOP:
        features['stop'] = 1

      # don't want it to reverse
      rev = Directions.REVERSE[state.getAgentState(
          self.agent.index).configuration.direction]
      if action == rev:
        features['reverse'] = 1


      return features
