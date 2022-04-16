# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                first = 'DummyAgent', second = 'DummyAgent'):
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

class BaseQAgent(CaptureAgent):
    # Basic QLearning agent

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.training_iters = 0
        self.episodes = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2
        self.food_prio = 2
    
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.last_action = None
        CaptureAgent.registerInitialState(self, gameState)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        pass

    def getReward(self, gameState):
        pass

    def getFeats(self, gameState, action):
        succ = self.getSuccessor(gameState, action)
        feats = util.Counter()
        feats['score'] = self.getScore(succ)
        if not self.red:
            feats['score'] *= -1
        feats['choices'] = len(succ.getLegalActions(self.index))
        return feats

    def getSuccessor(self, gameState, action):
        succ = gameState.generateSuccessor(self.index, action)
        pos = succ.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return succ.generateSuccessor(self.index, action)
        else:
            return succ

    def chooseAction(self, gameState):
        self.observed.append(gameState)
        
        actions = gameState.getLegalActions(self.index)
        action = None
        if len(actions) > 0:
            if util.flipCoin(self.epsilon) and self.training():
                action = random.choice(actions)
            else:
                action = self.computeActionFromQValues(gameState)
        
        self.last_action = action
        return action

    def computeActionFromQValues(self, gameState):
        pass

    def computeValueFromQValues(self, gameState):
        pass

    def observationFunction(self, gameState):
        pass

    def update(self, state, action, nextState, reward):
        difference = (reward + self.discount * self.computeValueFromQValues(nextState))
        difference -= self.getQValue(state, action)
        new_weights = self.weights.copy()
        features = self.getFeats(state, action)
        for feature in features:
            new_weight = new_weights[feature] + self.alpha * difference * features[feature]
        new_weights[feature] = new_weight
        self.weights = new_weights.copy()

    def finish(self, state):
        CaptureAgent.final(self, state)
        if self.isTraining():
            print "END WEIGHTS"
            print self.weights
        self.episodes += 1
        if self.episodes == self.num_training:
            print "FINISHED TRAINING"
    