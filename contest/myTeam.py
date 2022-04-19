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
from util import nearestPoint
import logging
import uuid
from create_weights import ParameterSetter

#################
# Team creation #
#################

arguments = {}
GENERATION_LENGTH = 10
NUM_GENERATIONS = 5

# init first generation with random weights
PS = ParameterSetter()

# set up logging
logging.basicConfig(filename='genetic.log', filemode='w', level=logging.DEBUG)
logging.info('STARTING GENERATION 0')

def createTeam(firstIndex, secondIndex, isRed,
                first = 'BaseAgent', second = 'BaseAgent', **args):
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
    if 'numTraining' in args:
        arguments['numTraining'] = args['numTraining']
        TRAINING = True
    else:
        TRAINING = False

    if TRAINING:
        return [PacBoyHallMonitor(firstIndex, arguments['numTraining'], train=TRAINING), PacBoyKilla(secondIndex, arguments['numTraining'], train=TRAINING)]
    else:
        return [PacBoyHallMonitor(firstIndex, 0, train=TRAINING), PacBoyKilla(secondIndex, 0, train=TRAINING)]

##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    # base agent to go off of for offense/defense
    def __init__(self, index, training_iters, train=True, timeForComputing=0.1):
        self.index = index
        # Whether or not you're on the red team
        self.red = None
        # Agent objects controlling you and your teammates
        self.agentsOnTeam = None
        # Maze distance calculator
        self.distancer = None
        # A history of observations
        self.observationHistory = []
        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing
        # Access to the graphics
        self.display = None

        self.iters = training_iters
        self.training = train
        self.mutation_rate = 0.05
        self.gen = 0
        self.weights = {}
        self.genWeights()
        self.current_generation = []
        self.parent = None
        self.turns = 0
        
        # defense only
        self.scared = False
        self.scaredEnd = None

        # offense only
        self.carrying = False

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # board size
        self.width, self.height = gameState.getWalls().asList()[-1]
        self.mid_x = self.width / 2
        if self.red:
            self.less_than_mid = True
        else:
            self.less_than_mid = False

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.turns += 1
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

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
        weights = self.getWeights()
        # logging.info('%d value, carrying is %s' % (features * weights, str(self.carrying)))
        return features * weights    

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: 
            features['stop'] = 1
        
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    

        # Compute distance to the nearest food
        myPos = successor.getAgentState(self.index).getPosition()

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
            features['foodLeft'] = len(foodList)

        ourFood = self.getFoodYouAreDefending(successor).asList()
        if len(ourFood) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in ourFood])
            features['distanceToOurFood'] = minDistance
            features['ourFoodLeft'] = len(ourFood)

        return features

    def getWeights(self):
        return self.weights

    def genWeights(self):
        if self.training:
            self.weights = PS.get_params(self.gen, defensive=True)
        else:
            self.weights = PS.read_params('parameters_d.json')

    def final(self, gameState):       
        if not self.training:
            return
        fitness = self.getScore(gameState) + self.turns
        weights = self.getWeights()

        self.turns = 0

        self.current_generation.append((fitness, weights))   
        log_str = 'AGENT %d, ITERATION %d: FITNESS: %d WEIGHTS:\n%s' % (self.index, (GENERATION_LENGTH * NUM_GENERATIONS) - self.iters, fitness, str(weights))
        logging.info(log_str)   

        # once at end of curr generation, choose best "parents", combine, 
        # adjust parameters to a combination of those and update based on mutation factor
        if len(self.current_generation) == GENERATION_LENGTH:
            # find top parent
            self.current_generation.sort(key=lambda tup: tup[0])
            self.parent = self.current_generation[-1]
            logging.info('AGENT %d CHOSEN PARENT: %s' % (self.index, str(self.parent)))
            
            # get new weights
            new_weights = {}
            for feat, weight in self.parent[1].items():
                mut = random.random() * self.mutation_rate
                if util.flipCoin(0.5):
                    new_weights[feat] = weight + (weight * -1 * mut)
                else:
                    new_weights[feat] = weight + (weight * mut)
            self.weights = new_weights

            # update baseline parameters

            # reset current generation
            self.current_generation = []
            self.gen += 1
            logging.info('STARTING GENERATION %d' % (self.gen))

            self.parent = (self.parent[0], self.weights)

        else:
            if self.parent:
                self.weights = PS.get_params(self.gen, parent=self.parent[1])
            else:
                self.weights = PS.get_params(self.gen)
        
        self.iters -= 1

        # LATER: possibly experiment with switching the mutation factor per generation
        
        # after final generation, pick the best overall weights based on score and compete using those values
        if self.iters == 0:
            PS.set_params(self.weights, defensive=False)
            logging.info('WEIGHTS SET:\n%s' % (self.parent[1]))


# offensive agent
class PacBoyKilla(BaseAgent):

    def genWeights(self):
        if self.training:
            self.weights = PS.get_params(self.gen, defensive=False)
        else:
            self.weights = PS.read_params('parameters_a.json')
    
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.turns += 1
        actions = gameState.getLegalActions(self.index)

        # choose best valued move
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        if myState.isPacman: 
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1
            if self.carrying:
                features['returnReward'] = 1
                self.carrying = False

        if self.carrying:
            dist_to_start = self.getMazeDistance(myPos, self.start)
            features['distanceToStart'] = dist_to_start
            return features

        ally_index = self.getTeam(gameState)[0]
        ally_pos = successor.getAgentState(ally_index).getPosition()

        features['distanceToAlly'] = self.getMazeDistance(myPos, ally_pos)

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        features['numEnemies'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['enemyDistance'] = min(dists)

        if action == Directions.STOP: 
            features['stop'] = 1
        
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        successor = self.getSuccessor(gameState, action)
        currFood = self.getFood(gameState).asList()
        foodList = self.getFood(successor).asList()    

        # Compute distance to the nearest food
        nextPos = successor.getAgentState(self.index).getPosition()

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(nextPos, food) for food in foodList])
            # we ate a food with this action
            if len(currFood) == len(foodList) + 1:
                features['distanceToFood'] = 0
                self.carrying = True
            else:
                features['distanceToFood'] = minDistance
            features['foodLeft'] = len(foodList)

        ourFood = self.getFoodYouAreDefending(successor).asList()
        if len(ourFood) > 0:
            minDistance = min([self.getMazeDistance(nextPos, food) for food in ourFood])
            features['distanceToOurFood'] = minDistance
            features['ourFoodLeft'] = len(ourFood)

        return features

    # override final
    def final(self, gameState):       
        if not self.training:
            return
        fitness = self.getScore(gameState) + self.turns
        weights = self.getWeights()

        self.turns = 0
        self.carrying = False

        self.current_generation.append((fitness, weights))   
        log_str = 'KILLA ITERATION %d: FITNESS: %d WEIGHTS:\n%s' % ((GENERATION_LENGTH * NUM_GENERATIONS) - self.iters, fitness, str(weights))
        logging.info(log_str)   

        # once at end of curr generation, choose best "parents", combine, 
        # adjust parameters to a combination of those and update based on mutation factor
        if len(self.current_generation) == GENERATION_LENGTH:
            # find top parent
            self.current_generation.sort(key=lambda tup: tup[0])
            self.parent = self.current_generation[-1]
            logging.info('KILLA: CHOSEN PARENT: %s' % (str(self.parent)))
            
            # get new weights
            new_weights = {}
            for feat, weight in self.parent[1].items():
                mut = random.random() * self.mutation_rate
                if util.flipCoin(0.5):
                    new_weights[feat] = weight + (weight * -1 * mut)
                else:
                    new_weights[feat] = weight + (weight * mut)
            self.weights = new_weights

            # reset current generation
            self.current_generation = []
            self.gen += 1
            logging.info('STARTING GENERATION %d' % (self.gen))

            self.parent = (self.parent[0], self.weights)

        else:
            if self.parent:
                self.weights = PS.get_params(self.gen, defensive=False, parent=self.parent[1])
            else:
                self.weights = PS.get_params(self.gen, defensive=False)
        
        self.iters -= 1

        # LATER: possibly experiment with switching the mutation factor per generation
        
        # after final generation, pick the best overall weights based on score and compete using those values
        if self.iters == 0:
            PS.set_params(self.weights, defensive=False)
            logging.info('WEIGHTS SET:\n%s' % (self.parent[1]))
            self.training = False

# defensive pacman agent
class PacBoyHallMonitor(BaseAgent):

    def genWeights(self):
        if self.training:
            self.weights = PS.get_params(self.gen, defensive=True)
        else:
            self.weights = PS.read_params('parameters_d.json')

    # override final
    def final(self, gameState):       
        if not self.training:
            return
        fitness = self.getScore(gameState) + self.turns
        weights = self.getWeights()

        self.turns = 0

        self.current_generation.append((fitness, weights))   
        log_str = 'HALL MONITOR ITERATION %d: FITNESS: %d WEIGHTS:\n%s' % ((GENERATION_LENGTH * NUM_GENERATIONS) - self.iters, fitness, str(weights))
        logging.info(log_str)   

        # once at end of curr generation, choose best "parents", combine, 
        # adjust parameters to a combination of those and update based on mutation factor
        if len(self.current_generation) == GENERATION_LENGTH:
            # find top parent
            self.current_generation.sort(key=lambda tup: tup[0])
            self.parent = self.current_generation[-1]
            logging.info('AGENT %d CHOSEN PARENT: %s' % (self.index, str(self.parent)))
            
            # get new weights
            new_weights = {}
            for feat, weight in self.parent[1].items():
                mut = random.random() * self.mutation_rate
                if util.flipCoin(0.5):
                    new_weights[feat] = weight + (weight * -1 * mut)
                else:
                    new_weights[feat] = weight + (weight * mut)
            self.weights = new_weights

            # update baseline parameters

            # reset current generation
            self.current_generation = []
            self.gen += 1
            logging.info('STARTING GENERATION %d' % (self.gen))

            self.parent = (self.parent[0], self.weights)

        else:
            if self.parent:
                self.weights = PS.get_params(self.gen, defensive=True, parent=self.parent[1])
            else:
                self.weights = PS.get_params(self.gen, defensive=True)
        
        self.iters -= 1

        # LATER: possibly experiment with switching the mutation factor per generation
        
        # after final generation, pick the best overall weights based on score and compete using those values
        if self.iters == 0:
            PS.set_params(self.weights, defensive=True)
            logging.info('WEIGHTS SET:\n%s' % (self.parent[1]))
            self.training = False