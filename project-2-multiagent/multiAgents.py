# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        # Factor 1: Distance to closest food (encourage moving toward food)
        if newFoodList:
            closestFoodDist = min([manhattanDistance(newPos, food) for food in newFoodList])
            score += 20.0 / (closestFoodDist + 1)  # Higher score for being closer to food
        
        # Factor 2: Ghost positions and scared times
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[i] > 0 and ghostDist > 0:
                score += 200.0 / ghostDist
            elif ghostDist < 2:
                score -= 500
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentIndex):
            # Base cases: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            # Get number of agents
            numAgents = gameState.getNumAgents()
            
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                maxValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Next agent is ghost (agent 1)
                    value = minimax(successor, depth, 1)
                    maxValue = max(maxValue, value)
                return maxValue
            
            # Ghost's turn (minimizing player)
            else:
                minValue = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    
                    # Check if this is the last ghost
                    if agentIndex == numAgents - 1:
                        # All ghosts have moved, go back to Pacman and decrease depth
                        value = minimax(successor, depth - 1, 0)
                    else:
                        # Move to next ghost
                        value = minimax(successor, depth, agentIndex + 1)
                    
                    minValue = min(minValue, value)
                return minValue
        
        # Find the best action for Pacman
        bestAction = None
        bestValue = float('-inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start with ghost agent 1 at full depth
            value = minimax(successor, self.depth, 1)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def getMaxValue(gameState, depth, alpha, beta):
            # Base cases: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            maxValue = float('-inf')
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                # Next agent is ghost (agent 1)
                value = getMinValue(successor, depth, 1, alpha, beta)
                maxValue = max(maxValue, value)
                
                # Alpha-beta pruning
                if maxValue > beta:
                    return maxValue
                alpha = max(alpha, maxValue)
            return maxValue
        
        def getMinValue(gameState, depth, agentIndex, alpha, beta):
            # Base cases: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            # Get number of agents
            numAgents = gameState.getNumAgents()
            
            minValue = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Check if this is the last ghost
                if agentIndex == numAgents - 1:
                    # All ghosts have moved, go back to Pacman and decrease depth
                    value = getMaxValue(successor, depth - 1, alpha, beta)
                else:
                    # Move to next ghost
                    value = getMinValue(successor, depth, agentIndex + 1, alpha, beta)
                
                minValue = min(minValue, value)
                
                # Alpha-beta pruning
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            return minValue
        
        # Find the best action for Pacman
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start with ghost agent 1 at full depth
            value = getMinValue(successor, self.depth, 1, alpha, beta)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
            
            # Update alpha for root level
            alpha = max(alpha, value)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
