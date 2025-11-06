# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """
    factors = list(factors)
    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    # Handle edge case: empty list of factors
    if not factors:
        return None

    # Get the variable domains dictionary from any factor (they're all the same)
    variableDomainsDict = factors[0].variableDomainsDict()
    
    # Calculate the set of unconditioned variables for the new factor
    # (union of all unconditioned variables from all factors)
    unconditionedVars = set()
    for factor in factors:
        unconditionedVars.update(factor.unconditionedVariables())
    
    # Calculate the set of conditioned variables for the new factor
    # (variables that are conditioned in at least one factor and 
    # don't appear as unconditioned in any factor)
    conditionedVars = set()
    for factor in factors:
        for var in factor.conditionedVariables():
            if var not in unconditionedVars:
                conditionedVars.add(var)
    
    # Create a new factor with the calculated variables
    joinedFactor = Factor(list(unconditionedVars), list(conditionedVars), variableDomainsDict)
    
    # For each possible assignment to the new factor's variables
    for assignment in joinedFactor.getAllPossibleAssignmentDicts():
        # Calculate the product of probabilities from all input factors
        probability = 1.0
        for factor in factors:
            probability *= factor.getProbability(assignment)
        
        # Set the probability in the new factor
        joinedFactor.setProbability(assignment, probability)
    
    return joinedFactor

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        # Get the variable domains dictionary
        variableDomainsDict = factor.variableDomainsDict()
        
        # Create new sets of unconditioned and conditioned variables
        # Remove the elimination variable from unconditioned variables
        newUnconditionedVars = [var for var in factor.unconditionedVariables() 
                            if var != eliminationVariable]
        
        # Conditioned variables stay the same
        conditionedVars = factor.conditionedVariables()
        
        # Create a new factor with the updated variables
        newFactor = Factor(newUnconditionedVars, conditionedVars, variableDomainsDict)
        
        # Get all possible assignments for the new factor
        for newAssignment in newFactor.getAllPossibleAssignmentDicts():
            # Sum over all values of the elimination variable
            eliminationValues = variableDomainsDict[eliminationVariable]
            
            # Calculate the sum of probabilities
            summedProbability = 0.0
            for elimValue in eliminationValues:
                # Create a full assignment including the elimination variable
                fullAssignment = newAssignment.copy()
                fullAssignment[eliminationVariable] = elimValue
                
                # Add the probability for this value of the elimination variable
                summedProbability += factor.getProbability(fullAssignment)
            
            # Set the summed probability in the new factor
            newFactor.setProbability(newAssignment, summedProbability)
        
        return newFactor

    return eliminate

eliminate = eliminateWithCallTracking()

