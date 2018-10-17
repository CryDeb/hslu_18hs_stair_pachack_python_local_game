# capture.py
# ----------
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
#
# Upgrading to python3 by Dane Wicki University of Applied Science Luzern
# Implementing of REST call by Dane Wicki University of Applied Science Luzern


# capture.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import configparser
import multiprocessing
import random
import sys

import mazeGenerator
import util
import captureGraphicsDisplay
import layout

from game import Actions
from game import Configuration
from game import Game
from game import GameStateData
from game import Grid
from util import manhattanDistance
from util import nearestPoint
from myTeam import createTeam

# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13  # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1) / 2 for i in range(SONAR_NOISE_RANGE)]
SIGHT_RANGE = 5  # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

DUMP_FOOD_ON_DEATH = True  # if we have the gameplay element that dumps dots on death

SCARED_TIME = 40


def noisyDistance(pos1, pos2):
    return int(util.manhattanDistance(pos1, pos2) + random.choice(SONAR_NOISE_VALUES))


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################


class GameState:
    """
  A GameState specifies the full game state, including the food, capsules,
  agent configurations and score changes.

  GameStates are used by the Game object to capture the actual state of the game and
  can be used by agents to reason about the game.

  Much of the information in a GameState is stored in a GameStateData object.  We
  strongly suggest that you access that data via the accessor methods below rather
  than referring to the GameStateData object directly.
  """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def getLegalActions(self, agentIndex=0):
        """
    Returns the legal actions for the agent specified.
    """
        return AgentRules.getLegalActions(self, agentIndex)

    def generateSuccessor(self, agentIndex, action):
        """
    Returns the successor state (a GameState object) after the specified agent takes the action.
    """
        # Copy current state
        state = GameState(self)

        # Find appropriate rules for the agent
        AgentRules.applyAction(state, action, agentIndex)
        AgentRules.checkDeath(state, agentIndex)
        AgentRules.decrementTimer(state.data.agentStates[agentIndex])

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
        state.data.timeleft = self.data.timeleft - 1
        return state

    def getAgentState(self, index):
        return self.data.agentStates[index]

    def getAgentPosition(self, index):
        """
    Returns a location tuple if the agent with the given index is observable;
    if the agent is unobservable, returns None.
    """
        agentState = self.data.agentStates[index]
        ret = agentState.getPosition()
        if ret:
            return tuple(int(x) for x in ret)
        return ret

    def getNumAgents(self):
        return len(self.data.agentStates)

    def getScore(self):
        """
    Returns a number corresponding to the current score.
    """
        return self.data.score

    def getRedFood(self):
        """
    Returns a matrix of food that corresponds to the food on the red team's side.
    For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
    red (meaning red is protecting it, blue is trying to eat it).
    """
        return halfGrid(self.data.food, red=True)

    def getBlueFood(self):
        """
    Returns a matrix of food that corresponds to the food on the blue team's side.
    For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
    blue (meaning blue is protecting it, red is trying to eat it).
    """
        return halfGrid(self.data.food, red=False)

    def getRedCapsules(self):
        return halfList(self.data.capsules, self.data.food, red=True)

    def getBlueCapsules(self):
        return halfList(self.data.capsules, self.data.food, red=False)

    def getWalls(self):
        """
    Just like getFood but for walls
    """
        return self.data.layout.walls

    def hasFood(self, x, y):
        """
    Returns true if the location (x,y) has food, regardless of
    whether it's blue team food or red team food.
    """
        return self.data.food[x][y]

    def hasWall(self, x, y):
        """
    Returns true if (x,y) has a wall, false otherwise.
    """
        return self.data.layout.walls[x][y]

    def isOver(self):
        return self.data._win

    def getRedTeamIndices(self):
        """
    Returns a list of agent index numbers for the agents on the red team.
    """
        return self.redTeam[:]

    def getBlueTeamIndices(self):
        """
    Returns a list of the agent index numbers for the agents on the blue team.
    """
        return self.blueTeam[:]

    def isOnRedTeam(self, agentIndex):
        """
    Returns true if the agent with the given agentIndex is on the red team.
    """
        return self.teams[agentIndex]

    def getAgentDistances(self):
        """
    Returns a noisy distance to each agent.
    """
        if 'agentDistances' in dir(self):
            return self.agentDistances
        else:
            return None

    def getDistanceProb(self, trueDistance, noisyDistance):
        "Returns the probability of a noisy distance given the true distance"
        if noisyDistance - trueDistance in SONAR_NOISE_VALUES:
            return 1.0 / SONAR_NOISE_RANGE
        else:
            return 0

    def getInitialAgentPosition(self, agentIndex):
        "Returns the initial position of an agent."
        return self.data.layout.agentPositions[agentIndex][1]

    def getCapsules(self):
        """
    Returns a list of positions (x,y) of the remaining capsules.
    """
        return self.data.capsules

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prevState=None):
        """
    Generates a new state by copying information from its predecessor.
    """
        if prevState != None:  # Initial state
            self.data = GameStateData(prevState.data)
            self.blueTeam = prevState.blueTeam
            self.redTeam = prevState.redTeam
            self.data.timeleft = prevState.data.timeleft

            self.teams = prevState.teams
            self.agentDistances = prevState.agentDistances
        else:
            self.data = GameStateData()
            self.agentDistances = []

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        state.data.timeleft = self.data.timeleft

        state.blueTeam = self.blueTeam[:]
        state.redTeam = self.redTeam[:]
        state.teams = self.teams[:]
        state.agentDistances = self.agentDistances[:]
        return state

    def makeObservation(self, index):
        state = self.deepCopy()

        # Adds the sonar signal
        pos = state.getAgentPosition(index)
        n = state.getNumAgents()
        distances = [noisyDistance(pos, state.getAgentPosition(i)) for i in range(n)]
        state.agentDistances = distances

        # Remove states of distant opponents
        if index in self.blueTeam:
            team = self.blueTeam
            otherTeam = self.redTeam
        else:
            otherTeam = self.blueTeam
            team = self.redTeam

        for enemy in otherTeam:
            seen = False
            enemyPos = state.getAgentPosition(enemy)
            for teammate in team:
                if util.manhattanDistance(enemyPos, state.getAgentPosition(teammate)) <= SIGHT_RANGE:
                    seen = True
            if not seen: state.data.agentStates[enemy].configuration = None
        return state

    def __eq__(self, other):
        """
    Allows two states to be compared.
    """
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        """
    Allows states to be keys of dictionaries.
    """
        return int(hash(self.data))

    def __str__(self):

        return str(self.data)

    def initialize(self, layout, numAgents):
        """
    Creates an initial game state from a layout array (see layout.py).
    """
        self.data.initialize(layout, numAgents)
        positions = [a.configuration for a in self.data.agentStates]
        self.blueTeam = [i for i, p in enumerate(positions) if not self.isRed(p)]
        self.redTeam = [i for i, p in enumerate(positions) if self.isRed(p)]
        self.teams = [self.isRed(p) for p in positions]
        # This is usually 60 (always 60 with random maps)
        # However, if layout map is specified otherwise, it could be less
        global TOTAL_FOOD
        TOTAL_FOOD = layout.totalFood

    def isRed(self, configOrPos):
        width = self.data.layout.width
        if type(configOrPos) == type((0, 0)):
            return configOrPos[0] < width / 2
        else:
            return configOrPos.pos[0] < width / 2


def halfGrid(grid, red):
    halfway = grid.width / 2
    halfgrid = Grid(grid.width, grid.height, False)
    if red:
        xrange = range(int(halfway))
    else:
        xrange = range(int(halfway), int(grid.width))

    for y in range(grid.height):
        for x in xrange:
            if grid[x][y]: halfgrid[x][y] = True

    return halfgrid


def halfList(l, grid, red):
    halfway = grid.width / 2
    newList = []
    for x, y in l:
        if red and x <= halfway:
            newList.append((x, y))
        elif not red and x > halfway:
            newList.append((x, y))
    return newList


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill


class CaptureRules:
    """
  These game rules manage the control flow of a game, deciding when
  and how the game starts and ends.
  """

    def __init__(self, quiet=False):
        self.quiet = quiet

    def newGame(self, layout, agents, display, length, catchExceptions):
        initState = GameState()
        initState.initialize(layout, len(agents))
        starter = random.randint(0, 1)
        print('%s team starts' % ['Red', 'Blue'][starter])
        game = Game(agents, display, self, startingIndex=starter, catchExceptions=catchExceptions)
        game.state = initState
        game.length = length
        game.state.data.timeleft = length
        if 'drawCenterLine' in dir(display):
            display.drawCenterLine()
        self._initBlueFood = initState.getBlueFood().count()
        self._initRedFood = initState.getRedFood().count()
        return game

    def process(self, state, game):
        """
    Checks to see whether it is time to end the game.
    """
        if 'moveHistory' in dir(game):
            if len(game.moveHistory) == game.length:
                state.data._win = True

        if state.isOver():
            game.gameOver = True
            if not game.rules.quiet:
                redCount = 0
                blueCount = 0
                foodToWin = (TOTAL_FOOD / 2) - MIN_FOOD
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned

                if blueCount >= foodToWin:  # state.getRedFood().count() == MIN_FOOD:
                    print('The Blue team has returned at least %d of the opponents\' dots.' % foodToWin)
                elif redCount >= foodToWin:  # state.getBlueFood().count() == MIN_FOOD:
                    print('The Red team has returned at least %d of the opponents\' dots.' % foodToWin)
                else:  # if state.getBlueFood().count() > MIN_FOOD and state.getRedFood().count() > MIN_FOOD:
                    print('Time is up.')
                    if state.data.score == 0:
                        print('Tie game!')
                    else:
                        winner = 'Red'
                        if state.data.score < 0: winner = 'Blue'
                        print('The %s team wins by %d points.' % (winner, abs(state.data.score)))

    def getProgress(self, game):
        blue = 1.0 - (game.state.getBlueFood().count() / float(self._initBlueFood))
        red = 1.0 - (game.state.getRedFood().count() / float(self._initRedFood))
        moves = len(self.moveHistory) / float(game.length)

        # return the most likely progress indicator, clamped to [0, 1]
        return min(max(0.75 * max(red, blue) + 0.25 * moves, 0.0), 1.0)

    def agentCrash(self, game, agentIndex):
        if agentIndex % 2 == 0:
            print("Red agent crashed", file=sys.stderr)
            game.state.data.score = -1
        else:
            print("Blue agent crashed", file=sys.stderr)
            game.state.data.score = 1

    def getMaxTotalTime(self, agentIndex):
        return 900  # Move limits should prevent this from ever happening

    def getMaxStartupTime(self, agentIndex):
        return 15  # 15 seconds for registerInitialState

    def getMoveWarningTime(self, agentIndex):
        return 1  # One second per move

    def getMoveTimeout(self, agentIndex):
        return 3  # Three seconds results in instant forfeit

    def getMaxTimeWarnings(self, agentIndex):
        return 2  # Third violation loses the game


class AgentRules:
    """
  These functions govern how each agent interacts with her environment.
  """

    def getLegalActions(state, agentIndex):
        """
    Returns a list of legal actions (which are both possible & allowed)
    """
        agentState = state.getAgentState(agentIndex)
        conf = agentState.configuration
        possibleActions = Actions.getPossibleActions(conf, state.data.layout.walls)
        return AgentRules.filterForAllowedActions(agentState, possibleActions)

    getLegalActions = staticmethod(getLegalActions)

    def filterForAllowedActions(agentState, possibleActions):
        return possibleActions

    filterForAllowedActions = staticmethod(filterForAllowedActions)

    def applyAction(state, action, agentIndex):
        """
    Edits the state to reflect the results of the action.
    """
        legal = AgentRules.getLegalActions(state, agentIndex)
        if action not in legal:
            action = legal[0]
            # raise Exception("Illegal action " + str(action))

        # Update Configuration
        agentState = state.data.agentStates[agentIndex]
        speed = 1.0
        # if agentState.isPacman: speed = 0.5
        vector = Actions.directionToVector(action, speed)
        oldConfig = agentState.configuration
        agentState.configuration = oldConfig.generateSuccessor(vector)

        # Eat
        next = agentState.configuration.getPosition()
        nearest = nearestPoint(next)

        if next == nearest:
            isRed = state.isOnRedTeam(agentIndex)
            # Change agent type
            agentState.isPacman = [isRed, state.isRed(agentState.configuration)].count(True) == 1
            # if he's no longer pacman, he's on his own side, so reset the num carrying timer
            # agentState.numCarrying *= int(agentState.isPacman)
            if agentState.numCarrying > 0 and not agentState.isPacman:
                score = agentState.numCarrying if isRed else -1 * agentState.numCarrying
                state.data.scoreChange += score

                agentState.numReturned += agentState.numCarrying
                agentState.numCarrying = 0

                redCount = 0
                blueCount = 0
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned
                if redCount >= (TOTAL_FOOD / 2) - MIN_FOOD or blueCount >= (TOTAL_FOOD / 2) - MIN_FOOD:
                    state.data._win = True

        if agentState.isPacman and manhattanDistance(nearest, next) <= 0.9:
            AgentRules.consume(nearest, state, state.isOnRedTeam(agentIndex))

    applyAction = staticmethod(applyAction)

    def consume(position, state, isRed):
        x, y = position
        # Eat food
        if state.data.food[x][y]:

            # blue case is the default
            teamIndicesFunc = state.getBlueTeamIndices
            score = -1
            if isRed:
                # switch if its red
                score = 1
                teamIndicesFunc = state.getRedTeamIndices

            # go increase the variable for the pacman who ate this
            agents = [state.data.agentStates[agentIndex] for agentIndex in teamIndicesFunc()]
            for agent in agents:
                if agent.getPosition() == position:
                    agent.numCarrying += 1
                    break  # the above should only be true for one agent...

            # do all the score and food grid maintainenace
            # state.data.scoreChange += score
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position
            # if (isRed and state.getBlueFood().count() == MIN_FOOD) or (not isRed and state.getRedFood().count() == MIN_FOOD):
            #  state.data._win = True

        # Eat capsule
        if isRed:
            myCapsules = state.getBlueCapsules()
        else:
            myCapsules = state.getRedCapsules()
        if (position in myCapsules):
            state.data.capsules.remove(position)
            state.data._capsuleEaten = position

            # Reset all ghosts' scared timers
            if isRed:
                otherTeam = state.getBlueTeamIndices()
            else:
                otherTeam = state.getRedTeamIndices()
            for index in otherTeam:
                state.data.agentStates[index].scaredTimer = SCARED_TIME

    consume = staticmethod(consume)

    def decrementTimer(state):
        timer = state.scaredTimer
        if timer == 1:
            state.configuration.pos = nearestPoint(state.configuration.pos)
        state.scaredTimer = max(0, timer - 1)

    decrementTimer = staticmethod(decrementTimer)

    def dumpFoodFromDeath(state, agentState, agentIndex):
        if not (DUMP_FOOD_ON_DEATH):
            # this feature is not turned on
            return

        if not agentState.isPacman:
            raise Exception('something is seriously wrong, this agent isnt a pacman!')

        # ok so agentState is this:
        if (agentState.numCarrying == 0):
            return

        # first, score changes!
        # we HACK pack that ugly bug by just determining if its red based on the first position
        # to die...
        dummyConfig = Configuration(agentState.getPosition(), 'North')
        isRed = state.isRed(dummyConfig)

        # the score increases if red eats dots, so if we are refunding points,
        # the direction should be -1 if the red agent died, which means he dies
        # on the blue side
        scoreDirection = (-1) ** (int(isRed) + 1)

        # state.data.scoreChange += scoreDirection * agentState.numCarrying

        def onRightSide(state, x, y):
            dummyConfig = Configuration((x, y), 'North')
            return state.isRed(dummyConfig) == isRed

        # we have food to dump
        # -- expand out in BFS. Check:
        #   - that it's within the limits
        #   - that it's not a wall
        #   - that no other agents are there
        #   - that no power pellets are there
        #   - that it's on the right side of the grid
        def allGood(state, x, y):
            width, height = state.data.layout.width, state.data.layout.height
            food, walls = state.data.food, state.data.layout.walls

            # bounds check
            if x >= width or y >= height or x <= 0 or y <= 0:
                return False

            if walls[x][y]:
                return False
            if food[x][y]:
                return False

            # dots need to be on the side where this agent will be a pacman :P
            if not onRightSide(state, x, y):
                return False

            if (x, y) in state.data.capsules:
                return False

            # loop through agents
            agentPoses = [state.getAgentPosition(i) for i in range(state.getNumAgents())]
            if (x, y) in agentPoses:
                return False

            return True

        numToDump = agentState.numCarrying
        state.data.food = state.data.food.copy()
        foodAdded = []

        def genSuccessors(x, y):
            DX = [-1, 0, 1]
            DY = [-1, 0, 1]
            return [(x + dx, y + dy) for dx in DX for dy in DY]

        # BFS graph search
        positionQueue = [agentState.getPosition()]
        seen = set()
        while numToDump > 0:
            if not len(positionQueue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = positionQueue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if (allGood(state, x, y)):
                state.data.food[x][y] = True
                foodAdded.append((x, y))
                numToDump -= 1

            # generate successors
            positionQueue = positionQueue + genSuccessors(x, y)

        state.data._foodAdded = foodAdded
        # now our agentState is no longer carrying food
        agentState.numCarrying = 0
        pass

    dumpFoodFromDeath = staticmethod(dumpFoodFromDeath)

    def checkDeath(state, agentIndex):
        agentState = state.data.agentStates[agentIndex]
        if state.isOnRedTeam(agentIndex):
            otherTeam = state.getBlueTeamIndices()
        else:
            otherTeam = state.getRedTeamIndices()
        if agentState.isPacman:
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if otherAgentState.isPacman: continue
                ghostPosition = otherAgentState.getPosition()
                if ghostPosition == None: continue
                if manhattanDistance(ghostPosition, agentState.getPosition()) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if otherAgentState.scaredTimer <= 0:
                        AgentRules.dumpFoodFromDeath(state, agentState, agentIndex)

                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        agentState.isPacman = False
                        agentState.configuration = agentState.start
                        agentState.scaredTimer = 0
                    else:
                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        otherAgentState.isPacman = False
                        otherAgentState.configuration = otherAgentState.start
                        otherAgentState.scaredTimer = 0
        else:  # Agent is a ghost
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if not otherAgentState.isPacman: continue
                pacPos = otherAgentState.getPosition()
                if pacPos == None: continue
                if manhattanDistance(pacPos, agentState.getPosition()) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if agentState.scaredTimer <= 0:
                        AgentRules.dumpFoodFromDeath(state, otherAgentState, agentIndex)

                        score = KILL_POINTS
                        if not state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        otherAgentState.isPacman = False
                        otherAgentState.configuration = otherAgentState.start
                        otherAgentState.scaredTimer = 0
                    else:
                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        agentState.isPacman = False
                        agentState.configuration = agentState.start
                        agentState.scaredTimer = 0

    checkDeath = staticmethod(checkDeath)

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start

    placeGhost = staticmethod(placeGhost)


#############################
# FRAMEWORK TO START A GAME #
#############################

def readCommand(argv):
    config = configparser.ConfigParser()
    config.read("settings.ini")

    args = dict()

    captureGraphicsDisplay.FRAME_TIME = 0
    args['display'] = captureGraphicsDisplay.PacmanGraphics(config.getfloat("Settings", "zoomfactor"), 0, True)

    redAddresses = config.get("RedTeam", "members").split("\n")
    blueAddresses = config.get("BlueTeam", "members").split("\n")

    redAgents = loadAgents(True, redAddresses)
    blueAgents = loadAgents(False, blueAddresses)

    args['agents'] = sum([list(el) for el in zip(redAgents, blueAgents)], [])  # list of agents

    layouts = []
    for i in range(config.getint("Settings", "numGames")):
        rand = randomLayout().split('\n')
        l = layout.Layout(rand)
        layouts.append(l)

    args['layouts'] = layouts
    args['length'] = config.getint("Settings", "maxMoves")
    args['numGames'] = config.getint("Settings", "numGames")
    args['catchExceptions'] = config.getboolean("Settings", "catchException")

    return args


def randomLayout(seed=None):
    if not seed:
        seed = random.randint(0, 99999999)
    return mazeGenerator.generateMaze(seed)


def loadAgents(isRed, addresses=[]):
    "Calls agent factories and returns lists of agents"
    numOfAgents = len(addresses)

    createTeamFunc = createTeam

    args = dict()
    args['ipAddresses'] = addresses

    indexAddend = 0
    if not isRed:
        indexAddend = 1
    indices = [2 * i + indexAddend for i in range(numOfAgents)]
    return createTeamFunc(indices, **args)

def runGames(layouts, agents, display, length, numGames, catchExceptions=False):
    rules = CaptureRules()
    games = []

    for i in range(numGames):
        layout = layouts[i]
        gameDisplay = display
        rules.quiet = False
        g = rules.newGame(layout, agents, gameDisplay, length, catchExceptions)
        g.run()
        games.append(g)

    if numGames > 1:
        scores = [game.state.data.score for game in games]
        redWinRate = [s > 0 for s in scores].count(True) / float(len(scores))
        blueWinRate = [s < 0 for s in scores].count(True) / float(len(scores))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Red Win Rate:  %d/%d (%.2f)' % ([s > 0 for s in scores].count(True), len(scores), redWinRate))
        print('Blue Win Rate: %d/%d (%.2f)' % ([s < 0 for s in scores].count(True), len(scores), blueWinRate))
        print('Record:       ', ', '.join([('Blue', 'Tie', 'Red')[max(0, min(2, 1 + s))] for s in scores]))

    return games


def save_score(game):
    with open('score', 'w') as f:
        print(game.state.data.score, end="", file=f)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for windows with pyinstaller
    options = readCommand(sys.argv[1:])  # Get game components based on input
    games = runGames(**options)
    save_score(games[0])
