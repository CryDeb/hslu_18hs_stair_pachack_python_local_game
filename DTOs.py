import json

def obj_dict(obj):
    return obj.__dict__

class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

    LEFT = {NORTH: WEST,
            SOUTH: EAST,
            EAST: NORTH,
            WEST: SOUTH,
            STOP: STOP}

    RIGHT = dict([(y, x) for x, y in LEFT.items()])

    REVERSE = {NORTH: SOUTH,
               SOUTH: NORTH,
               EAST: WEST,
               WEST: EAST,
               STOP: STOP}

class PublicGameState:
    def __init__(self, gameState=None, jsonString=None):
        self.gameField = [[]]
        self.publicPlayers = []
        if jsonString != None:
            self._create_self_from_json(jsonString)
        else:
            layout = gameState.data.layout
            height, width = layout.height, layout.width
            self.gameField = [[PublicFields.WALL if layout.walls[x][y] else PublicFields.EMPTY for x in range(width)]
                              for y in range(height)]
            for agent in gameState.data.agentStates[:]:
                self.publicPlayers.append(PublicPlayer(isPacman=agent.isPacman,
                                                       direction=agent.getDirection(),
                                                       position=agent.getPosition()))

    def _create_self_from_json(self, jsonString):
        loadedJsonString = json.loads(jsonString)
        for key, value in loadedJsonString.items():
            if key == "publicPlayers":
                self.publicPlayers = self._instance_players_out_of_json_string(value)
            if key == "gameField":
                self.gameField = self._instance_game_field_out_of_json_string(value)

    @staticmethod
    def _instance_game_field_out_of_json_string(jsonString):
        myGameField = [[]]
        for fieldRowElements in jsonString:
            myGameField.append(fieldRowElements)
        return myGameField

    @staticmethod
    def _instance_players_out_of_json_string(jsonString):
        myPublicPlayers = []
        for publicPlayersJsonString in jsonString:
            myPublicPlayers.append(PublicPlayer(jsonString=publicPlayersJsonString))
        return myPublicPlayers

    def addTeamInfoToField(self, gameState, teamIndices):
        print("Agent Indices")
        for teamRedId in teamIndices[:]:
            teamRedPosition = gameState.getAgentPosition(teamRedId)
            teamRedAgentStatus = gameState.getAgentState(teamRedId)
            self.gameField[teamRedPosition[0]][teamRedPosition[1]] = PublicPlayer(teamRed=True,
                                                                                  isPacman=teamRedAgentStatus.isPacman,
                                                                                  direction=teamRedAgentStatus.getDirection())

    def __str__(self):
        return json.dumps(self, default=obj_dict)


class PublicPlayer:
    def __init__(self, isPacman=True, direction=Directions.NORTH, position=[0, 0], jsonString=None):
        self.isPacman = isPacman
        self.direction = direction
        self.position = position
        if (jsonString != None):
            self.__dict__ = jsonString

    def __str__(self):
        returnVal = 'G'
        if self.direction == Directions.NORTH:
            returnVal = 'N'
        if self.direction == Directions.SOUTH:
            returnVal = 'S'
        if self.direction == Directions.WEST:
            returnVal = 'W'
        if self.direction == Directions.EAST:
            returnVal = 'E'
        if self.isPacman:
            return returnVal.lower()
        return returnVal.upper()


class PublicFields:
    EMPTY = " "
    WALL = "%"
    PLAYER = "P"

