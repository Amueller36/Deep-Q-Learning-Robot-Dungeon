from pydantic import BaseModel

from Config import Config


class CreateGameRequest(BaseModel):
    max_players: int = Config.NUM_PLAYERS
    max_rounds: int = Config.NUM_ROUNDS
    map_size: int = Config.MAP_SIZE