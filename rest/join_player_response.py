from pydantic import BaseModel


class JoinPlayerResponse(BaseModel):
    player_name: str
    game_id: str
    money: int