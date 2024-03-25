from pydantic import BaseModel


class JoinPlayerRequest(BaseModel):
    player_name: str