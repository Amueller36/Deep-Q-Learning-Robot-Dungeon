from pydantic import BaseModel


class CreateGameResponse(BaseModel):
    game_id: str