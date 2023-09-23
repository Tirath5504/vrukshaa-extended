from pydantic import BaseModel


class Item(BaseModel):
    img : str
    description : str