from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from typing import List
import DetectionClass as DC
from JsonToPng import *
import uvicorn

class TomatoLocalization(BaseModel):
    left: int
    top: int
    width: int
    height: int


class StalkLocalization(BaseModel):
    left: int
    top: int
    width: int
    height: int


class Localizations(BaseModel):
    tomato: TomatoLocalization
    stalk: StalkLocalization


class Frame(BaseModel):
    c: Optional[str] = None
    d: Optional[str] = None
    tag: str


class Localization(BaseModel):
    tomatoes: List[Localizations]
    tag: str

def return_localization(network, tag):
    print('detect')
    tuple_ts = network.detection()
    width = tuple_ts[2]
    height = tuple_ts[3]
    print(f"tuple_ts = {tuple_ts}")
    localization_list = []
    for i in range(len(tuple_ts[0])):
        tomato = tuple_ts[0][i]
        t_pos = TomatoLocalization(top=tomato[0] * height, left=tomato[1] * width,
                                   height=(tomato[2] - tomato[0]) * height,
                                   width=(tomato[3] - tomato[1]) * width)
        stalk = tuple_ts[1][i]
        s_pos = StalkLocalization(top=stalk[0] * height, left=stalk[1] * width,
                                  height=(stalk[2] - stalk[0]) * height, width=(stalk[3] - stalk[1]) * width)
        localization_list.append(Localizations(tomato=t_pos, stalk=s_pos))
    # print(f"localization_list = {localization_list}")
    final_list = Localization(tomatoes=localization_list, tag=tag)
    print(f"final_list = {final_list}")
    return final_list

i = 0
k = 0

if __name__ == "__main__":
    network = DC.DetectionClass()
    app = FastAPI()
    frames = {}
    localizations = {}
    network.detection()



    @app.get("/get_frame/")
    async def get_frame():
        return frames[i-1]


    @app.post("/create_frame/")
    async def create_frame(frame: Frame):
        global i
        global k
        frames[i] = frame
        Create_Png_From_Hex(frame.c)
        localizations[k] = return_localization(network, frame.tag)
        k = k + 1
        i = i + 1
        return frame


    @app.get("/get_localization/")
    async def get_localization():
        return localizations[k-1]


    @app.post("/create_localization/")
    async def create_localization(localization: Localization):
        global k
        localizations[k] = localization
        k = k + 1
        return localization

    uvicorn.run(app, host="localhost", port=4322)
