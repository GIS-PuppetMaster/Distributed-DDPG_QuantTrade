from jqdatasdk import *
from datetime import *
import pandas as pd
import numpy as np
import json
import glo

auth("13074581737", "trustno1")


def download_data(env):
    raw = get_price(env.stock_code, start_date=datetime(2007, 1, 1, 0, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frequency="1m", skip_paused=True)
    raw.to_csv('Data/' + env.stock_code.replace(".", "_") + ".csv")


"""
j = np.array(raw)[..., [0, 1, 2, 3]].transpose().tolist()
with open('Data/'+glo.stock_code.replace(".", "_") + ".json", "w") as f:
    json.dump(j, f)
print("完成")
raw = get_price(glo.stock_code, start_date=datetime(2007, 1, 1, 0, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                end_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frequency="1d", skip_paused=True)
raw.to_csv('Data/'+glo.stock_code.replace(".", "_") + "_day.csv")
"""
