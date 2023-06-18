import asyncio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skillsnetwork
from sklearn.datasets import fetch_california_housing

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv'

async def d():
    await skillsnetwork.download_dataset(URL)

asyncio.run(d())