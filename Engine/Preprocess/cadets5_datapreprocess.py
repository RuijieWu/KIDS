#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import os
import json
import multiprocessing as mp
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *
import threading
import networkx as nx
import math

filePath="/the/absolute/path/of/raw_log/"

import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest() 


# In[4]:


filelist = [
 'ta1-cadets-1-e5-official-2.bin.100.json',
 'ta1-cadets-1-e5-official-2.bin.100.json.1',
 'ta1-cadets-1-e5-official-2.bin.100.json.2',
 'ta1-cadets-1-e5-official-2.bin.101.json',
 'ta1-cadets-1-e5-official-2.bin.101.json.1',
 'ta1-cadets-1-e5-official-2.bin.101.json.2',
 'ta1-cadets-1-e5-official-2.bin.102.json',
 'ta1-cadets-1-e5-official-2.bin.102.json.1',
 'ta1-cadets-1-e5-official-2.bin.102.json.2',
 'ta1-cadets-1-e5-official-2.bin.103.json',
 'ta1-cadets-1-e5-official-2.bin.103.json.1',
 'ta1-cadets-1-e5-official-2.bin.103.json.2',
 'ta1-cadets-1-e5-official-2.bin.104.json',
 'ta1-cadets-1-e5-official-2.bin.104.json.1',
 'ta1-cadets-1-e5-official-2.bin.104.json.2',
 'ta1-cadets-1-e5-official-2.bin.105.json',
 'ta1-cadets-1-e5-official-2.bin.105.json.1',
 'ta1-cadets-1-e5-official-2.bin.105.json.2',
 'ta1-cadets-1-e5-official-2.bin.106.json',
 'ta1-cadets-1-e5-official-2.bin.106.json.1',
 'ta1-cadets-1-e5-official-2.bin.106.json.2',
 'ta1-cadets-1-e5-official-2.bin.107.json',
 'ta1-cadets-1-e5-official-2.bin.107.json.1',
 'ta1-cadets-1-e5-official-2.bin.107.json.2',
 'ta1-cadets-1-e5-official-2.bin.108.json',
 'ta1-cadets-1-e5-official-2.bin.108.json.1',
 'ta1-cadets-1-e5-official-2.bin.108.json.2',
 'ta1-cadets-1-e5-official-2.bin.109.json',
 'ta1-cadets-1-e5-official-2.bin.109.json.1',
 'ta1-cadets-1-e5-official-2.bin.109.json.2',
 'ta1-cadets-1-e5-official-2.bin.10.json',
 'ta1-cadets-1-e5-official-2.bin.10.json.1',
 'ta1-cadets-1-e5-official-2.bin.10.json.2',
 'ta1-cadets-1-e5-official-2.bin.110.json',
 'ta1-cadets-1-e5-official-2.bin.110.json.1',
 'ta1-cadets-1-e5-official-2.bin.110.json.2',
 'ta1-cadets-1-e5-official-2.bin.111.json',
 'ta1-cadets-1-e5-official-2.bin.111.json.1',
 'ta1-cadets-1-e5-official-2.bin.111.json.2',
 'ta1-cadets-1-e5-official-2.bin.112.json',
 'ta1-cadets-1-e5-official-2.bin.112.json.1',
 'ta1-cadets-1-e5-official-2.bin.112.json.2',
 'ta1-cadets-1-e5-official-2.bin.113.json',
 'ta1-cadets-1-e5-official-2.bin.113.json.1',
 'ta1-cadets-1-e5-official-2.bin.113.json.2',
 'ta1-cadets-1-e5-official-2.bin.114.json',
 'ta1-cadets-1-e5-official-2.bin.114.json.1',
 'ta1-cadets-1-e5-official-2.bin.114.json.2',
 'ta1-cadets-1-e5-official-2.bin.115.json',
 'ta1-cadets-1-e5-official-2.bin.115.json.1',
 'ta1-cadets-1-e5-official-2.bin.115.json.2',
 'ta1-cadets-1-e5-official-2.bin.116.json',
 'ta1-cadets-1-e5-official-2.bin.116.json.1',
 'ta1-cadets-1-e5-official-2.bin.116.json.2',
 'ta1-cadets-1-e5-official-2.bin.117.json',
 'ta1-cadets-1-e5-official-2.bin.117.json.1',
 'ta1-cadets-1-e5-official-2.bin.117.json.2',
 'ta1-cadets-1-e5-official-2.bin.118.json',
 'ta1-cadets-1-e5-official-2.bin.118.json.1',
 'ta1-cadets-1-e5-official-2.bin.118.json.2',
 'ta1-cadets-1-e5-official-2.bin.119.json',
 'ta1-cadets-1-e5-official-2.bin.119.json.1',
 'ta1-cadets-1-e5-official-2.bin.119.json.2',
 'ta1-cadets-1-e5-official-2.bin.11.json',
 'ta1-cadets-1-e5-official-2.bin.11.json.1',
 'ta1-cadets-1-e5-official-2.bin.11.json.2',
 'ta1-cadets-1-e5-official-2.bin.120.json',
 'ta1-cadets-1-e5-official-2.bin.120.json.1',
 'ta1-cadets-1-e5-official-2.bin.120.json.2',
 'ta1-cadets-1-e5-official-2.bin.121.json',
 'ta1-cadets-1-e5-official-2.bin.121.json.1',
 'ta1-cadets-1-e5-official-2.bin.12.json',
 'ta1-cadets-1-e5-official-2.bin.12.json.1',
 'ta1-cadets-1-e5-official-2.bin.12.json.2',
 'ta1-cadets-1-e5-official-2.bin.13.json',
 'ta1-cadets-1-e5-official-2.bin.13.json.1',
 'ta1-cadets-1-e5-official-2.bin.13.json.2',
 'ta1-cadets-1-e5-official-2.bin.14.json',
 'ta1-cadets-1-e5-official-2.bin.14.json.1',
 'ta1-cadets-1-e5-official-2.bin.14.json.2',
 'ta1-cadets-1-e5-official-2.bin.15.json',
 'ta1-cadets-1-e5-official-2.bin.15.json.1',
 'ta1-cadets-1-e5-official-2.bin.15.json.2',
 'ta1-cadets-1-e5-official-2.bin.16.json',
 'ta1-cadets-1-e5-official-2.bin.16.json.1',
 'ta1-cadets-1-e5-official-2.bin.16.json.2',
 'ta1-cadets-1-e5-official-2.bin.17.json',
 'ta1-cadets-1-e5-official-2.bin.17.json.1',
 'ta1-cadets-1-e5-official-2.bin.17.json.2',
 'ta1-cadets-1-e5-official-2.bin.18.json',
 'ta1-cadets-1-e5-official-2.bin.18.json.1',
 'ta1-cadets-1-e5-official-2.bin.18.json.2',
 'ta1-cadets-1-e5-official-2.bin.19.json',
 'ta1-cadets-1-e5-official-2.bin.19.json.1',
 'ta1-cadets-1-e5-official-2.bin.19.json.2',
 'ta1-cadets-1-e5-official-2.bin.1.json',
 'ta1-cadets-1-e5-official-2.bin.1.json.1',
 'ta1-cadets-1-e5-official-2.bin.1.json.2',
 'ta1-cadets-1-e5-official-2.bin.20.json',
 'ta1-cadets-1-e5-official-2.bin.20.json.1',
 'ta1-cadets-1-e5-official-2.bin.20.json.2',
 'ta1-cadets-1-e5-official-2.bin.21.json',
 'ta1-cadets-1-e5-official-2.bin.21.json.1',
 'ta1-cadets-1-e5-official-2.bin.21.json.2',
 'ta1-cadets-1-e5-official-2.bin.22.json',
 'ta1-cadets-1-e5-official-2.bin.22.json.1',
 'ta1-cadets-1-e5-official-2.bin.22.json.2',
 'ta1-cadets-1-e5-official-2.bin.23.json',
 'ta1-cadets-1-e5-official-2.bin.23.json.1',
 'ta1-cadets-1-e5-official-2.bin.23.json.2',
 'ta1-cadets-1-e5-official-2.bin.24.json',
 'ta1-cadets-1-e5-official-2.bin.24.json.1',
 'ta1-cadets-1-e5-official-2.bin.24.json.2',
 'ta1-cadets-1-e5-official-2.bin.25.json',
 'ta1-cadets-1-e5-official-2.bin.25.json.1',
 'ta1-cadets-1-e5-official-2.bin.25.json.2',
 'ta1-cadets-1-e5-official-2.bin.26.json',
 'ta1-cadets-1-e5-official-2.bin.26.json.1',
 'ta1-cadets-1-e5-official-2.bin.26.json.2',
 'ta1-cadets-1-e5-official-2.bin.27.json',
 'ta1-cadets-1-e5-official-2.bin.27.json.1',
 'ta1-cadets-1-e5-official-2.bin.27.json.2',
 'ta1-cadets-1-e5-official-2.bin.28.json',
 'ta1-cadets-1-e5-official-2.bin.28.json.1',
 'ta1-cadets-1-e5-official-2.bin.28.json.2',
 'ta1-cadets-1-e5-official-2.bin.29.json',
 'ta1-cadets-1-e5-official-2.bin.29.json.1',
 'ta1-cadets-1-e5-official-2.bin.29.json.2',
 'ta1-cadets-1-e5-official-2.bin.2.json',
 'ta1-cadets-1-e5-official-2.bin.2.json.1',
 'ta1-cadets-1-e5-official-2.bin.2.json.2',
 'ta1-cadets-1-e5-official-2.bin.30.json',
 'ta1-cadets-1-e5-official-2.bin.30.json.1',
 'ta1-cadets-1-e5-official-2.bin.30.json.2',
 'ta1-cadets-1-e5-official-2.bin.31.json',
 'ta1-cadets-1-e5-official-2.bin.31.json.1',
 'ta1-cadets-1-e5-official-2.bin.31.json.2',
 'ta1-cadets-1-e5-official-2.bin.32.json',
 'ta1-cadets-1-e5-official-2.bin.32.json.1',
 'ta1-cadets-1-e5-official-2.bin.32.json.2',
 'ta1-cadets-1-e5-official-2.bin.33.json',
 'ta1-cadets-1-e5-official-2.bin.33.json.1',
 'ta1-cadets-1-e5-official-2.bin.33.json.2',
 'ta1-cadets-1-e5-official-2.bin.34.json',
 'ta1-cadets-1-e5-official-2.bin.34.json.1',
 'ta1-cadets-1-e5-official-2.bin.34.json.2',
 'ta1-cadets-1-e5-official-2.bin.35.json',
 'ta1-cadets-1-e5-official-2.bin.35.json.1',
 'ta1-cadets-1-e5-official-2.bin.35.json.2',
 'ta1-cadets-1-e5-official-2.bin.36.json',
 'ta1-cadets-1-e5-official-2.bin.36.json.1',
 'ta1-cadets-1-e5-official-2.bin.36.json.2',
 'ta1-cadets-1-e5-official-2.bin.37.json',
 'ta1-cadets-1-e5-official-2.bin.37.json.1',
 'ta1-cadets-1-e5-official-2.bin.37.json.2',
 'ta1-cadets-1-e5-official-2.bin.38.json',
 'ta1-cadets-1-e5-official-2.bin.38.json.1',
 'ta1-cadets-1-e5-official-2.bin.38.json.2',
 'ta1-cadets-1-e5-official-2.bin.39.json',
 'ta1-cadets-1-e5-official-2.bin.39.json.1',
 'ta1-cadets-1-e5-official-2.bin.39.json.2',
 'ta1-cadets-1-e5-official-2.bin.3.json',
 'ta1-cadets-1-e5-official-2.bin.3.json.1',
 'ta1-cadets-1-e5-official-2.bin.3.json.2',
 'ta1-cadets-1-e5-official-2.bin.40.json',
 'ta1-cadets-1-e5-official-2.bin.40.json.1',
 'ta1-cadets-1-e5-official-2.bin.40.json.2',
 'ta1-cadets-1-e5-official-2.bin.41.json',
 'ta1-cadets-1-e5-official-2.bin.41.json.1',
 'ta1-cadets-1-e5-official-2.bin.41.json.2',
 'ta1-cadets-1-e5-official-2.bin.42.json',
 'ta1-cadets-1-e5-official-2.bin.42.json.1',
 'ta1-cadets-1-e5-official-2.bin.42.json.2',
 'ta1-cadets-1-e5-official-2.bin.43.json',
 'ta1-cadets-1-e5-official-2.bin.43.json.1',
 'ta1-cadets-1-e5-official-2.bin.43.json.2',
 'ta1-cadets-1-e5-official-2.bin.44.json',
 'ta1-cadets-1-e5-official-2.bin.44.json.1',
 'ta1-cadets-1-e5-official-2.bin.44.json.2',
 'ta1-cadets-1-e5-official-2.bin.45.json',
 'ta1-cadets-1-e5-official-2.bin.45.json.1',
 'ta1-cadets-1-e5-official-2.bin.45.json.2',
 'ta1-cadets-1-e5-official-2.bin.46.json',
 'ta1-cadets-1-e5-official-2.bin.46.json.1',
 'ta1-cadets-1-e5-official-2.bin.46.json.2',
 'ta1-cadets-1-e5-official-2.bin.47.json',
 'ta1-cadets-1-e5-official-2.bin.47.json.1',
 'ta1-cadets-1-e5-official-2.bin.47.json.2',
 'ta1-cadets-1-e5-official-2.bin.48.json',
 'ta1-cadets-1-e5-official-2.bin.48.json.1',
 'ta1-cadets-1-e5-official-2.bin.48.json.2',
 'ta1-cadets-1-e5-official-2.bin.49.json',
 'ta1-cadets-1-e5-official-2.bin.49.json.1',
 'ta1-cadets-1-e5-official-2.bin.49.json.2',
 'ta1-cadets-1-e5-official-2.bin.4.json',
 'ta1-cadets-1-e5-official-2.bin.4.json.1',
 'ta1-cadets-1-e5-official-2.bin.4.json.2',
 'ta1-cadets-1-e5-official-2.bin.50.json',
 'ta1-cadets-1-e5-official-2.bin.50.json.1',
 'ta1-cadets-1-e5-official-2.bin.50.json.2',
 'ta1-cadets-1-e5-official-2.bin.51.json',
 'ta1-cadets-1-e5-official-2.bin.51.json.1',
 'ta1-cadets-1-e5-official-2.bin.51.json.2',
 'ta1-cadets-1-e5-official-2.bin.52.json',
 'ta1-cadets-1-e5-official-2.bin.52.json.1',
 'ta1-cadets-1-e5-official-2.bin.52.json.2',
 'ta1-cadets-1-e5-official-2.bin.53.json',
 'ta1-cadets-1-e5-official-2.bin.53.json.1',
 'ta1-cadets-1-e5-official-2.bin.53.json.2',
 'ta1-cadets-1-e5-official-2.bin.54.json',
 'ta1-cadets-1-e5-official-2.bin.54.json.1',
 'ta1-cadets-1-e5-official-2.bin.54.json.2',
 'ta1-cadets-1-e5-official-2.bin.55.json',
 'ta1-cadets-1-e5-official-2.bin.55.json.1',
 'ta1-cadets-1-e5-official-2.bin.55.json.2',
 'ta1-cadets-1-e5-official-2.bin.56.json',
 'ta1-cadets-1-e5-official-2.bin.56.json.1',
 'ta1-cadets-1-e5-official-2.bin.56.json.2',
 'ta1-cadets-1-e5-official-2.bin.57.json',
 'ta1-cadets-1-e5-official-2.bin.57.json.1',
 'ta1-cadets-1-e5-official-2.bin.57.json.2',
 'ta1-cadets-1-e5-official-2.bin.58.json',
 'ta1-cadets-1-e5-official-2.bin.58.json.1',
 'ta1-cadets-1-e5-official-2.bin.58.json.2',
 'ta1-cadets-1-e5-official-2.bin.59.json',
 'ta1-cadets-1-e5-official-2.bin.59.json.1',
 'ta1-cadets-1-e5-official-2.bin.59.json.2',
 'ta1-cadets-1-e5-official-2.bin.5.json',
 'ta1-cadets-1-e5-official-2.bin.5.json.1',
 'ta1-cadets-1-e5-official-2.bin.5.json.2',
 'ta1-cadets-1-e5-official-2.bin.60.json',
 'ta1-cadets-1-e5-official-2.bin.60.json.1',
 'ta1-cadets-1-e5-official-2.bin.60.json.2',
 'ta1-cadets-1-e5-official-2.bin.61.json',
 'ta1-cadets-1-e5-official-2.bin.61.json.1',
 'ta1-cadets-1-e5-official-2.bin.61.json.2',
 'ta1-cadets-1-e5-official-2.bin.62.json',
 'ta1-cadets-1-e5-official-2.bin.62.json.1',
 'ta1-cadets-1-e5-official-2.bin.62.json.2',
 'ta1-cadets-1-e5-official-2.bin.63.json',
 'ta1-cadets-1-e5-official-2.bin.63.json.1',
 'ta1-cadets-1-e5-official-2.bin.63.json.2',
 'ta1-cadets-1-e5-official-2.bin.64.json',
 'ta1-cadets-1-e5-official-2.bin.64.json.1',
 'ta1-cadets-1-e5-official-2.bin.64.json.2',
 'ta1-cadets-1-e5-official-2.bin.65.json',
 'ta1-cadets-1-e5-official-2.bin.65.json.1',
 'ta1-cadets-1-e5-official-2.bin.65.json.2',
 'ta1-cadets-1-e5-official-2.bin.66.json',
 'ta1-cadets-1-e5-official-2.bin.66.json.1',
 'ta1-cadets-1-e5-official-2.bin.66.json.2',
 'ta1-cadets-1-e5-official-2.bin.67.json',
 'ta1-cadets-1-e5-official-2.bin.67.json.1',
 'ta1-cadets-1-e5-official-2.bin.67.json.2',
 'ta1-cadets-1-e5-official-2.bin.68.json',
 'ta1-cadets-1-e5-official-2.bin.68.json.1',
 'ta1-cadets-1-e5-official-2.bin.68.json.2',
 'ta1-cadets-1-e5-official-2.bin.69.json',
 'ta1-cadets-1-e5-official-2.bin.69.json.1',
 'ta1-cadets-1-e5-official-2.bin.69.json.2',
 'ta1-cadets-1-e5-official-2.bin.6.json',
 'ta1-cadets-1-e5-official-2.bin.6.json.1',
 'ta1-cadets-1-e5-official-2.bin.6.json.2',
 'ta1-cadets-1-e5-official-2.bin.70.json',
 'ta1-cadets-1-e5-official-2.bin.70.json.1',
 'ta1-cadets-1-e5-official-2.bin.70.json.2',
 'ta1-cadets-1-e5-official-2.bin.71.json',
 'ta1-cadets-1-e5-official-2.bin.71.json.1',
 'ta1-cadets-1-e5-official-2.bin.71.json.2',
 'ta1-cadets-1-e5-official-2.bin.72.json',
 'ta1-cadets-1-e5-official-2.bin.72.json.1',
 'ta1-cadets-1-e5-official-2.bin.72.json.2',
 'ta1-cadets-1-e5-official-2.bin.73.json',
 'ta1-cadets-1-e5-official-2.bin.73.json.1',
 'ta1-cadets-1-e5-official-2.bin.73.json.2',
 'ta1-cadets-1-e5-official-2.bin.74.json',
 'ta1-cadets-1-e5-official-2.bin.74.json.1',
 'ta1-cadets-1-e5-official-2.bin.74.json.2',
 'ta1-cadets-1-e5-official-2.bin.75.json',
 'ta1-cadets-1-e5-official-2.bin.75.json.1',
 'ta1-cadets-1-e5-official-2.bin.75.json.2',
 'ta1-cadets-1-e5-official-2.bin.76.json',
 'ta1-cadets-1-e5-official-2.bin.76.json.1',
 'ta1-cadets-1-e5-official-2.bin.76.json.2',
 'ta1-cadets-1-e5-official-2.bin.77.json',
 'ta1-cadets-1-e5-official-2.bin.77.json.1',
 'ta1-cadets-1-e5-official-2.bin.77.json.2',
 'ta1-cadets-1-e5-official-2.bin.78.json',
 'ta1-cadets-1-e5-official-2.bin.78.json.1',
 'ta1-cadets-1-e5-official-2.bin.78.json.2',
 'ta1-cadets-1-e5-official-2.bin.79.json',
 'ta1-cadets-1-e5-official-2.bin.79.json.1',
 'ta1-cadets-1-e5-official-2.bin.79.json.2',
 'ta1-cadets-1-e5-official-2.bin.7.json',
 'ta1-cadets-1-e5-official-2.bin.7.json.1',
 'ta1-cadets-1-e5-official-2.bin.7.json.2',
 'ta1-cadets-1-e5-official-2.bin.80.json',
 'ta1-cadets-1-e5-official-2.bin.80.json.1',
 'ta1-cadets-1-e5-official-2.bin.80.json.2',
 'ta1-cadets-1-e5-official-2.bin.81.json',
 'ta1-cadets-1-e5-official-2.bin.81.json.1',
 'ta1-cadets-1-e5-official-2.bin.81.json.2',
 'ta1-cadets-1-e5-official-2.bin.82.json',
 'ta1-cadets-1-e5-official-2.bin.82.json.1',
 'ta1-cadets-1-e5-official-2.bin.82.json.2',
 'ta1-cadets-1-e5-official-2.bin.83.json',
 'ta1-cadets-1-e5-official-2.bin.83.json.1',
 'ta1-cadets-1-e5-official-2.bin.83.json.2',
 'ta1-cadets-1-e5-official-2.bin.84.json',
 'ta1-cadets-1-e5-official-2.bin.84.json.1',
 'ta1-cadets-1-e5-official-2.bin.84.json.2',
 'ta1-cadets-1-e5-official-2.bin.85.json',
 'ta1-cadets-1-e5-official-2.bin.85.json.1',
 'ta1-cadets-1-e5-official-2.bin.85.json.2',
 'ta1-cadets-1-e5-official-2.bin.86.json',
 'ta1-cadets-1-e5-official-2.bin.86.json.1',
 'ta1-cadets-1-e5-official-2.bin.86.json.2',
 'ta1-cadets-1-e5-official-2.bin.87.json',
 'ta1-cadets-1-e5-official-2.bin.87.json.1',
 'ta1-cadets-1-e5-official-2.bin.87.json.2',
 'ta1-cadets-1-e5-official-2.bin.88.json',
 'ta1-cadets-1-e5-official-2.bin.88.json.1',
 'ta1-cadets-1-e5-official-2.bin.88.json.2',
 'ta1-cadets-1-e5-official-2.bin.89.json',
 'ta1-cadets-1-e5-official-2.bin.89.json.1',
 'ta1-cadets-1-e5-official-2.bin.89.json.2',
 'ta1-cadets-1-e5-official-2.bin.8.json',
 'ta1-cadets-1-e5-official-2.bin.8.json.1',
 'ta1-cadets-1-e5-official-2.bin.8.json.2',
 'ta1-cadets-1-e5-official-2.bin.90.json',
 'ta1-cadets-1-e5-official-2.bin.90.json.1',
 'ta1-cadets-1-e5-official-2.bin.90.json.2',
 'ta1-cadets-1-e5-official-2.bin.91.json',
 'ta1-cadets-1-e5-official-2.bin.91.json.1',
 'ta1-cadets-1-e5-official-2.bin.91.json.2',
 'ta1-cadets-1-e5-official-2.bin.92.json',
 'ta1-cadets-1-e5-official-2.bin.92.json.1',
 'ta1-cadets-1-e5-official-2.bin.92.json.2',
 'ta1-cadets-1-e5-official-2.bin.93.json',
 'ta1-cadets-1-e5-official-2.bin.93.json.1',
 'ta1-cadets-1-e5-official-2.bin.93.json.2',
 'ta1-cadets-1-e5-official-2.bin.94.json',
 'ta1-cadets-1-e5-official-2.bin.94.json.1',
 'ta1-cadets-1-e5-official-2.bin.94.json.2',
 'ta1-cadets-1-e5-official-2.bin.95.json',
 'ta1-cadets-1-e5-official-2.bin.95.json.1',
 'ta1-cadets-1-e5-official-2.bin.95.json.2',
 'ta1-cadets-1-e5-official-2.bin.96.json',
 'ta1-cadets-1-e5-official-2.bin.96.json.1',
 'ta1-cadets-1-e5-official-2.bin.96.json.2',
 'ta1-cadets-1-e5-official-2.bin.97.json',
 'ta1-cadets-1-e5-official-2.bin.97.json.1',
 'ta1-cadets-1-e5-official-2.bin.97.json.2',
 'ta1-cadets-1-e5-official-2.bin.98.json',
 'ta1-cadets-1-e5-official-2.bin.98.json.1',
 'ta1-cadets-1-e5-official-2.bin.98.json.2',
 'ta1-cadets-1-e5-official-2.bin.99.json',
 'ta1-cadets-1-e5-official-2.bin.99.json.1',
 'ta1-cadets-1-e5-official-2.bin.99.json.2',
 'ta1-cadets-1-e5-official-2.bin.9.json',
 'ta1-cadets-1-e5-official-2.bin.9.json.1',
 'ta1-cadets-1-e5-official-2.bin.9.json.2',
 'ta1-cadets-1-e5-official-2.bin.json',
 'ta1-cadets-1-e5-official-2.bin.json.1',
 'ta1-cadets-1-e5-official-2.bin.json.2'
]


# In[5]:


from datetime import datetime, timezone
import time
import pytz
from time import mktime
from datetime import datetime
import time
def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s

def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)




# In[ ]:





# # Database setting (Make sure the database and tables are created)

# In[6]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'tc_e5_cadets_dataset_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )

cur = connect.cursor()


# In[ ]:





# In[10]:


include_edge_type=[
    'EVENT_CLOSE',
    'EVENT_OPEN',
    'EVENT_READ',
    'EVENT_WRITE',
     'EVENT_EXECUTE',
    'EVENT_RECVFROM',
    'EVENT_RECVMSG',
    'EVENT_SENDMSG',
    'EVENT_SENDTO',
]


# ## Netflow

# In[7]:


netobjset=set()
netobj2hash={}# 
datalist=[]
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
#                 pass
                if "avro.cdm20.NetFlowObject" in line:
#                     print(line)
                    try:
                        res=re.findall('NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}',line)[0]

                        nodeid=res[0]
                        srcaddr=res[2]
                        srcport=res[3]
                        dstaddr=res[4]
                        dstport=res[5]

                        nodeproperty=srcaddr+","+srcport+","+dstaddr+","+dstport 
#                         nodeproperty=dstaddr+","+dstport # 
                        hashstr=stringtomd5(nodeproperty)
                        netobj2hash[nodeid]=[hashstr,nodeproperty]
                        netobj2hash[hashstr]=nodeid
                        netobjset.add(hashstr)
                    except:
                        pass
#                     print(match)


# In[8]:


datalist=[]
for i in netobj2hash.keys():
    if len(i)!=64:
        datalist.append([i]+[netobj2hash[i][0]]+netobj2hash[i][1].split(","))

#write to database


sql = '''insert into netflow_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit() 

del netobj2hash
del datalist


# In[ ]:





# # Extracting UUIDs of Process and File

# In[56]:


subject_uuid2path={}# 
file_uuid2path={}# 

for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
#             for line in tqdm(f): 
            for line in (f):
                if "schema.avro.cdm20.Subject" in line:                
                    pattern='{"com.bbn.tc.schema.avro.cdm20.Subject":{"uuid":"(.*?)"'
                    match_ans=re.findall(pattern,line)[0]           
                    subject_uuid2path[match_ans]='none'
                elif "schema.avro.cdm20.FileObject" in line:   
                    pattern='{"com.bbn.tc.schema.avro.cdm20.FileObject":{"uuid":"(.*?)"'
                    match_ans=re.findall(pattern,line)[0]           
                    file_uuid2path[match_ans]='none'
#                     print(line)
#                     subject_uuid2path


# In[ ]:





# ## Process 

# In[57]:


scusess_count=0
fail_count=0

for file in tqdm(filelist):
    with open(filePath + file, "r") as f:
#             for line in tqdm(f): 
        for line in (f):
            if "schema.avro.cdm20.Event" in line:
#                     print(line)
                relation_type=re.findall('"type":"(.*?)"',line)[0]
                if relation_type in include_edge_type:
                     # 0: subject uuid  1:object uuid  2 object path name   -1: subject name
                    try: 
                        pattern='"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},(.*?)"exec":"(.*?)",'
                        match_ans=re.findall(pattern,line)
                        if match_ans[0][0] in subject_uuid2path:
                            subject_uuid2path[match_ans[0][0]]=match_ans[0][-1]
                    except:
                        fail_count+=1
                            


# In[58]:


len(subject_uuid2path)


# In[59]:


datalist=[]
for i in subject_uuid2path.keys():
    if subject_uuid2path[i]!='none':
        datalist.append([i]+[stringtomd5(subject_uuid2path[i]),subject_uuid2path[i]])
        


sql = '''insert into subject_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit() 


# In[60]:


fail_count


# ## File 

# In[61]:


scusess_count=0
fail_count=0

for file in tqdm(filelist):
    with open(filePath + file, "r") as f:
#             for line in tqdm(f): 
        for line in (f):
            if "schema.avro.cdm20.Event" in line:
#                     print(line)
                relation_type=re.findall('"type":"(.*?)"',line)[0]
                if relation_type in include_edge_type:                            
                    try:    
                        object_uuid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},',line)[0]
                        if object_uuid in file_uuid2path:
                            object_path=re.findall('"predicateObjectPath":{"string":"(.*?)"}',line)                            
                            if len(object_path)==0:                                
                                file_uuid2path[object_uuid]='null' 
                            else:
                                file_uuid2path[object_uuid]=object_path[0]
                    except:
                        fail_count+=1

#                                 print(line)


# In[62]:


datalist=[]
for i in file_uuid2path.keys():
    if file_uuid2path[i]!='none':
        datalist.append([i]+[stringtomd5(file_uuid2path[i]),file_uuid2path[i]])
datalist_new=[]
for i in datalist:
    if i[-1]!='null':
        datalist_new.append(i)


sql = '''insert into file_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist_new,page_size=10000)
connect.commit()  
datalist_new.clear()
        
        


# ## Processing event data

# ### extracting the node list

# In[63]:


# Generate the data of node2id table
node_list={}
##################################################################################################
sql="""
select * from file_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:    
    node_list[i[1]]=["file",i[-1]]

file_uuid2hash={}
for i in records:
    file_uuid2hash[i[0]]=i[1]
##################################################################################################    
sql="""
select * from subject_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:
    node_list[i[1]]=["subject",i[-1]]

subject_uuid2hash={}
for i in records:
    subject_uuid2hash[i[0]]=i[1]
##################################################################################################
sql="""
select * from netflow_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:
    
    node_list[i[1]]=["netflow",i[-2]+":"+i[-1]]

net_uuid2hash={}
for i in records:
    net_uuid2hash[i[0]]=i[1]


# In[64]:


node_list_database=[]
node_index=0
for i in node_list:
    node_list_database.append([i]+node_list[i]+[node_index])
    node_index+=1


# In[65]:


sql = '''insert into node2id
                     values %s
        '''
ex.execute_values(cur,sql, node_list_database,page_size=10000)
connect.commit()  


# In[66]:


# Constructing the map for nodeid to msg
sql="select * from node2id ORDER BY index_id;"
cur.execute(sql)
rows = cur.fetchall()

nodeid2msg={}  # nodeid => msg and node hash => nodeid
for i in rows:
    nodeid2msg[i[0]]=i[-1]
    nodeid2msg[i[-1]]={i[1]:i[2]} 


# In[67]:


nodeid2msg


# ### Start to process

# In[68]:


include_edge_type=[
    'EVENT_CLOSE',
    'EVENT_OPEN',
    'EVENT_READ',
    'EVENT_WRITE',
     'EVENT_EXECUTE',
    'EVENT_RECVFROM',
    'EVENT_RECVMSG',
    'EVENT_SENDMSG',
    'EVENT_SENDTO',
]


# In[69]:


def write_event_in_DB(datalist_new):
    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur,sql, datalist_new,page_size=10000)
    connect.commit() 


# In[70]:


datalist=[]
edge_type=set()
total_event_count=0
reverse=["EVENT_READ","EVENT_RECVFROM","EVENT_RECVMSG"]        
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
                    total_event_count+=1
#                     print(line)
                    subject_uuid=re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"}',line)
                    predicateObject_uuid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"}',line)
                    if len(subject_uuid) >0 and len(predicateObject_uuid)>0:
                        if subject_uuid[0] in subject_uuid2hash\
                        and (predicateObject_uuid[0] in file_uuid2hash or predicateObject_uuid[0] in net_uuid2hash):
                            relation_type=re.findall('"type":"(.*?)"',line)[0]
                            time_rec=re.findall('"timestampNanos":(.*?),',line)[0]
                            time_rec=int(time_rec)
                            subjectId=subject_uuid2hash[subject_uuid[0]]
                            if predicateObject_uuid[0] in file_uuid2hash:
                                objectId=file_uuid2hash[predicateObject_uuid[0]]
                            else:
                                objectId=net_uuid2hash[predicateObject_uuid[0]]
#                                 print(line)
                            edge_type.add(relation_type)
                            if relation_type in reverse:
                                datalist.append([objectId,nodeid2msg[objectId],relation_type,subjectId,nodeid2msg[subjectId],time_rec])
                            else:
                                datalist.append([subjectId,nodeid2msg[subjectId],relation_type,objectId,nodeid2msg[objectId],time_rec])
                            if len(datalist)==50000:
                                write_event_in_DB(datalist)
                                datalist.clear()
                                
write_event_in_DB(datalist)
datalist.clear()

                    
print("total_event_count:",total_event_count)     
#output: total_event_count: 1193669198


# In[ ]:





# In[ ]:





# In[ ]:





# # Featurization

# In[71]:


from sklearn.feature_extraction import FeatureHasher
from torch_geometric.transforms import NormalizeFeatures

from sklearn import preprocessing
import numpy as np


FH_string=FeatureHasher(n_features=16,input_type="string")
FH_dict=FeatureHasher(n_features=16,input_type="dict")


def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
#     print(l)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
#     print(l)
    return l


def subject2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
#     print(l)
    return l


def list2str(l):
    s=''
    for i in l:
        s+=i
    return s


# In[72]:


node_msg_vec=[]
node_msg_dic_list=[]
for i in tqdm(nodeid2msg.keys()):
    if type(i)==int:
        if 'netflow' in nodeid2msg[i].keys():
            higlist=['netflow']
            higlist+=ip2higlist(nodeid2msg[i]['netflow'])
            
        if 'file' in nodeid2msg[i].keys():
            higlist=['file']
            higlist+=path2higlist(nodeid2msg[i]['file'])
            
#             print(higlist)
        if 'subject' in nodeid2msg[i].keys():
            higlist=['subject']
            higlist+=subject2higlist(nodeid2msg[i]['subject'])
        node_msg_dic_list.append(list2str(higlist))


# In[73]:


node2higvec=[]
for i in tqdm(node_msg_dic_list):
    vec=FH_string.transform([i]).toarray()
    node2higvec.append(vec)


# In[74]:


node2higvec=np.array(node2higvec).reshape([-1,16])


# In[75]:


rel2id={1: 'EVENT_CLOSE',
 'EVENT_CLOSE': 1,
 2: 'EVENT_OPEN',
 'EVENT_OPEN': 2,
 3: 'EVENT_READ',
 'EVENT_READ': 3,
 4: 'EVENT_WRITE',
 'EVENT_WRITE': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 6,
 7: 'EVENT_RECVMSG',
 'EVENT_RECVMSG': 7,
 8: 'EVENT_SENDMSG',
 'EVENT_SENDMSG': 8,
 9: 'EVENT_SENDTO',
 'EVENT_SENDTO': 9}


# In[76]:


# Geneate edge type one-hot
relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)


# In[77]:


# Map different relation types to their one-hot encoding
rel2vec={}
for i in rel2id.keys():
    if type(i) is not int:
        rel2vec[i]= relvec[rel2id[i]-1]
        rel2vec[relvec[rel2id[i]-1]]=i


# In[78]:


## save the results
torch.save(node2higvec,"node2higvec")
torch.save(rel2vec,"rel2vec")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Generate the dataset

# In[ ]:


node2higvec=torch.load("./node2higvec")
rel2vec=torch.load("./rel2vec")


# In[80]:


os.system("mkdir -p ./train_graph/")
for day in tqdm(range(8,18)):
    start_timestamp=datetime_to_ns_time_US('2019-05-'+str(day)+' 00:00:00')
    end_timestamp=datetime_to_ns_time_US('2019-05-'+str(day+1)+' 00:00:00')
    sql="""
    select * from event_table
    where
          timestamp_rec>'%s' and timestamp_rec<'%s'
           ORDER BY timestamp_rec;
    """%(start_timestamp,end_timestamp)
    cur.execute(sql)
    events = cur.fetchall()
    print('2019-05-'+str(day)," events count:",str(len(events)))
    edge_list=[]
    for e in events:
        edge_temp=[int(e[1]),int(e[4]),e[2],e[5]]
        if e[2] in include_edge_type:# if this edge type is considered, include it into our graphs
#         if True:
            edge_list.append(edge_temp)
    print('2019-05-'+str(day)," edge list len:",str(len(edge_list)))

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in edge_list:
        src.append(int(i[0]))
        dst.append(int(i[1]))
    #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))
        msg.append(torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])] ))
        t.append(int(i[3]))
    if len(edge_list)>0:
        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        torch.save(dataset, "./train_graph/graph_5_"+str(day)+".TemporalData.simple")  


# In[ ]:





# In[ ]:




