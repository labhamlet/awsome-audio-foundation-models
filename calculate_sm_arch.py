import os 
import json 
import pandas as pd
import numpy as np


def sm(baseline, models):
  sota = np.max(np.concatenate([baseline, models]), axis = 0)
  s = (models - baseline) / ((sota - baseline) + 0.0001)
  s[s < 0] = 0
  return np.round(100 * s.mean(axis = 1),1)

scores = """HEAR-Naive N/A N/A N/A 13.0 36.0 2.2 22.0 39.0 9.9 19.9 35.2 22.6 45.7 5.4 18.4
Wav2Vec2.0 B LS 960 45.7 55.5 19.4 31.5 50.5 37.6 35.1 66.1 55.3 86.4 14.4 31.8
WavLM B LS 960 49.9 61.8 17.6 36.3 48.7 34.9 32.6 54.2 67.9 99.5 31.0 43.1
HuBERT B LS 960 58.9 67.3 24.5 40.5 54.6 38.8 36.7 58.5 65.3 99.6 33.8 40.5
Data2Vec B LS 960 23.6 45.6 10.1 30.2 40.6 27.6 25.9 50.7 48.0 99.1 43.6 27.3
WavJEPA B LS 960 64.9 71.1 33.5 45.2 60.5 46.7 47.6 74.9 71.2 99.7 39.6 47.6
Wav2Vec2.0 L LV 60k 13.1 42.7 5.8 22.0 41.7 21.0 19.9 50.2 11.6 45.7 7.3 19.3
WavLM L Mix+ 94k 67.2 70.9 32.2 42.5 61.1 41.3 42.5 68.0 71.8 99.8 42.3 45.3
HuBERT L LV 60k 64.0 70.0 29.5 41.0 54.8 38.4 36.8 64.1 72.6 99.9 45.3 43.8
Data2Vec L LS 960 25.4 49.2 10.8 30.6 43.5 28.5 27.1 44.2 45.1 99.2 28.6 23.
Wav2Vec2.0 B AS 4.3k 52.6 70.5 21.3 31.3 59.5 37.9 35.9 64.6 45.9 88.1 11.0 30.8
HuBERT B AS 4.3k 68.8 79.1 31.1 40.1 65.9 43.4 47.7 67.8 63.5 98.8 20.5 33.4
WavJEPA B AS 4.7k 86.0 83.6 49.3 52.3 67.9 46.7 58.6 84.3 72.9 99.7 24.9 44.0
Wav2Vec2.0 L AS 4.3k 74.4 79.0 37.6 39.7 66.6 44.5 49.9 76.9 59.5 99.4 17.7 38.2
HuBERT L AS 4.3k 71.5 75.6 37.4 44.3 67.5 43.4 50.5 77.8 73.3 99.6 20.5 38.6"""


model_performances = {}
for i in scores.split("\n"):
  name, model_size, data, scores_  = i.split(" ")[0], i.split(" ")[1], i.split(" ")[2], i.split(" ")[4:]
  print(scores_)
  model_performances[(name, model_size, data)] = list(map(lambda x: float(x),scores_))

baseline = model_performances[("HEAR-Naive", "N/A", "N/A")]

perf = np.zeros([len(model_performances), len(baseline)])
for i, model_name in enumerate(model_performances):
  for j in range(len(model_performances[model_name])):
    perf[i,j] = model_performances[model_name][j]


print(perf)
print(sm([baseline], perf[:, :]))
print(np.round(perf.mean(axis = 1), 1))