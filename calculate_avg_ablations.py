import os 
import json 
import pandas as pd
import numpy as np


def sm(baseline, models):
  sota = np.max(np.concatenate([baseline, models]), axis = 0)
  s = (models - baseline) / ((sota - baseline) + 0.0001)
  s[s < 0] = 0
  return np.round(100 * s.mean(axis = 1),1)

target_probs = """
DCASE	FSD50k	LC	ESC-50	CD	VL	SC	NS	BO	Mri-S	Mri-T	target_prob
93	53.8	77.2 +- 2.2	84.8 +- 2.0	67.5 +- 1.1	49.7 +- 4.8	89.4	32.2	92.4 +- 5.8	97.3 +- 0.5	89.2 +- 0.9	0.15
93.3	55.2	78.0 +- 1.3	85.5 +- 3.1	70.5 +- 1.1	49.9 +- 1.8	91.9	36	92.8 +- 6.8	97.1 +- 0.3	90.4 +- 0.6	0.2
94	55.1	77.5 ± 2.3	86.8 ± 2.8	71.1 ± 0.8	53.8 ± 1.7	90.8	34.8	93.2 ± 5.3	97.0 ± 0.5	87.8 ± 1.1	0.25
92.7	54.1	73.0 +- 1.8	84.6 +- 3.2	68.8 +- 0.8	47.5 +- 3.7	90.4	34.8	89.4 +- 3.4	97.0 +- 0.7	87.6 +- 1.1	0.3
"""

context_lengths = """
DCASE	FSD50k	LC	ESC-50	CD	VL	SC	NS	BO	Mri-S	Mri-T	context_length
93.4	54.4	76.7 +- 1.1	84.1 +- 2.4	69.0 +- 0.8	54.7 +- 1.8	90.6	32.4	91.5 +- 4.5	97.4 +- 0.4	89.5 +- 0.5	5
94	55.1	77.5 ± 2.3	86.8 ± 2.8	71.1 ± 0.8	53.8 ± 1.7	90.8	34.8	93.2 ± 5.3	97.0 ± 0.5	87.8 ± 1.1	10
93.8	53.3	75.6 +- 2.7	84.6 +- 2.0	70.1 +- 1.1	50.8 +- 4.5	89.7	35.2	90.2 +- 8.1	96.9 +- 0.7	88.9 +- 0.8	15
"""

target_lengths = """
DCASE	FSD50k	LC	ESC-50	CD	VL	SC	NS	BO	Mri-S	Mri-T target_length
93.7	55.3	75.9 +- 1.8	85.2 +- 2.5	71.2 +- 1.1	53.1 +- 2.5	92.2	35.4	90.7 +- 4.9	97.4 +- 0.7	89.6 +- 0.6	5
94	55.1	77.5 ± 2.3	86.8 ± 2.8	71.1 ± 0.8	53.8 ± 1.7	90.8	34.8	93.2 ± 5.3	97.0 ± 0.5	87.8 ± 1.1	10
93	53.2	75.0 +- 1.0	81.6 +- 4.2	68.0 +- 0.4	44.0 +- 2.3	91.2	34.4	90.7 +- 8.2	97.2 +- 0.8	89.1 +- 0.8	15
"""

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