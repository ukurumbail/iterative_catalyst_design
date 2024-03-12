import numpy as np
from src.data import make_dataset
from src.models import knn

#General Parameters
SEED = 1 #Random Seed
LIST_OF_EXPTS = "Runs to Analyze.txt"
OUTPUT_FILE = "2024-03-01_002-to-011_no-Mn-Zn"

#KNN Parameters
N_NEIGHBORS = 5
MINKOWSKI_DISTANCE = 2

make_dataset(





