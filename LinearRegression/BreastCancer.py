import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

with open("LinearRegression\Breast_cancer_dataset.csv", 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)