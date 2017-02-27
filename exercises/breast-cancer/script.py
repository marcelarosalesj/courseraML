import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

data = pd.read_csv('./data.csv')
data.hist()
plt.show()
