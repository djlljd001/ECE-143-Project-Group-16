import numpy as np
import pandas as pd
import math
#import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys


def clean():
	#data here

	print("Read data...")
	sc=pd.read_csv('Data/hsls_school_v1_0.csv').iloc[:,0:476]
	st=pd.read_csv('Data/hsls_student_v1_0.csv').iloc[:,0:3579]
	#st=st.drop(columns=['X1RACE','X1HISPANIC','X1BLACK'])

	print("Clean data...")
	#remove -5 values, private data
	sc=sc.replace(-5,np.nan)
	st=st.replace(-5,np.nan)
	sc=sc.dropna(axis=1)
	st=st.dropna(axis=1)

	#Remove invalid datat
	sc=sc.replace([-9,-8,-7],np.nan)
	st=st.replace([-9,-8,-7],np.nan)
	#Replace NA with minimum value in that column
	sc=sc.fillna(sc.min(axis=0))
	st=st.fillna(st.min(axis=0))
	print('Finish Cleaning')

	return [sc, st]