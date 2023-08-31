import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from joblib import dump, load
import pickle

data=pd.read_csv('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\test.csv',header=None)


def nonMealClassificationMatrix(non_meal_data):
    index_to_remove_non_meal=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    non_meal_data_cleaned=non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    non_meal_data_cleaned=non_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=non_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    non_meal_data_cleaned=non_meal_data_cleaned.drop(non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    nmf_matrix=pd.DataFrame()
    q1=[]
    q2=[]    
    for i in range(len(non_meal_data_cleaned)):
        array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        q1.append(sorted_array[-3])
        q2.append(sorted_array[-4])
    nmf_matrix['power_second_max']=q1
    nmf_matrix['power_third_max']=q2
    twod_value=[]
    std_var=[]
    for i in range(len(non_meal_data_cleaned)):
        twod_value.append(np.diff(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
        std_var.append(np.std(non_meal_data_cleaned.iloc[i]))
    nmf_matrix['2ndDifferential']=twod_value
    nmf_matrix['standard_deviation']=std_var
    return nmf_matrix




dataset=nonMealClassificationMatrix(data)


from joblib import dump, load
with open('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\Classifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(dataset)   
    pd.DataFrame(predict).to_csv('Result.csv',index=False,header=False)
 
    pre_trained.close()
