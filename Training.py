import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score,f1_score



#importing Dataset
insulin_dataframe=pd.read_csv('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
cgm_dataframe=pd.read_csv('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulinpatient_dataframe=pd.read_csv('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
cgmpatient_dataframe=pd.read_csv('C:\\Users\\sshivar4\\Desktop\\Data Mining\\Project-2\\Project-2-Files\\CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])

insulin_dataframe['date_time_stamp']=pd.to_datetime(insulin_dataframe['Date'] + ' ' + insulin_dataframe['Time'])
cgm_dataframe['date_time_stamp']=pd.to_datetime(cgm_dataframe['Date'] + ' ' + cgm_dataframe['Time'])

insulinpatient_dataframe['date_time_stamp']=pd.to_datetime(insulinpatient_dataframe['Date'] + ' ' + insulinpatient_dataframe['Time'])
cgmpatient_dataframe['date_time_stamp']=pd.to_datetime(cgmpatient_dataframe['Date'] + ' ' + cgmpatient_dataframe['Time'])

#No Meal data Extraction
def nonMealClassification(insulin_dataframe,cgm_dataframe):
    inm_df=insulin_dataframe.copy()
    demo_df=inm_df.sort_values(by='date_time_stamp',ascending=True).replace(0.0,np.nan).dropna().copy()
    demo_df=demo_df.reset_index().drop(columns='index')
    list2=[]
    for j,i in enumerate(demo_df['date_time_stamp']):
        try:
            value=(demo_df['date_time_stamp'][j+1]-i).seconds//3600
            if value >=4:
                list2.append(i)
        except KeyError:
            break
    dataset=[]
    for j, i in enumerate(list2):
        var1=1
        try:
            cgm_dataset_len=len(cgm_dataframe.loc[(cgm_dataframe['date_time_stamp']>=list2[j]+pd.Timedelta(hours=2))&(cgm_dataframe['date_time_stamp']<list2[j+1])])//24
            while (var1<=cgm_dataset_len):
                if var1==1:
                    dataset.append(cgm_dataframe.loc[(cgm_dataframe['date_time_stamp']>=list2[j]+pd.Timedelta(hours=2))&(cgm_dataframe['date_time_stamp']<list2[j+1])]['Sensor Glucose (mg/dL)'][:var1*24].values.tolist())
                    var1+=1
                else:
                    dataset.append(cgm_dataframe.loc[(cgm_dataframe['date_time_stamp']>=list2[j]+pd.Timedelta(hours=2))&(cgm_dataframe['date_time_stamp']<list2[j+1])]['Sensor Glucose (mg/dL)'][(var1-1)*24:(var1)*24].values.tolist())
                    var1+=1
        except IndexError:
            break
    return pd.DataFrame(dataset)

#Meal data Extraction
def mealClassification(insulin_dataframe,cgm_dataframe,datevar):
    i_dataframe=insulin_dataframe.copy()
    i_dataframe=i_dataframe.set_index('date_time_stamp')
    ts_dataframe=i_dataframe.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    ts_dataframe['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    ts_dataframe=ts_dataframe.dropna()
    ts_dataframe=ts_dataframe.reset_index().drop(columns='index')
    list2=[]
    temp=0
    for j,i in enumerate(ts_dataframe['date_time_stamp']):
        try:
            temp=(ts_dataframe['date_time_stamp'][j+1]-i).seconds / 60.0
            if temp >= 120:
                list2.append(i)
        except KeyError:
            break
    
    dataframelist=[]
    if datevar==1:
        for j,i in enumerate(list2):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%#m/%#d/%Y')
            dataframelist.append(cgm_dataframe.loc[cgm_dataframe['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%#H:%#M:%#S'),end_time=end.strftime('%#H:%#M:%#S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(dataframelist)
    else:
        for j,i in enumerate(list2):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%Y-%m-%d')
            dataframelist.append(cgm_dataframe.loc[cgm_dataframe['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(dataframelist)



mdata_object=mealClassification(insulin_dataframe,cgm_dataframe,1)
mdata_object1=mealClassification(insulinpatient_dataframe,cgmpatient_dataframe,2)
mdata_object=mdata_object.iloc[:,0:30]
mdata_object1=mdata_object1.iloc[:,0:30]
nmdata_object=nonMealClassification(insulin_dataframe,cgm_dataframe)
nmdata_object1=nonMealClassification(insulinpatient_dataframe,cgmpatient_dataframe)


#Classifying No Meal Data 
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

#Classifying the Meal Data
def mealClassificationMatrix(meal_data):
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    meal_data_cleaned=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_data_cleaned=meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.dropna().reset_index().drop(columns='index')
    q1=[]
    q2=[]
    for i in range(len(meal_data_cleaned)):
        array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        q1.append(sorted_array[-3])
        q2.append(sorted_array[-4])
    mf_matrix=pd.DataFrame()
    mf_matrix['power_second_max']=q1
    mf_matrix['power_third_max']=q2
    tm=meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)
    maximum=meal_data_cleaned.iloc[:,5:19].idxmax(axis=1)
    twod_value=[]
    std_var=[]
    for i in range(len(meal_data_cleaned)):
        twod_value.append(np.diff(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        std_var.append(np.std(meal_data_cleaned.iloc[i]))
    mf_matrix['2ndDifferential']=twod_value
    mf_matrix['standard_deviation']=std_var
    return mf_matrix


mf_matrix=mealClassificationMatrix(mdata_object)
mf_matrix1=mealClassificationMatrix(mdata_object1)
mf_matrix=pd.concat([mf_matrix,mf_matrix1]).reset_index().drop(columns='index')
nmf_matrix=nonMealClassificationMatrix(nmdata_object)
nmf_matrix1=nonMealClassificationMatrix(nmdata_object1)
nmf_matrix=pd.concat([nmf_matrix,nmf_matrix]).reset_index().drop(columns='index')



#Training data using K-Fold Cross Validation 
mf_matrix['label']=1
nmf_matrix['label']=0
total_data=pd.concat([mf_matrix,nmf_matrix]).reset_index().drop(columns='index')
dataset=shuffle(total_data,random_state=1).reset_index().drop(columns='index')
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
principaldata=dataset.drop(columns='label')
scores_rf = []
pred_rf =[]

#Using Classifier For Training Data

model=DecisionTreeClassifier(criterion="entropy")

score_list=[]
f1_score_list=[]
precision_score_list=[]
recall_score_list=[]
for train_index, test_index in kfold.split(principaldata):
    X_train,X_test,y_train,y_test = principaldata.loc[train_index],principaldata.loc[test_index],\
    dataset.label.loc[train_index],dataset.label.loc[test_index]
    model.fit(X_train,y_train)
    scores_rf.append(model.score(X_test,y_test))
    
    pred_rf= model.predict(X_test)
    value_of_accuracy=model.score(X_test,y_test)
    f1_score_value=f1_score(y_test,pred_rf,average='weighted')
    precision_score_value = precision_score(y_test,pred_rf,average='weighted')
    recall_score_value = recall_score(y_test,pred_rf,average='weighted')
    score_list.append(value_of_accuracy)
    f1_score_list.append(f1_score_value)
    precision_score_list.append(precision_score_value)
    recall_score_list.append(recall_score_value)

X, y= principaldata, dataset['label']
model.fit(X,y)


print('Prediction Score :',np.mean(score_list)*100)
print('Precision Score:', np.mean(f1_score_list))
print('Recall Score :', np.mean(precision_score_list))
print('F1 Score :', np.mean(recall_score_list))
dump(model, 'Classifier.pickle')