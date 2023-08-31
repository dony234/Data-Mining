import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from scipy import signal
from scipy.stats import entropy, iqr
from scipy.fftpack import fft, ifft, rfft
import pickle
import math
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

def calMealTimeInfo(insulin_info):
    minute =[]
    val_of_insulin =[]
    level_of_insulin =[]
    Time1=[]
    Time2 =[]
    timestamp = []
    diff =[]
    val_of_col= insulin_info['BWZ Carb Input (grams)']
    max_value= val_of_col.max()
    min_value = val_of_col.min()
    CalcValues = math.ceil(max_value-min_value/60)
     
    for p in insulin_info['datetime']:
        minute.append(p)
    for p in insulin_info['BWZ Carb Input (grams)']:
        val_of_insulin.append(p)
    for p,q in enumerate(minute):
        if(p<len(minute)-1):
            diff.append((minute[p+1]-minute[p]).total_seconds()/3600)
    minute1 = minute[0:-1]
    minute2 = minute[1:]
    result=[]
    for p in val_of_insulin[0:-1]:
        result.append(0 if (p>=min_value and p<=min_value+20)
                          else 1 if (p>=min_value+21 and p<=min_value+40)
                          else 2 if(p>=min_value+41 and p<=min_value+60) 
                          else 3 if(p>=min_value+61 and p<=min_value+80)
                          else 4 if(p>=min_value+81 and p<=min_value+100) 
                          else 5 )
    list1 = list(zip(minute1, minute2, diff,result))
    for q in list1:
        if q[2]>2.5:
            timestamp.append(q[0])
            level_of_insulin.append(q[3])
        else:
            continue
    return timestamp,level_of_insulin


def getMealData(time_of_meal,timer_start,timer_end,level_of_insulin,level_of_glucose):
    new_rows = []
    for q,nt in enumerate(time_of_meal):
        index_of_meal= level_of_glucose[level_of_glucose['datetime'].between(nt+ pd.DateOffset(hours=timer_start),nt + pd.DateOffset(hours=timer_end))]
        
        if index_of_meal.shape[0]<8:
            del level_of_insulin[q]
            continue
        val_of_glucose = index_of_meal['Sensor Glucose (mg/dL)'].to_numpy()
        avg = index_of_meal['Sensor Glucose (mg/dL)'].mean()
        count = 30 - len(val_of_glucose)
        if count > 0:
            for p in range(count):
                val_of_glucose = np.append(val_of_glucose, avg)
        new_rows.append(val_of_glucose[0:30])
    return pd.DataFrame(data=new_rows),level_of_insulin

def CalcData(insulin_info,glucose_info):
    md = pd.DataFrame()
    glucose_info['Sensor Glucose (mg/dL)'] = glucose_info['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    insulin_info= insulin_info[::-1]
    glucose_info= glucose_info[::-1]
    insulin_info['datetime']= insulin_info['Date']+" "+insulin_info['Time']
    insulin_info['datetime']=pd.to_datetime(insulin_info['datetime'])
    glucose_info['datetime']= glucose_info['Date']+" "+glucose_info['Time']
    glucose_info['datetime']=pd.to_datetime(insulin_info['datetime'])
    
    new_insulin_value = insulin_info[['datetime','BWZ Carb Input (grams)']]
    new_glucose_value = glucose_info[['datetime','Sensor Glucose (mg/dL)']]

    new_insulin = new_insulin_value[(new_insulin_value['BWZ Carb Input (grams)']>0) ]
    timestamp,level_of_insulin = calMealTimeInfo(new_insulin)
    meal_info,new_level_insulin = getMealData(timestamp,-0.5,2,level_of_insulin,new_glucose_value)

    return meal_info,new_level_insulin

# def Mean_Abs_val(attr):
#     var = 0
#     for p in range(0, len(attr) - 1):
#         var = var + np.abs(attr[(p + 1)] - attr[p])
#     return var / len(attr)

def Entropy(attr):
    size = len(attr)
    entr = 0
    if size <= 1:
        return 0
    else:
        value, c = np.unique(attr, return_counts=True)
        temp = c / size
        ratio_zero = np.count_nonzero(temp)
        if ratio_zero <= 1:
            return 0
        for p in temp:
            entr -= p * np.log2(p)
        return entr   

# def RMS(attr):
#     attr = 0
#     for p in range(0, len(attr) - 1):
        
#         mean_square = mean_square + np.square(attr[p])
#     return np.sqrt(mean_square / len(attr))

# def FF(attr):
#     result = fft(attr)
#     leng = len(attr)
#     k = 2/300
#     amp_var = []
#     frweq = np.linspace(0, leng * k, leng)
#     for z in result:
#         amp_var.append(np.abs(z))
#     amp_sort = amp_var
#     amp_sort = sorted(amp_sort)
#     max_amp = amp_sort[(-2)]
#     max_freq = frweq.tolist()[amp_var.index(max_amp)]
#     return [max_amp, max_freq]


# def ZeroCrossing(row_var, xaxis_var):
#     slide = [0]
#     cross_zero_var = list()
#     cross_zero_rate = 0
#     x_var = [i for i in range(xaxis_var)][::-1]
#     y_var = row_var[::-1]
#     for p in range(0, len(x_var) - 1):
#         slide.append((y_var[(p + 1)] - y_var[p]) / (x_var[(p + 1)] - x_var[p]))

#     for p in range(0, len(slide) - 1):
#         if slide[p] * slide[(p + 1)] < 0:
#             cross_zero_var.append([slide[(p + 1)] - slide[p], x_var[(p + 1)]])

#     cross_zero_rate = np.sum([np.abs(np.sign(slide[(i + 1)]) - np.sign(slide[i])) for i in range(0, len(slide) - 1)]) / (2 * len(slide))
#     if len(cross_zero_var) > 0:
#         return [max(cross_zero_var)[0], cross_zero_rate]
#     else:
#         return [0, 0]

def calculate_fft(df):
    pf1 = []
    pf2 = []
    pf3 = []
    pf4 = []
    pf5 = []
    pf6 = []
    
    for i in range(len(df)):
        array = abs(rfft(df.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        array.sort()
        
        pf1.append(array[-2])
        pf2.append(array[-3])
        pf3.append(array[-4])
        pf4.append(array[-5])
        pf5.append(array[-6])
        pf6.append(array[-7])
    
    return pf1, pf2, pf3, pf4, pf5, pf6

def calculate_psd(df):
    ps1, ps2, ps3 = [],[],[]
    for p in range(len(df)):
        Temp = df.iloc[:,0:30].iloc[p].values.tolist()
        f, pk = signal.periodogram(Temp)
        k1 = pk[0:6].mean()
        k2 = pk[5:11].mean()
        k3 = pk[10:16].mean()
        ps1.append(k1)
        ps2.append(k2)
        ps3.append(k3)
    return ps1, ps2, ps3

def Glucose(md):

    var=md.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    meal_clean_var=md.drop(md.index[var]).reset_index().drop(columns='index')
    meal_clean_var=meal_clean_var.interpolate(method='linear',axis=1)
    index_drop_var=meal_clean_var.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_clean_var=meal_clean_var.drop(md.index[index_drop_var]).reset_index().drop(columns='index')
    meal_clean_var=meal_clean_var.dropna().reset_index().drop(columns='index')
    matrix_var=pd.DataFrame()
    
    #Entropy
    entr = []
    for p in range(len(meal_clean_var)):
        kl = meal_clean_var.iloc[:,0:30].iloc[p].values.tolist()
        entr.append(entropy(kl))
    
    matrix_var["Entropy"] = entr
    
    #Calculating iqr
    Iqr = []
    for i in range(len(meal_clean_var)):
        I = meal_clean_var.iloc[:,0:30].iloc[i].values.tolist()
        Iqr.append(iqr(I))
    
    matrix_var["IQR"] = Iqr

    #Calculating psd
    ps1, ps2, ps3 = calculate_psd(meal_clean_var)
    matrix_var["Psd_1"] = ps1
    matrix_var["Psd_2"] = ps2
    matrix_var["Psd_3"] = ps3
    
    #Calculating FFT
    pf1, pf2, pf3, pf4, pf5, pf6 = calculate_fft(meal_clean_var)
    
    matrix_var['p_f_1']=pf1
    matrix_var['p_f_2']=pf2
    matrix_var['p_f_3']=pf3
    matrix_var['p_f_4']=pf4
    matrix_var['p_f_5']=pf5
    matrix_var['p_f_6']=pf6
    
    var6 = 6
    max_var = meal_clean_var.iloc[:,7:].idxmax(axis=1)
    
    #Velocity and Acceleration
    max_diff1 = []
    min_diff1 = []
    avg_diff1 = []
    max_diff2 = []
    min_diff2 = []
    avg_diff2 = []

    for p in range(len(meal_clean_var)):
        
        max_diff1.append(np.diff(meal_clean_var.iloc[p,var6:(max_var[p]+1)].tolist()).max())
        min_diff1.append(np.diff(meal_clean_var.iloc[p,var6:(max_var[p]+1)].tolist()).min())
        avg_diff1.append(np.diff(meal_clean_var.iloc[p,var6:(max_var[p]+1)].tolist()).mean())
        
        if(len(meal_clean_var.iloc[p,var6:(max_var[p]+1)])>2):
            max_diff2.append(np.diff(np.diff(meal_clean_var.iloc[p,var6:max_var[p]+1].tolist())).max())
            min_diff2.append(np.diff(np.diff(meal_clean_var.iloc[p,var6:max_var[p]+1].tolist())).min()) 
            avg_diff2.append(np.diff(np.diff(meal_clean_var.iloc[p,var6:max_var[p]+1].tolist())).mean())
        else:
            max_diff2.append(0)
            min_diff2.append(0)
            avg_diff2.append(0)
         
    matrix_var['Diff_1_max'] = max_diff1
    matrix_var['Diff_1_min'] = min_diff1
    matrix_var['Diff_1_avg'] = avg_diff1
    
    matrix_var['Diff_2_max'] = max_diff2
    matrix_var['Diff_2_min'] = min_diff2
    matrix_var['Diff_2_avg'] = avg_diff2                                                  
    return matrix_var

def Features(md):
    required_feature = Glucose(md.iloc[:,:-1])
    scalr_var = StandardScaler()
    meal_std = scalr_var.fit_transform(required_feature)
    
    pca_var = PCA(n_components=12)
    pca_var.fit(meal_std)
    
    with open('picle_file.pkl', 'wb') as (f):
        pickle.dump(pca_var, f)
        
    pca_meal_var = pd.DataFrame(pca_var.fit_transform(meal_std))
    return pca_meal_var

def Calc_Entropy(val_of_calc):
    entr_val= []
    for value_ofinsulin in val_of_calc:
        value_ofinsulin = np.array(value_ofinsulin)
        value_ofinsulin = value_ofinsulin / float(value_ofinsulin.sum())
        cal_entr_value = (value_ofinsulin * [ np.log2(glucose) if glucose!=0 else 0 for glucose in value_ofinsulin]).sum()
        entr_val += [cal_entr_value]
   
    return entr_val

def Purity(val_of_calc):
    purity_of_meal = []
    for value_ofinsulin in val_of_calc:
        value_ofinsulin = np.array(value_ofinsulin)
        value_ofinsulin = value_ofinsulin / float(value_ofinsulin.sum())
        purity_val_calc = value_ofinsulin.max()
        purity_of_meal += [purity_val_calc]
    return purity_of_meal


def CalcDBSCAN(dbscan_var,temp_var,pca2_var):
     for i in temp_var.index:
         dbscan_var=0
         for index,row in pca2_var[pca2_var['clusters']==i].iterrows(): 
             row_var_test=list(temp_var.iloc[0,:])
             meal_var_test=list(row[:-1])
             for q in range(0,12):
                 dbscan_var+=((row_var_test[q]-meal_var_test[q])**2)
     return dbscan_var

def CalcClusterMatrix(truthvalue,cluster_value,f):
    result= np.zeros((f, f))
    for p,q in enumerate(truthvalue):
         temp_var1 = q
         temp_var2 = cluster_value[p]
         result[temp_var1,temp_var2]+=1
    return result

if __name__=='__main__':
       
    insulin_info=pd.read_csv("InsulinData.csv",low_memory=False)
    g_info=pd.read_csv("CGMData.csv",low_memory=False)
    p_info,level_of_insulin = CalcData(insulin_info,g_info)
    
    meal_pca = Features(p_info)

#calculating Kmeans with 6 clusters
var_of_kmeans = KMeans(n_clusters=6)
var_of_kmeans.fit_predict(meal_pca)
label_var=list(var_of_kmeans.labels_)
dataframe = pd.DataFrame()
dataframe['bins']=level_of_insulin
dataframe['kmeans_clusters']=label_var 

matrix_calc = CalcClusterMatrix(dataframe['bins'],dataframe['kmeans_clusters'],6)
matrix_entr_val = Calc_Entropy(matrix_calc)
MatrixPurity = Purity(matrix_calc)
temp_var = np.array([InsulinValue.sum() for InsulinValue in matrix_calc])
value_of_count = temp_var / float(temp_var.sum())

kmean_var = var_of_kmeans.inertia_
kmean_purity_var =  (MatrixPurity*value_of_count).sum()
kmean_entr_var = -(matrix_entr_val*value_of_count).sum()


dbscan_info=pd.DataFrame()
db = DBSCAN(eps=0.127,min_samples=7)
clsters_var = db.fit_predict(meal_pca)
dbscan_info=pd.DataFrame({'pc1':list(meal_pca.iloc[:,0]),'pc2':list(meal_pca.iloc[:,1]),'clusters':list(clsters_var)})
outliner_data_var=dbscan_info[dbscan_info['clusters']==-1].iloc[:,0:2]


start=0
no_of_bins = 6
p = max(dbscan_info['clusters'])
while p<no_of_bins-1:
        max_label=stats.mode(dbscan_info['clusters']).mode[0] 
        data_cluster=dbscan_info[dbscan_info['clusters']==stats.mode(dbscan_info['clusters']).mode[0]] #mode(dbscan_df['clusters'])]
        means1_var= KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(data_cluster)
        means2_var=list(means1_var.labels_)
        data_cluster['bi_pcluster']=means2_var
        data_cluster=data_cluster.replace(to_replace =0,  value =max_label) 
        data_cluster=data_cluster.replace(to_replace =1,  value =max(dbscan_info['clusters'])+1) 
       
        for x_var,y_var in zip(data_cluster['pc1'],data_cluster['pc2']):
            NewDataLabel=data_cluster.loc[(data_cluster['pc1'] == x_var) & (data_cluster['pc2'] == y_var)]
            dbscan_info.loc[(dbscan_info['pc1'] == x_var) & (dbscan_info['pc2'] == y_var),'clusters']=NewDataLabel['bi_pcluster']
        dataframe['clusters']=dbscan_info['clusters']
        p+=1  


matrix_dbscan = CalcClusterMatrix(dataframe['bins'],dbscan_info['clusters'],6)
    
cluster_entr = Calc_Entropy(matrix_dbscan)
cluster_purity = Purity(matrix_dbscan)
temp_var = np.array([InsulinValue.sum() for InsulinValue in matrix_dbscan])
value_of_count = temp_var / float(temp_var.sum())

meal2= meal_pca. join(dbscan_info['clusters'])

Centroids = meal2.groupby(dbscan_info['clusters']).mean()

dbscan = CalcDBSCAN(start,Centroids.iloc[:, : 12],meal2)
DBSCANPurity =  (cluster_purity*value_of_count).sum()        
DBSCANEntropy = -(cluster_entr*value_of_count).sum()

result_var = pd.DataFrame([[kmean_var,dbscan,kmean_entr_var,DBSCANEntropy,kmean_purity_var,DBSCANPurity]],columns=['K-Means SSE','DBSCAN SSE','K-Means entropy','DBSCAN entropy','K-Means purity','DBSCAN purity'])
result_var=result_var.fillna(0)
result_var.to_csv('Result11.csv',index=False,header=None)
result_var

print('Bin Matrix is :')
print(dataframe['bins'])
print('Kmeans Bin Cluster Matrix is :')
print(matrix_calc)
print('DBSCAN Bin Cluster Matrix is :')
print(matrix_dbscan)