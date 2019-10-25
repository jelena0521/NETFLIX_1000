import os
import time
import pandas as pd
import random
from surprise import Reader,Dataset
from surprise.model_selection import KFold
from surprise import SVDpp,KNNWithMeans,SlopeOne,BaselineOnly,NormalPredictor
from surprise import accuracy

def process(from_path,save_path):
    # path='training_set'
    if not os.path.exists(from_path):
         print('数据路径不存在')
    path_list=os.listdir(from_path)
    path_list.sort()
    i=0
    start=time.time()
    for f in path_list:
        i=i+1
        file_path=os.path.join(from_path,f)
        name=f.split('.')[0]+'.csv'
        save_path=os.path.join(save_path,name)
        data=pd.read_table(file_path,header=None,skiprows=1,sep=',',names=['userid','rate','time'])
        data['itemid'] = i
        data.to_csv(save_path,index=False)
    end=time.time()
    print('转换数据用时',end-start)
    return i

#读取数据
def get_data(from_path,save_path):
    if not os.listdir(save_path):
        i=process(from_path)
    print('读取数据')
    data_all=pd.DataFrame(columns=('userid','itemid','rate','time'))
    # path_list = os.listdir(save_path)
    # path_list.sort()
    #n=0
    for f in os.listdir(save_path):
        file_path = os.path.join(save_path, f)
        data=pd.read_csv(file_path)
        data=data.reindex(columns=['userid','itemid','rate','time'])
        # n=n+1
        # print('导入数据',n)
        data_all=data_all.append(data)
    print('读完数据')
    return data_all

#随机选用户进行训练
# def select_10000_users(from_path,save_path):
#     data=get_data(from_path,save_path)
#     print('随机选取100个用户')
#     users=set()
#     userids=data['userid']
#     for userid in userids:
#         users.add(userid)
#     users_10000=random.sample(list(users),10000)
#     return users_10000

def get_samples(from_path, save_path):
    #users_10000=select_10000_users(from_path,save_path)
    data = get_data(from_path, save_path)
    #data_samples = pd.DataFrame(columns=('userid', 'itemid', 'rate', 'time'))
    # userids=data['userid']
    # for userid in userids:
    #     if userid in users_10000:
    #         data_samples.append(data[data['userid'].isin([userid])],ignore_index=True)
    # data_samples=data_samples.reindex(columns=['userid', 'itemid', 'rate', 'time'])
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    dataset = Dataset.load_from_df(data.iloc[:, :3], reader=reader)
    #dataset = Dataset.load_from_df(data_samples.iloc[:,:3], reader=reader)
    print('获得surprise数据')
    return dataset


#定义baseline算法SGD跟新
def baseline_sgd(data_set):
    start=time.time()
    algo=BaselineOnly(bsl_options={'method':'sgd','n_epochs':5})
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('baseline_sgd的时间为',(end-start)/60)
    # print('baseline的准确率为',acc)
    return acc

#定义baseline算法ALS跟新
def baseline_als(data_set):
    start=time.time()
    algo=BaselineOnly(bsl_options={'method':'als','n_epochs':5})
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('baseline_als的时间为',(end-start)/60)
    #print('baseline的准确率为',acc)
    return acc


#定义normalpredictor算法
def normalpredictor(data_set):
    start=time.time()
    algo=NormalPredictor()
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('normalpredictor的时间为',(end-start)/60)
    #print('normalpredictor的准确率为',acc)
    return acc

#定义slopeone算法
def slopeone(data_set):
    start=time.time()
    algo=SlopeOne()
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('slopeone的时间为',(end-start)/60)
    #print('slopeone的准确率为',acc)
    return acc



#定义svdpp算法
def svdpp(data_set):
    start=time.time()
    algo=SVDpp()
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('svdpp的时间为',(end-start)/60)
    #print('svdpp的准确率为',acc)
    return acc

#定义knnwithmeans算法
def knnwithmeans(data_set):
    start=time.time()
    algo=KNNWithMeans()
    kf=KFold(n_splits=3)
    for trainset,testset in kf.split(data_set):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('KNNWithMeans的时间为',(end-start)/60)
    #print('KNNWithMeans的准确率为',acc)
    return acc

if __name__=='__main__':
    from_path='training_set'
    save_path='samples'
    data_set=get_samples(from_path,save_path)
     acc1=baseline_sgd(data_set)
     print('baseline_sgd的rsme为', acc1)
     acc2=baseline_als(data_set)
     print('baseline_als的rmse为', acc2)
     acc3=normalpredictor(data_set)
     print('normalpredictor的rmse为', acc3)
     acc4=slopeone(data_set)
     print('slopeone的rmse为', acc4)
     acc5=svdpp(data_set)
     print('svdpp的rmse为', acc5)
    acc6=knnwithmeans(data_set)
    print('knnwithmeans的rmse为', acc6)



'''
读取数据
读完数据
获得surprise数据
RMSE: 0.9569
RMSE: 0.9590
RMSE: 0.9568
baseline_sgd的时间为 5.869822080930074
baseline_sgd的rmse为 0.956793098317742
RMSE: 0.9421
RMSE: 0.9414
RMSE: 0.9415
baseline_als的时间为 6.096065413951874
baseline_als的rmse为 0.9414730747316377
RMSE: 1.4530
RMSE: 1.4522
RMSE: 1.4537
normalpredictor的时间为 5.310757629076639
normalpredictor的rmse为 1.4537487413173003
RMSE: 0.9480
RMSE: 0.9480
RMSE: 0.9482
slopeone的时间为 16.998791563510895
slopeone的rmse为 0.9482381144025018
RMSE: 0.9448
RMSE: 0.9452
RMSE: 0.9441
svdpp的时间为 254.8044933994611
svdpp的rmse为 0.9441227844041323
Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''
