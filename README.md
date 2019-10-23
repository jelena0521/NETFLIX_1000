# 仅用于学习日记，不可用于任何商业
#随机选取1000个数据来评估surprise接口中各个算法的优劣
#因为surprise是对dataframe数据进行分析，而本人希望对netflix中所有数据进行分析（虽然这里暂时抽取1000个用户）所以一开始就对TXT数据进行了CSV转换
#1、TXT转化为CSV  用panada读取TXT，跳过第一行，且加上一列data['itemid']  然后利用to_csv存于CSV
#2、读取所有数据，利用data.reindex转为surprise要求的[user,item,rate]顺序，append成一个完整数据。利用dataset转成surprise要求的格式
#3、定义算法
#实例化算法，利用KFold得到trainset testset
#利用fit训练，test验证
#accuracy.rmse评估
