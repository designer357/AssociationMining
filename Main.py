from S1RmIr import *
from S2Norma import *
from S3Window import *
from S4Clustering import *
from S5GenerateItem import *
from S6Apriori import *
filter_size = 7
window = 120
k = 2
min_supp = 0.3
min_conf = 0.6

Feature_List = "TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE".split(',')

Step_3_Run(window,filter_size,k)
Step_4_Run(window,k)
Step_5_Run(window,k)
Result = Step_6_Run(min_supp,min_conf,window,k)
Label = []
"""
for eachk, eachv in Result.items():
    temp = eachk.strip()[2]
    print(Feature_List[int(temp)])
    eachv = map(lambda a: float(a), eachv.strip().split(','))
    x_index = [i for i in range(len(eachv))]
    plt.plot(x_index, eachv, 'bs-')
    plt.show()
"""
