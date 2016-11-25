import os
import time
import numpy as np
import pandas as pd
import collections
import datetime
def WritetoFile(Data,Feature_List,Output_folder,Type):
    assert len(Data) == len(Feature_List)
    for tab in range(len(Data)):
        with open(os.path.join(Output_folder,Feature_List[tab]+'_'+str(Type)+'.txt'),"w")as fout:
            fout.write(','.join(map(lambda a:str(a),Data[tab])) + '\n')
def _sum(x):
    if len(x) == 0: return 0
    else: return sum(x)
def Job1():
    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"AJOB1_" + input_folder_name.replace('S1_',''))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    avg_Month = []
    avg_Day = []
    Feature_List = ["TOTALVOLTAGE","SOC","MAXVOLTAGE","TEMPERATURE"]
    for eachfile in filelist:
        #if '2015' in eachfile:continue
        print(eachfile)

        with open(os.path.join(input_folder,eachfile))as fin:
            val_all_lines = fin.readlines()
            all_lines_Month = []
            all_lines_Month2 = []

            if len(val_all_lines) > 1:
                for eachline in val_all_lines:
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    tempX = []
                    temp = []
                    try:
                        temp.append(pd.to_datetime(val[-1]))#TIME

                        tempX.append(float(val[0]))  # TOTALVOLTAGE
                        temp.append(float(val[0]))

                        tempX.append(float(val[2]))  # SOC
                        temp.append(float(val[2]))

                        tempX.append(float(val[3]))  # MAXVOLTAGE
                        temp.append(float(val[3]))

                        tempX.append(float(val[18]))  # ABTEMPERATURE
                        temp.append(float(val[18]))


                    except:
                        continue

                    all_lines_Month.append(tempX)
                    all_lines_Month2.append(temp)

            else:
                continue
            all_lines_Month = np.array(all_lines_Month)
            all_lines_Month2 = np.array(all_lines_Month2).T
            #print(all_lines_Month2[0])


            df = pd.DataFrame({'TIME_STAMP':all_lines_Month2[0],
                               Feature_List[0]: all_lines_Month2[1],
                               Feature_List[1]: all_lines_Month2[2],
                               Feature_List[2]: all_lines_Month2[3],
                               Feature_List[3]: all_lines_Month2[4]})

            df['TOTALVOLTAGE'] = df.TOTALVOLTAGE.astype(float)
            df['SOC'] = df.SOC.astype(float)
            df['MAXVOLTAGE'] = df.MAXVOLTAGE.astype(float)
            df['TEMPERATURE'] = df.TEMPERATURE.astype(float)

            print(df.head())
            df = df.set_index(['TIME_STAMP'])
            df = df[~df.index.duplicated(keep='first')]
            df_reindexed = df.reindex(pd.date_range(start=df.index.min(),
                                                           end=df.index.max(),
                                                           freq='30S'))

            df = df_reindexed.interpolate(method='index', axis=0)
            #df['daily'] = df.resample('D',how='mean')
            #davg = df.resample('D', how='mean')
            #print(davg)
            ##davg_NA = davg.loc[df.index]
            ##davg_daily = davg_NA.fillna(method='ffill')
            ##print(AAA)
            #print(df.index)

            DFList = [group[1] for group in df.groupby(df.index.day)]
            #print(DFList[0])
            #print(df[df.index.duplicated()])
            #DFList[0] = DFList[0][~DFList[0].index.duplicated(keep='first')]
            #df_reindexed = DFList[0].reindex(pd.date_range(start=DFList[0].index.min(),
             #                                       end=DFList[0].index.max(),
              #                                      freq='30S'))
            #AAA = df_reindexed.interpolate(method = 'index', axis = 0)
            #AAA = df_reindexed.interpolate(method = 'piecewise_polynomial', axis = 0)
            for tab_DF_List in range(len(DFList)):
                #print(DFList[tab_DF_List].resample('1D', how='mean').values.tolist())
                avg_Day.append(DFList[tab_DF_List].resample('1D', how='mean').values.tolist()[0])


            avg_Month.append(np.average(a=all_lines_Month,axis=0))

    avg_Month = np.array(avg_Month).T

    avg_Day = np.array(avg_Day).T
    print(avg_Day.shape)
    WritetoFile(avg_Day, list(DFList[0].columns), output_folder, "avg_Day")

    print(avg_Month.shape)
    WritetoFile(avg_Month,Feature_List,output_folder,"avg_Month")

def Job2():
    input_folder_name1 = "cellinfo"
    input_folder1 = os.path.join(os.getcwd(),input_folder_name1)
    output_folder1 = os.path.join(os.getcwd(),"S1_" + input_folder_name1)
    if not os.path.isdir(output_folder1):
        os.makedirs(output_folder1)

    filelist1 = os.listdir(input_folder1)
    Battery1 = []
    for eachfile1 in filelist1:
        if '.DS' in eachfile1:continue
        print(os.path.join(input_folder1,eachfile1))
        with open(os.path.join(input_folder1,eachfile1))as fin1:
            for eachline in fin1.readlines():
                val = eachline.strip().split(',')
                if not len(val) > 1: continue
                battery = val[1].replace('\"', '') + '_' + val[3].replace('\"', '')
                if not 'NUM' in battery:
                    if not battery in Battery1:
                        Battery1.append(battery)

    print(Battery1)
    Battery = collections.defaultdict(int)
    for eachbattery in Battery1:
        Battery[eachbattery] = 0

    Battery_Max_V = collections.OrderedDict(sorted(Battery.items()))
    Battery_Min_V = collections.OrderedDict(sorted(Battery.items()))
    Battery_Max_T = collections.OrderedDict(sorted(Battery.items()))
    Battery_Min_T = collections.OrderedDict(sorted(Battery.items()))


    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"AJOB2_" + input_folder_name.replace('S1_',''))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    #Feature_List = ["TOTALVOLTAGE","SOC","MAXVOLTAGE","TEMPERATURE"]
    for eachfile in filelist:
        #if '2015' in eachfile:continue
        print(eachfile)

        with open(os.path.join(input_folder,eachfile))as fin:
            val_all_lines = fin.readlines()
            if len(val_all_lines) > 1:
                for eachline in val_all_lines:
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    try:

                        Max_Vol = str(val[4]) + '_' + str(val[5])  #MAXVOLTAGEBOXNUM_MAXVOLTAGEBATTERYNUM
                        Battery_Max_V[Max_Vol] += 1

                        Min_Vol = str(val[7]) + '_' + str(val[8])  #MINVOLTAGEBOXNUM_MINVOLTAGEBATTERYNUM
                        Battery_Min_V[Min_Vol] += 1

                        Max_Temp = str(val[12]) + '_' + str(val[13])  # MAXTEMPERATUREBOXNUM_MAXTEMPERATUREBATTERYNUM
                        Battery_Max_T[Max_Temp] += 1

                        Min_Temp = str(val[15]) + '_' + str(val[16])  # MINTEMPERATUREBOXNUM_MINTEMPERATUREBATTERYNUM
                        Battery_Min_T[Min_Temp] += 1



                    except:
                        continue

            else:
                continue


    with open(os.path.join(output_folder,"Battery_Num_List.txt"),'w')as fout:
        for eachbattery in Battery1:
            fout.write(str(eachbattery)+',')
        fout.write('\n')
    with open(os.path.join(output_folder,"Battery_Max_Voltage_Distribution.txt"),'w')as fout:
        for eachbattery in Battery1:
            fout.write(str(eachbattery)+':'+str(Battery_Max_V[eachbattery])+',')
        fout.write('\n')
    with open(os.path.join(output_folder,"Battery_Min_Voltage_Distribution.txt"),'w')as fout:
        for eachbattery in Battery1:
            fout.write(str(eachbattery)+':'+str(Battery_Min_V[eachbattery])+',')
        fout.write('\n')
    with open(os.path.join(output_folder,"Battery_Max_Temperature_Distribution.txt"),'w')as fout:
        for eachbattery in Battery1:
            fout.write(str(eachbattery)+':'+str(Battery_Max_T[eachbattery])+',')
        fout.write('\n')
    with open(os.path.join(output_folder,"Battery_Min_Temperature_Distribution.txt"),'w')as fout:
        for eachbattery in Battery1:
            fout.write(str(eachbattery)+':'+str(Battery_Min_T[eachbattery])+',')
        fout.write('\n')



def Job3_2():
    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"AJOB3_" + input_folder_name.replace('S1_',''))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    Temperature = []
    SOC = []
    Feature_List = ["SOC","TEMPERATURE"]
    for eachfile in filelist:
        #if '2015' in eachfile:continue
        print(eachfile)

        with open(os.path.join(input_folder,eachfile))as fin:
            val_all_lines = fin.readlines()
            all_lines_T = []
            all_lines_SOC = []

            if len(val_all_lines) > 1:
                for eachline in val_all_lines:
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    temp_T = []
                    temp_SOC = []
                    try:
                        temp_T.append(pd.to_datetime(val[-1]))#TIME
                        temp_T.append(float(val[18]))  # TEMPERATURE
                        temp_T.append(pd.to_datetime(val[-1].split(' ')[-1]))  # TIME ONLY


                        temp_SOC.append(pd.to_datetime(val[-1]))#TIME
                        temp_SOC.append(float(val[2]))  # SOC
                        temp_SOC.append(str(val[-1].split(' ')[-1]))  # TIME ONLY

                    except:
                        continue

                    all_lines_T.append(temp_T)
                    all_lines_SOC.append(temp_SOC)

            else:
                continue
            all_lines_T = np.array(all_lines_T).T
            all_lines_SOC = np.array(all_lines_SOC).T
            #print(all_lines_Month2[0])


            df_T = pd.DataFrame({'TIME_STAMP':all_lines_T[0],
                               "TEMPERATURE": all_lines_T[1],
                               "TIMEONLY": all_lines_T[2]})


            df_SOC = pd.DataFrame({'TIME_STAMP':all_lines_SOC[0],
                               "SOC": all_lines_SOC[1],
                                "TIMEONLY": all_lines_SOC[2]})

            df_T['TEMPERATURE'] = df_T.TEMPERATURE.astype(float)
            df_T['TIMEONLY'] = df_T.TIMEONLY.astype(str)

            df_SOC['SOC'] = df_SOC.SOC.astype(float)
            df_SOC['TIMEONLY'] = df_SOC.TIMEONLY.astype(str)

            #print(df_T.head())
            #print(df_SOC.head())

            df_T = df_T.set_index(['TIME_STAMP'])
            df_SOC = df_SOC.set_index(['TIME_STAMP'])

            df_T = df_T[~df_T.index.duplicated(keep='first')]
            df_SOC = df_SOC[~df_SOC.index.duplicated(keep='first')]

            #df_T_reindexed = df_T.reindex(pd.date_range(start=df_T.index.min(),end=df_T.index.max(),
            #                            freq='30T'))

            #df_SOC_reindexed = df_SOC.reindex(pd.date_range(start=df_SOC.index.min(),end=df_SOC.index.max(),
            #                            freq='30T'))

            #df = df_T_reindexed.interpolate(method='index', axis=0)
            #df['daily'] = df.resample('D',how='mean')
            #davg = df.resample('D', how='mean')
            ##davg_NA = davg.loc[df.index]
            ##davg_daily = davg_NA.fillna(method='ffill')

            DFList_T = [group[1] for group in df_T.groupby(df_T.index.day)]
            DFList_SOC = [group[1] for group in df_SOC.groupby(df_SOC.index.day)]

            print(DFList_T[0])
            #print(df[df.index.duplicated()])
            #DFList[0] = DFList[0][~DFList[0].index.duplicated(keep='first')]

            #AAA = df_reindexed.interpolate(method = 'index', axis = 0)
            #AAA = df_reindexed.interpolate(method = 'piecewise_polynomial', axis = 0)

            #AAA = AAA.interpolate(method='index', axis=0)
            #print("111111")
            #print(AAA.head())
            #print("222222")

            # AAA = AAA.interpolate(method='index', axis=0)
            #print("33333333")
            #import datetime
            #aaa = pd.DatetimeIndex(DFList_T[0].index).time[-1]
            #bbb = pd.DatetimeIndex(DFList_T[1].index).date[0]
            #bbb2 = np.array([bbb for i in range(len(aaa))])
            import datetime
            #datetime.datetime.combine(datetime.date(2011, 01, 01), datetime.time(10, 23))
            #print(aaa)
            #print(bbb2)
            #print(datetime.datetime.combine(bbb, aaa[0]))
            #print(pd.to_datetime(aaa + bbb2))
            #print("44444444")

            AAA = pd.DataFrame({'TEMPERATURE': DFList_T[0]['TEMPERATURE'],
                                  'TIMEONLY': DFList_T[0]['TIMEONLY']})
            AAA = AAA.set_index(['TIMEONLY'])


            for tab_DF_List_T in range(len(DFList_T)):
                newDF = pd.DataFrame({'TEMPERATURE':DFList_T[tab_DF_List_T]['TEMPERATURE'],'TIMEONLY':DFList_T[tab_DF_List_T]['TIMEONLY']})
                newDF = newDF.set_index(['TIMEONLY'])
                print("NNNNNNNNNN")
                print(newDF.head())
                #df_reindexed = (DFList_T[tab_DF_List_T].reindex(pd.date_range(start=AAA.index.min(),end=AAA.index.max(), freq='30T'))).interpolate(method='index', axis=0)
                df_reindexed = (DFList_T[tab_DF_List_T].reindex(pd.date_range(start=AAA.index.min(),end=AAA.index.max(), freq='30T')))
                Abc = df_reindexed.resample('30T', how='mean')

                #AAA  =  pd.concat([AAA['TEMPERATURE'], df_reindexed['TEMPERATURE']], join='outer', axis = 1)

                #print((DFList_T[tab_DF_List_T].resample('30T', how='mean').values.tolist()))
                print(Abc.head())
                #print((AAA.resample('30T', how='mean').values.tolist()))
                Temperature.append(DFList_T[tab_DF_List_T].resample('30T', how='mean').values.tolist())
            for tab_DF_List_SOC in range(len(DFList_SOC)):
                SOC.append(DFList_SOC[tab_DF_List_SOC].resample('30T', how='mean').values.tolist())

            """
            for tab_DF_List_T in range(len(DFList_T)):
                start_time = datetime.datetime.combine(pd.DatetimeIndex(DFList_T[tab_DF_List_T].index).date[0],pd.DatetimeIndex(DFList_T[0].index).time[0])
                end_time = datetime.datetime.combine(pd.DatetimeIndex(DFList_T[tab_DF_List_T].index).date[0],pd.DatetimeIndex(DFList_T[0].index).time[-1])

                df_reindexed = (DFList_T[tab_DF_List_T].reindex(pd.date_range(start=start_time,end=end_time, freq='30S'))).interpolate(method='index', axis=0)
                AAA  =  pd.concat([AAA['TEMPERATURE'], df_reindexed['TEMPERATURE']], join='outer', axis = 1)

                print((DFList_T[tab_DF_List_T].resample('30T', how='mean').values.tolist()))
                print(AAA.head())
                #print((AAA.resample('30T', how='mean').values.tolist()))
                Temperature.append(DFList_T[tab_DF_List_T].resample('30T', how='mean').values.tolist())
            for tab_DF_List_SOC in range(len(DFList_SOC)):
                SOC.append(DFList_SOC[tab_DF_List_SOC].resample('30T', how='mean').values.tolist())
            """
    Temperature = np.array(Temperature).T
    SOC = np.array(SOC).T
    print(Temperature)
    #WritetoFile(avg_Day, list(DFList[0].columns), output_folder, "avg_Day")
    print(SOC)
    #WritetoFile(avg_Month,Feature_List,output_folder,"avg_Month")
def Job3():
    start_time = 1446307200.0
    A_T = np.zeros(86400)
    A_SOC = np.zeros(86400)
    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"AJOB3_" + input_folder_name.replace('S1_',''))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    #Temperature = []
    Total_Soc = []
    Total_Temperature = []
    for eachfile in filelist:
        #if '2015' in eachfile:continue
        print(eachfile)
        TEMPERATURE_ = []
        SOC_ = []

        Temperature = []
        Soc = []

        TIME_STAMP_ = []
        with open(os.path.join(input_folder,eachfile))as fin:
            val_all_lines = fin.readlines()
            if len(val_all_lines) > 1:
                for eachline in val_all_lines:
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    try:
                        temp_time = datetime.datetime.strptime(val[-1].split('.')[0],'%Y-%m-%d %H:%M:%S')
                        TIME_STAMP_.append(time.mktime(temp_time.timetuple()))#TIME
                        TEMPERATURE_.append(float(val[18]))  # TEMPERATURE
                        SOC_.append(float(val[2]))  # SOC

                    except:
                        continue
            else:
                continue

            for tab_i in range(len(TIME_STAMP_)):
                if TIME_STAMP_[tab_i] - start_time < 86400:
                    A_T[int(TIME_STAMP_[tab_i]-start_time)] = TEMPERATURE_[tab_i]
                    A_SOC[int(TIME_STAMP_[tab_i]-start_time)] = SOC_[tab_i]
                else:
                    Temperature.append(list(Average(A_T)))
                    Soc.append(list(Average(A_SOC)))

                    start_time += 86400
                    A_T = np.zeros(86400)
                    A_SOC = np.zeros(86400)

        Temperature = np.array(Temperature)
        Soc = np.array(Soc)

        Temperature_M = Mean(Temperature)
        Soc_M = Mean(Soc)


        Total_Temperature.append(list(Temperature_M))
        Total_Soc.append(list(Soc_M))

        print(Temperature_M)
        print(Soc_M)

        with open(os.path.join(output_folder,eachfile[2:9].replace('.','')+'_TEMPERATURE.txt'),'w')as fout:
            for each in Temperature_M:
                fout.write(str(each)+',')
            fout.write('\n')

        with open(os.path.join(output_folder,eachfile[2:9].replace('.','')+'_SOC.txt'),'w')as fout:
            for each in Soc_M:
                fout.write(str(each)+',')
            fout.write('\n')

    Total_Temperature = np.array(Total_Temperature)
    Total_Soc = np.array(Total_Soc)

    Total_Temperature_M = Mean(Total_Temperature)
    Total_Soc_M = Mean(Total_Soc)

    with open(os.path.join(output_folder, 'TOTAL_TEMPERATURE.txt'), 'w')as fout:
        for each in Total_Temperature_M:
            fout.write(str(each) + ',')
        fout.write('\n')

    with open(os.path.join(output_folder, 'TOTAL_SOC.txt'), 'w')as fout:
        for each in Total_Soc_M:
            fout.write(str(each) + ',')
        fout.write('\n')
    Total_Temperature_List = []
    Total_Soc_List = []
    for each1 in Total_Temperature:
        for each2 in each1:
            if each2 > 0:
                Total_Temperature_List.append(each2)

    for each1 in Total_Soc:
        for each2 in each1:
            if each2 > 0:
                Total_Soc_List.append(each2)


def Mean(Data):
    N = len(Data)
    D = len(Data[0])
    Temp = []
    for i in range(D):
        sum = 0
        count = 0
        for j in range(N):
            if Data[j,i] > 0: count += 1
            sum += Data[j,i]
        try:
            Temp.append(float(sum)/count)
        except:
            Temp.append(float(sum)/(count+1))

    return np.array(Temp)
def Average(Data):
    N = len(Data)
    n = 1800
    Temp = []
    for i in range(N/n):
        sum = 0
        count = 1
        for j in range(n):
            if Data[i*n+j] > 0: count += 1
            sum += Data[i*n+j]
        Temp.append(float(sum)/count)
    return np.array(Temp)

            #all_lines_T = np.array(all_lines_T).T
            #all_lines_SOC = np.array(all_lines_SOC).T
Job3()
"""
2015-11-01 22:05:00    3.416499  97.211876    31.000000    573.649855
2015-11-01 22:05:30    3.415318  97.239800    31.000000    573.688098
2015-11-01 22:06:00    3.414078  97.265919    31.000000    573.726239
"""

