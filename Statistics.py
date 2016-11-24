import os
import numpy as np
import pandas as pd
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
                battery = val[1].replace('\"', '') + '_' + val[3].replace('\"', '') + ','
                #if not 'NUM' in battery:
                if not battery in Battery1:
                    Battery1.append(battery.replace(',',''))

    print(Battery1)
    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"AJOB1_" + input_folder_name.replace('S1_',''))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    #Feature_List = ["TOTALVOLTAGE","SOC","MAXVOLTAGE","TEMPERATURE"]
    for eachfile in filelist:
        #if '2015' in eachfile:continue
        print(eachfile)
        """
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

                        tempX.append(float(val[4]))  #MAXVOLTAGEBOXNUM
                        temp.append(float(val[5])) #MAXVOLTAGEBATTERYNUM
                        tempX.append(float(val[7]))  #MINVOLTAGEBOXNUM
                        temp.append(float(val[8])) #MINVOLTAGEBATTERYNUM

                        tempX.append(float(val[4]))  # MAXVOLTAGEBOXNUM
                        temp.append(float(val[5]))  # MAXVOLTAGEBATTERYNUM
                        tempX.append(float(val[7]))  # MINVOLTAGEBOXNUM
                        temp.append(float(val[8]))  # MINVOLTAGEBATTERYNUM



                    except:
                        continue

                    all_lines_Month.append(tempX)
                    all_lines_Month2.append(temp)

            else:
                continue
        """



Job2()

"""
2015-11-01 22:05:00    3.416499  97.211876    31.000000    573.649855
2015-11-01 22:05:30    3.415318  97.239800    31.000000    573.688098
2015-11-01 22:06:00    3.414078  97.265919    31.000000    573.726239
"""

