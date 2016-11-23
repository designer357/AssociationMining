import os
import numpy as np
def WritetoFile(Data,Feature_List,Output_folder,Type):
    assert len(Data) == len(Feature_List)
    for tab in range(len(Data)):
        with open(os.path.join(Output_folder,Feature_List[tab]+'_'+str(Type)+'.txt'),"w")as fout:
            fout.write(','.join(map(lambda a:str(a),Data[tab])) + '\n')

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
        print(eachfile)
        with open(os.path.join(input_folder,eachfile))as fin:
            val_all_lines = fin.readlines()
            all_lines_Month = []
            if len(val_all_lines) > 1:
                for eachline in val_all_lines:
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    tempX = []
                    try:
                        tempX.append(float(val[0]))  # TOTALVOLTAGE
                        tempX.append(float(val[2]))  # SOC
                        tempX.append(float(val[3]))  # MAXVOLTAGE
                        tempX.append(float(val[18]))  # ABTEMPERATURE
                    except:
                        continue

                    all_lines_Month.append(tempX)
            else:
                continue
            all_lines_Month = np.array(all_lines_Month)
            print(all_lines_Month.shape)

            avg_Month.append(np.average(a=all_lines_Month,axis=0))

    avg_Month = np.array(avg_Month).T
    print(avg_Month.shape)
    WritetoFile(avg_Month,Feature_List,output_folder,"avg_Month")

Job1()



