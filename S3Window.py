import os
import numpy as np
def medfilt(x, k):
    print("The length is "+str(len(x)))
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    # for i in range(len(x)):
    # if x[i]:
    # start=i

    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)

def Generate_Pattern(Data_OneDimen,filter_size,window=60):
    N = len(Data_OneDimen)

    A = int(N/window)

    Result = []

    for tab1 in range(A):
        Result.append([])
        temp = []
        for tab2 in range(window):
            temp.append(float(Data_OneDimen[tab1*window+tab2]))
            #print(len(temp))
        #print(np.array(temp))

        temp2 = list(medfilt(np.array(temp),filter_size))
        Result[-1].extend(temp2)

    return Result

def Step_3_Run(window,filter_size,k):
    input_folder_name = "S2_extremeinfo"
    input_folder = os.path.join(os.getcwd(), input_folder_name)

    output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S2", "S3")+'_W_'+ str(window) + '_K_'+str(k))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    all_lines_Data = []
    all_lines_Time = []
    for eachfile in filelist:
        if "normal" in eachfile:pass
        else:continue
        #all_lines_Data = []
        #all_lines_Time = []
        print(eachfile)
        with open(os.path.join(input_folder, eachfile))as fin:
            for eachline in fin.readlines():
                if 'TOTALVOLTAGE' in eachline: continue
                val = eachline.strip().split(',')
                val = filter(lambda a: a.strip(), val)
                all_lines_Data.append(val[:-1])
        all_lines_Data = np.array(all_lines_Data)
        all_lines_Data_T = []
        for tab in range(len(all_lines_Data[0])):
            all_lines_Data_T.append(Generate_Pattern(all_lines_Data[:,tab],filter_size,window))

        Feature_List ="TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE".split(',')
        print(len(all_lines_Data_T))
        print(len(Feature_List))

    for tab in range(len(Feature_List)):
        with open(os.path.join(output_folder,Feature_List[tab]+'_'+"2015_2016.txt"),"w")as fout:
            #fout.write(Feature_List[tab]+'\n')
            for eachline in all_lines_Data_T[tab]:
                eachline = map(lambda a:str(a),eachline)
                fout.write(','.join(eachline)+'\n')
