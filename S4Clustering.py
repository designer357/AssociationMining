import os
import numpy as np
from sklearn.cluster import KMeans
def Step_4_Run(window,k):
    number_of_cluster = k
    input_folder_name = "S3_extremeinfo"+'_W_'+ str(window) + '_K_'+str(k)

    input_folder = os.path.join(os.getcwd(), input_folder_name)
    Feature_List = "TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE".split(
        ',')

    output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S3", "S4"))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    for tab1 in range(len(Feature_List)):
        all_lines_Data = []
        for eachfile in filelist:
            if Feature_List[tab1] in eachfile:
                pass
            else:
                continue

            with open(os.path.join(input_folder, eachfile))as fin:
                for eachline in fin.readlines():
                    if 'TOTALVOLTAGE' in eachline: continue
                    val = eachline.strip().split(',')
                    val = filter(lambda a: a.strip(), val)
                    all_lines_Data.append(val)
        all_lines_Data = np.array(all_lines_Data)
        print(all_lines_Data.shape)
        kmeans = KMeans(n_clusters=number_of_cluster, random_state=0).fit(all_lines_Data)
        all_lines_Cluster_Labels = kmeans.labels_
        all_lines_Cluster_Centers = kmeans.cluster_centers_
        centor_labels = [i for i in range(number_of_cluster)]

        print(len(all_lines_Cluster_Labels))
        with open(os.path.join(output_folder, Feature_List[tab1] + '_Clustering' + "2015_2016.txt"), "w")as fout:
            # fout.write(Feature_List[tab]+'\n')
            all_lines_Cluster_Labels = map(lambda a: 'F_'+str(tab1)+'_'+str(a), list(all_lines_Cluster_Labels))
            for tab2 in range(len(all_lines_Cluster_Labels)):
                writeline = all_lines_Cluster_Labels[tab2] + ',' + ','.join(all_lines_Data[tab2])
                fout.write(writeline + '\n')
        with open(os.path.join(output_folder, Feature_List[tab1] + '_Centors' + "2015_2016.txt"), "w")as fout:
            for tab2 in range(len(all_lines_Cluster_Centers)):
                temp = map(lambda a:  str(a),list(all_lines_Cluster_Centers[tab2]))
                writeline = 'F_' + str(tab1) + '_' + str(centor_labels[tab2])+',' + ','.join(temp)
                fout.write(writeline + '\n')


