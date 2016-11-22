import os
def Step_1_Run():
    input_folder_name = "cellinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"S1_" + input_folder_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    for eachfile in filelist:
        print(eachfile)
        with open(os.path.join(input_folder,eachfile))as fin:
            all_lines = []
            for eachline in fin.readlines():
                val = eachline.strip().split(',')

                if not len(val) > 1: continue
                all_lines.append(val[1].replace('\"','')+'_'+val[3].replace('\"','')+',')
                all_lines.append(val[6].replace('\"','')+',')

                try:
                    all_lines.append(str(int(val[7].replace('\"',''))+40)+',')
                except:
                    all_lines.append(val[7].replace('\"','')+',')

                all_lines.append(val[8].replace('\"','')+',')
                all_lines.append('\n')

        with open(os.path.join(output_folder,'c_'+eachfile),"w")as fout:
            for eachline in all_lines:
                fout.write(eachline)

    input_folder_name = "extremeinfo"
    input_folder = os.path.join(os.getcwd(),input_folder_name)
    output_folder = os.path.join(os.getcwd(),"S1_" + input_folder_name)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)

    for eachfile in filelist:
        print(eachfile)
        with open(os.path.join(input_folder,eachfile))as fin:
            all_lines = []
            for eachline in fin.readlines():
                try:
                    val = eachline.strip().split(',')
                    all_lines.append(val[1].replace('\"','')+',')
                    all_lines.append(val[2].replace('\"','')+',')
                    all_lines.append(val[3].replace('\"','')+',')
                    all_lines.append(val[4].replace('\"','')+',')
                    all_lines.append(val[5].replace('\"','')+',')
                    all_lines.append(val[6].replace('\"','')+',')
                    all_lines.append(val[7].replace('\"','')+',')
                    all_lines.append(val[8].replace('\"','')+',')
                    all_lines.append(val[9].replace('\"','')+',')
                    all_lines.append(val[10].replace('\"','')+',')
                    all_lines.append(val[11].replace('\"','')+',')
                    all_lines.append(val[13].replace('\"','')+',')
                    all_lines.append(val[14].replace('\"','')+',')
                    all_lines.append(val[15].replace('\"','')+',')
                    all_lines.append(val[16].replace('\"','')+',')
                    all_lines.append(val[17].replace('\"','')+',')
                    all_lines.append(val[18].replace('\"','')+',')
                    all_lines.append(val[19].replace('\"','')+',')
                    all_lines.append(val[20].replace('\"','')+',')
                    all_lines.append(val[22].replace('\"','')+',')

                    all_lines.append('\n')
                except:
                    continue
        with open(os.path.join(output_folder,'e_'+eachfile),"w")as fout:
            for eachline in all_lines:
                fout.write(eachline)
