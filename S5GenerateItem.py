import os
def Step_5_Run(window,k):

    input_folder_name = "S4_extremeinfo"+'_W_'+ str(window) + '_K_'+str(k)

    input_folder = os.path.join(os.getcwd(), input_folder_name)

    output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S4", "S5"))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = filter(lambda a:'Clustering' in a,os.listdir(input_folder))
    A = []
    for eachfile in filelist:
        print("S5 eachfile ...\n")
        A.append([])
        with open(os.path.join(input_folder,eachfile))as fin:
            for eachline in fin.readlines():
                val = eachline.strip().split(',')
                A[-1].append(val[0])

    B = []
    for tab1 in range(len(A[0])):
        B.append([])
        for tab2 in range(len(A)):
            B[-1].append(A[tab2][tab1])

    with open(os.path.join(output_folder,"Association.txt"),"w") as fout:
        for tab in range(len(B)):
            writeline = ','.join(B[tab])
            fout.write(writeline + '\n')
