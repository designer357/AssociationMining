# coding:utf-8
from numpy import *
import itertools
import os
import shutil
import re
import matplotlib
import matplotlib.pyplot as plt
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()
support_dic = {}
from S2Norma import *

# 生成原始数据，用于测试
def loadDataSet(window,k):

    input_folder_name = "S5_extremeinfo"+'_W_'+ str(window) + '_K_'+str(k)

    input_folder = os.path.join(os.getcwd(), input_folder_name)
    #output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S5", "S6"))
    #if not os.path.isdir(output_folder):
        #os.makedirs(output_folder)

    filelist = os.listdir(input_folder)

    for eachfile in filelist:
        if 'Association' in eachfile:
            pass
        else:
            continue
        D = []
        with open(os.path.join(input_folder, eachfile))as fin:
            for eachline in fin.readlines():
                D.append([])
                val = eachline.strip().split(',')
                val = filter(lambda a: len(a) > 1, val)
                D[-1].extend(val)
        print(len(D))
    return D
    #return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 获取整个数据库中的一阶元素
def createC1(dataSet):
    C1 = set([])
    for item in dataSet:
        C1 = C1.union(set(item))
    return [frozenset([i]) for i in C1]


# 输入数据库（dataset） 和 由第K-1层数据融合后得到的第K层数据集（Ck），
# 用最小支持度（minSupport)对 Ck 过滤，得到第k层剩下的数据集合（Lk）
def getLk(dataset, Ck, minSupport):
    global support_dic
    Lk = {}
    # 计算Ck中每个元素在数据库中出现次数
    for item in dataset:
        for Ci in Ck:
            if Ci.issubset(item):
                if not Ci in Lk:
                    Lk[Ci] = 1
                else:
                    Lk[Ci] += 1
    # 用最小支持度过滤
    Lk_return = []
    for Li in Lk:
        support_Li = Lk[Li] / float(len(dataset))
        if support_Li >= minSupport:
            Lk_return.append(Li)
            support_dic[Li] = support_Li
    return Lk_return


# 将经过支持度过滤后的第K层数据集合（Lk）融合
# 得到第k+1层原始数据Ck1
def genLk1(Lk):
    Ck1 = []
    for i in range(len(Lk) - 1):
        for j in range(i + 1, len(Lk)):
            if sorted(list(Lk[i]))[0:-1] == sorted(list(Lk[j]))[0:-1]:
                Ck1.append(Lk[i] | Lk[j])
    return Ck1


# 遍历所有二阶及以上的频繁项集合
def genItem(freqSet, support_dic,minConf):
    global Rules_Left,Rules_Right,Rules_Confidence

    for i in range(1, len(freqSet)):
        for freItem in freqSet[i]:
            genRule(freItem,minConf)
    #print(Rules_Right)
    return Rules_Left,Rules_Right,Rules_Confidence


# 输入一个频繁项，根据“置信度”生成规则
# 采用了递归，对规则树进行剪枝
def genRule(Item, minConf=0.5):
    global Rules_Left,Rules_Right,Rules_Confidence
    if len(Item) >= 2:
        for element in itertools.combinations(list(Item), 1):
            if support_dic[Item] / float(support_dic[Item - frozenset(element)]) >= minConf:
                print str([Item - frozenset(element)]) + "----->" + str(element) + ',confidence:'+ str(support_dic[Item] / float(support_dic[Item - frozenset(element)]))
                Rules_Left.append(str(list([Item - frozenset(element)][0])))
                Rules_Right.append(str(list(element)[0]))
                Rules_Confidence.append(str(support_dic[Item] / float(support_dic[Item - frozenset(element)])))
                genRule(Item - frozenset(element),minConf)





# 输出结果
#if __name__ == '__main__':

def Step_6_Run(minSupp,minConf,window,k):

    global Rules_Left,Rules_Right,Rules_Confidence
    Rules_Left = []
    Rules_Right = []
    Rules_Confidence = []

    Feature_List = "TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE".split(
        ',')
    D_Mean = []
    D_Std = []
    for each_file in os.listdir(os.path.join(os.getcwd(),"S2_extremeinfo")):
        with open(os.path.join(os.path.join(os.getcwd(),"S2_extremeinfo"),each_file))as fin:
            val = fin.readlines()[1].strip().split(',')
            if 'Mean' in each_file:
                D_Mean = map(lambda a:float(a),val)
            elif 'Std' in each_file:
                D_Std = map(lambda a: float(a), val)


    input_folder_name = "S4_extremeinfo"+'_W_'+ str(window) + '_K_'+str(k)

    input_folder = os.path.join(os.getcwd(), input_folder_name)

    output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S4", "S5"))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = filter(lambda a: 'Centors' in a, os.listdir(input_folder))

    dataSet = loadDataSet(window,k)
    result_list = []
    Ck = createC1(dataSet)
    # 循环生成频繁项集合，直至产生空集
    while True:
        Lk = getLk(dataSet, Ck, minSupp)
        if not Lk:
            break
        result_list.append(Lk)
        Ck = genLk1(Lk)
        if not Ck:
            break
    #输出频繁项及其“支持度”
    #print(support_dic)
    #输出规则
    #print("AAA")
    #print((result_list))
    genItem(result_list, support_dic,minConf)
    Rules_Left2 = []
    Rules_Right2 = []
    Rules_Confidence2 = []

    #print("BBB")
    #print((Rules_Right))
    for tab in range(len(Rules_Left)):
        temp1 = Rules_Left[tab].replace('[','').replace(']','')
        temp2 = Rules_Right[tab].replace('[','').replace(']','')
        temp3 = Rules_Confidence[tab].replace('[','').replace(']','')

        if not temp1 in Rules_Left2:
            Rules_Left2.append(temp1.replace('[','').replace(']',''))
            Rules_Right2.append(temp2.replace('[','').replace(']',''))
            Rules_Confidence2.append(temp3.replace('[', '').replace(']', ''))

    Rules_Left2 = map(lambda a:a.replace('\'',''),Rules_Left2)
    Rules_Right2 = map(lambda a:a.replace('\'',''),Rules_Right2)
    Rules_Confidence2 = map(lambda a:a.replace('\'',''),Rules_Confidence2)

    Temp = []
    Temp.extend(Rules_Left2)
    Temp.extend(Rules_Right2)


    D = []

    for each in Temp:
        for e in each.strip().split(','):
            if not e in D:
                D.append(e.strip())
    D = list(set(D))
    print("Frequent Item Set:")
    print(D)
    #print(Rules_Left2)
    #print(Rules_Right2)
    #print(Rules_Confidence2)
    #print(len(Rules_Left2[0]))

    AAA = {}
    for e in D:
        for eachfile in filelist:
            with open(os.path.join(input_folder,eachfile))as fin:
                for eachline in fin.readlines():
                    val = eachline.split(',',1)
                    if val[0] in D:
                        AAA[val[0]] = val[1].strip()

    #print(AAA)

    output_folder = os.path.join(os.getcwd(),"Out_put_"+'W_'+str(window)+'_K_'+str(k) + '_MinSupp_'+str(minSupp)+'_MinConf_'+str(minConf))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    Name_Cond = []
    Name_Result = []
    for tab_base in range(len(Rules_Left2)):
        Name_Cond_Temp = []
        for e_name_left in Rules_Left2[tab_base].strip().split(','):
            Name_Cond_Temp.append(Feature_List[int(e_name_left.strip().replace('F_','').split('_')[0].strip())])
        Name_Result_Temp = []
        for e_name_right in Rules_Right2[tab_base].split(','):
            Name_Result_Temp.append(Feature_List[int(e_name_right.replace('F_','').split('_')[0])])

        Name_Cond.append(','.join(Name_Cond_Temp))
        Name_Result.append(','.join(Name_Result_Temp))

    Value_Cond = []
    Value_Result = []

    for tab_base in range(len(Rules_Left2)):
        Value_Cond_Temp = []
        for e_value_left in Rules_Left2[tab_base].strip().split(','):
            condvalue = map(lambda a: float(a), AAA[e_value_left.strip()].strip().replace('[', '').replace(']', '').split(','))
            d_mean = D_Mean[int(e_value_left.strip().replace('F_','').split('_')[0])]
            d_std = D_Std[int(e_value_left.strip().replace('F_','').split('_')[0])]
            condvalue = ReverseScale(condvalue,d_mean,d_std)
            Value_Cond_Temp.append(condvalue)

        Value_Result_Temp = []
        for e_value_right in Rules_Right2[tab_base].split(','):
            resultvalue = map(lambda a: float(a), AAA[e_value_right.strip()].strip().replace('[', '').replace(']', '').split(','))
            d_mean = D_Mean[int(e_value_right.strip().replace('F_','').split('_')[0])]
            d_std = D_Std[int(e_value_right.strip().replace('F_','').split('_')[0])]
            resultvalue = ReverseScale(resultvalue,d_mean,d_std)
            Value_Result_Temp.append(resultvalue)
        Value_Cond.append(Value_Cond_Temp)
        Value_Result.append(Value_Result_Temp)

    for tab_base in range(len(Rules_Left2)):
        output_folder_subfolder = os.path.join(output_folder,"Rule_"+str(tab_base+1))
        if not os.path.isdir(output_folder_subfolder):
            os.makedirs(output_folder_subfolder)
        with open(os.path.join(output_folder_subfolder,"Rule_"+str(tab_base+1) + '.txt'),"w") as fout1:
            fout1.write(Name_Cond[tab_base] + '------------>' + Name_Result[tab_base] + ', confidence:' + Rules_Confidence2[tab_base] + '\n')

        with open(os.path.join(output_folder_subfolder,"Rule_Condition_"+str(tab_base+1) + '.txt'),"w") as fout2:
            for each_value in Value_Cond[tab_base]:
                each_value = map(lambda a:str(a),list(each_value))
                fout2.write(','.join(each_value).strip() + '\n')

        Plotting(Name_Cond[tab_base],Value_Cond[tab_base],output_folder_subfolder,"Rule_Condition_"+str(tab_base+1))
        Plotting(Name_Result[tab_base],Value_Result[tab_base],output_folder_subfolder,"Rule_Result_"+str(tab_base+1))

        with open(os.path.join(output_folder_subfolder, "Rule_Result_" + str(tab_base + 1) + '.txt'), "w") as fout3:
            for each_value in Value_Result[tab_base]:
                each_value = map(lambda a:str(a),list(each_value))
                fout3.write(','.join(each_value).strip() + '\n')
            #print(Rules_Left2[tab])

    #print(Name_Cond)
    #print(Rules_Left2)
    #print(Value_Cond)
    #print(D_Mean)
    return AAA
def Plotting(Each_Name,Each_Value,Out_Put_Folder,Out_Put_Name):
    Temp = Each_Name.split(',')
    index = [i for i in range(len(Each_Value[0]))]
    plt.figure(figsize=(12,6))
    if len(Temp) == 1:
        plt.plot(index,Each_Value[0],'b',label= Temp[0])
        plt.legend()
        plt.grid()
    elif len(Temp) == 2:
        plt.subplot(121)
        plt.plot(index,Each_Value[0],'b',label= Temp[0])
        plt.legend()
        plt.grid()
        plt.subplot(122)
        plt.plot(index, Each_Value[1], 'b',label= Temp[1])
        plt.legend()
        plt.grid()
    elif len(Temp) == 3:
        plt.subplot(131)
        plt.plot(index, Each_Value[0], 'b',label= Temp[0])
        plt.legend()
        plt.grid()
        plt.subplot(132)
        plt.plot(index, Each_Value[1], 'b',label= Temp[1])
        plt.legend()
        plt.grid()
        plt.subplot(133)
        plt.plot(index, Each_Value[2], 'b',label= Temp[2])
        plt.legend()
        plt.grid()
    elif len(Temp) == 4:
        plt.subplot(141)
        plt.plot(index, Each_Value[0], 'b',label= Temp[0])
        plt.legend()
        plt.grid()
        plt.subplot(142)
        plt.plot(index, Each_Value[1], 'b',label= Temp[1])
        plt.legend()
        plt.grid()
        plt.subplot(143)
        plt.plot(index, Each_Value[2], 'b',label= Temp[2])
        plt.legend()
        plt.grid()
        plt.subplot(144)
        plt.plot(index, Each_Value[3], 'b',label= Temp[3])
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(Out_Put_Folder, Out_Put_Name + '.png'),dpi=200)
    plt.clf()
    #for tab in range(len(Temp)):



#Step_6_Run(0.55,1)
