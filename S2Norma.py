from scipy import *
from scipy.linalg import norm, pinv
import matplotlib
import numpy as np
import os
from sklearn import preprocessing,linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()


def holtwinters(y, alpha, beta, gamma, c, debug=True):
    """
    y - time series data.
    alpha , beta, gamma - exponential smoothing coefficients
                                      for level, trend, seasonal components.
    c -  extrapolated future data points.
          4 quarterly
          7 weekly.
          12 monthly


    The length of y must be a an integer multiple  (> 2) of c.
    """
    # Compute initial b and intercept using the first two complete c periods.
    ylen = len(y)
    print("hahaha"+str(ylen))
    if ylen % c != 0:
        pass
        #return None
    fc = float(c)
    ybar2 = sum([y[i] for i in range(c, 2 * c)]) / fc
    ybar1 = sum([y[i] for i in range(c)]) / fc
    b0 = (ybar2 - ybar1) / fc
    if debug: print "b0 = ", b0

    # Compute for the level estimate a0 using b0 above.
    tbar = sum(i for i in range(1, c + 1)) / fc
    print tbar
    a0 = ybar1 - b0 * tbar
    if debug: print "a0 = ", a0

    # Compute for initial indices
    I = [y[i] / (a0 + (i + 1) * b0) for i in range(0, ylen)]
    if debug: print "Initial indices = ", I

    S = [0] * (ylen + c)
    for i in range(c):
        S[i] = (I[i] + I[i + c]) / 2.0

    # Normalize so S[i] for i in [0, c)  will add to c.
    tS = c / sum([S[i] for i in range(c)])
    for i in range(c):
        S[i] *= tS
        if debug: print "S[", i, "]=", S[i]

    # Holt - winters proper ...
    if debug: print "Use Holt Winters formulae"
    F = [0] * (ylen + c)

    At = a0
    Bt = b0
    Result = []
    for i in range(ylen):
        Atm1 = At
        Btm1 = Bt
        At = alpha * y[i] / S[i] + (1.0 - alpha) * (Atm1 + Btm1)
        Bt = beta * (At - Atm1) + (1 - beta) * Btm1
        S[i + c] = gamma * y[i] / At + (1.0 - gamma) * S[i]
        F[i] = (a0 + b0 * (i + 1)) * S[i]
        print "i=", i + 1, "y=", y[i], "S=", S[i], "Atm1=", Atm1, "Btm1=", Btm1, "At=", At, "Bt=", Bt, "S[i+c]=", S[
            i + c], "F=", F[i]
        print i, y[i], F[i]
        Result.append(F[i])

    # Forecast for next c periods:
    for m in range(c):
        print "forecast:", (At + Bt * (m + 1)) * S[ylen + m]
    return Result


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters

        self.centers = [random.uniform(0, 1, indim) for i in xrange(numCenters)]

        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        print(rnd_idx)
        self.centers = [X[i, :] for i in rnd_idx]

        print "center\n", self.centers
        # calculate activations of RBFs
        print("-----------------------")
        G = self._calcAct(X)
        print G

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
def LoadData(input_folder,start,end):
    filelist = os.listdir(input_folder)[start:end]
    all_lines_X = []
    all_lines_Y = []
    all_lines_Data = []
    for eachfile in filelist:
        with open(os.path.join(input_folder, eachfile))as fin:
            for eachline in fin.readlines():
                if 'TOTALVOLTAGE' in eachline: continue
                val = eachline.strip().split(',')
                val = filter(lambda a:a.strip(),val)
                tempX = []

                try:
                    tempX.append(float(val[0]))#TOTALVOLTAGE
                    tempX.append(float(val[1]))#TOTALCURRENT
                    tempX.append(float(val[2]))#SOC
                    tempX.append(float(val[3]))#MAXVOLTAGE
                    tempX.append(float(val[6]))#MINVOLTAGE
                    tempX.append(float(val[9]))#ABVOLTAGE
                    tempX.append(float(val[10]))#NORMALVOLTAGE
                    tempX.append(float(val[11]))#MAXTEMPERATURE
                    tempX.append(float(val[14]))#MINTEMPERATURE
                    tempX.append(float(val[18]))#ABTEMPERATURE
                    tempX.append(float(val[17]))#NORMALTEMPERATURE

                except:
                    continue
                all_lines_X.append(tempX)
                all_lines_Y.append(float(val[2]))
                #tempX.insert(0, float(val[2]))  # SOC
                all_lines_Data.append(tempX[:])

    Y = np.array(all_lines_Y)
    X = np.array(all_lines_X)
    D = np.array(all_lines_Data)

    X_Mean = X.mean(axis=0)
    X_Std = X.std(axis=0)
    D_Mean = D.mean(axis=0)
    D_Std = D.std(axis=0)

    X = preprocessing.scale(X)
    Y = preprocessing.scale(Y)

    #print("1111")
    #print(X.shape)
    #print(D.shape)
    #print(X_Mean)
    #print(X_Std)
    #print("2222")
    #print(D_Mean)
    #print(D_Std)
    return X,Y


def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST


def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)

    results['polynomial'] = coeffs.tolist()

    p = np.poly1d(coeffs)
    yhat = p(x)

    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    print" results :", results
    return results
def Generate_Pattern(Data_OneDimen,window=60):
    N = len(Data_OneDimen)

    A = int(N/window)

    Result = []

    for tab1 in range(A):
        Result.append([])
        for tab2 in range(window):
            Result[-1].extend(Data_OneDimen[tab1*window+tab2])
    return Result

def ReverseScale(Data_One_Dimension,Mean_One_Dimension,Std_One_Dimension):
    data = list(Data_One_Dimension)[:]
    Temp = []
    for tab in range(len(data)):
        Temp.append(data[tab]*Std_One_Dimension+Mean_One_Dimension)
    if type(Data_One_Dimension) == list:
        return Temp
    else:
        return np.array(Temp)

def Step_2_Run():
    input_folder_name = "S1_extremeinfo"
    input_folder = os.path.join(os.getcwd(), input_folder_name)

    output_folder = os.path.join(os.getcwd(), input_folder_name.replace("S1","S2"))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist = os.listdir(input_folder)
    all_lines_X = []
    all_lines_Y = []
    all_lines_Data = []
    all_lines_Time = []

    for eachfile in filelist:
        #all_lines_X = []
        #all_lines_Y = []
        #all_lines_Data = []
        #all_lines_Time = []

        print(eachfile)
        with open(os.path.join(input_folder, eachfile))as fin:
            for eachline in fin.readlines():
                if 'TOTALVOLTAGE' in eachline: continue
                val = eachline.strip().split(',')
                val = filter(lambda a:a.strip(),val)
                tempX = []
                try:
                    tempX.append(float(val[0]))#TOTALVOLTAGE
                    tempX.append(float(val[1]))#TOTALCURRENT
                    tempX.append(float(val[2]))#SOC
                    tempX.append(float(val[3]))#MAXVOLTAGE
                    tempX.append(float(val[6]))#MINVOLTAGE
                    tempX.append(float(val[9]))#ABVOLTAGE
                    tempX.append(float(val[10]))#NORMALVOLTAGE
                    tempX.append(float(val[11]))#MAXTEMPERATURE
                    tempX.append(float(val[14]))#MINTEMPERATURE
                    tempX.append(float(val[18]))#ABTEMPERATURE
                    tempX.append(float(val[17]))#NORMALTEMPERATURE
                except:
                    continue

                all_lines_X.append(tempX)
                all_lines_Y.append(float(val[2]))


                all_lines_Data.append(tempX[:])
                all_lines_Time.append(val[-1])

                #tempX.append(val[-1])#RECEIVETIME
                #tempX.append("hahahahahha")
                #tempX = []
                #count += 1
                #print(all_lines_Time)

    D = np.array(all_lines_Data)
    D_Mean = D.mean(axis=0)
    D_Std = D.std(axis=0)
    all_lines_Data = preprocessing.scale(D)
    AAA = ReverseScale(all_lines_Data[:,0],D_Mean[0],D_Std[0])
    all_lines2 = []
    for tab in range(len(all_lines_Data)):
        all_lines2.append([])
        all_lines2[tab].extend(all_lines_Data[tab])
        all_lines2[tab].append(all_lines_Time[tab])
    print(D[:,0])
    print(all_lines_Data[:,0])
    print(AAA)
    print(len(AAA))

    """
    with open(os.path.join(output_folder,'Normal_'+'2015_2016.txt'),"w")as fout:
        fout.write("TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE\n")
        for eachline in all_lines2:
            eachline = map(lambda a:str(a),eachline)
            fout.write(','.join(eachline)+'\n')

    with open(os.path.join(output_folder,'Mean_'+'2015_2016.txt'),"w")as fout:
        fout.write("TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE\n")
        #fout.write("SOC,TOTALVOLTAGE,TOTALCURRENT,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE\n")
        D_Mean = map(lambda a:str(a),list(D_Mean))
        fout.write(','.join(D_Mean)+'\n')

    with open(os.path.join(output_folder, 'Std_' + '2015_2016.txt'), "w")as fout:
        fout.write("TOTALVOLTAGE,TOTALCURRENT,SOC,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE\n")
        #fout.write("SOC,TOTALVOLTAGE,TOTALCURRENT,MAXVOLTAGE,MINVOLTAGE,ABVOLTAGE,NORMALVOLTAGE,MAXTEMPERATURE,MINTEMPERATURE,ABTEMPERATURE,NORMALTEMPERATURE\n")
        D_Std = map(lambda a: str(a), list(D_Std))
        fout.write(','.join(D_Std) + '\n')
    """



    feature_selected = 1

    X,Y = LoadData(input_folder,0,1)
    ratio = 0.7
    x_train1 = X[0:int(len(X)*ratio),1]
    x_train2 = X[0:int(len(X)*ratio),2]
    x_train3 = X[0:int(len(X)*ratio),3]
    x_train4 = X[0:int(len(X)*ratio),4]

    y_train = Y[0:int(len(X)*ratio)]
    x_test = X[int(len(X)*ratio):-1,feature_selected]
    y_test = Y[int(len(X)*ratio):-1]

    index = [i for i in range(len(y_train))]
    plt.subplot(4,1,1)
    plt.plot(index,x_train1,'b')
    plt.grid()

    plt.subplot(4,1,2)
    plt.plot(index,x_train2,'b')
    plt.grid()

    plt.subplot(4,1,3)
    plt.plot(index,x_train3,'b')
    plt.grid()
    plt.subplot(4,1,4)
    plt.plot(index,x_train4,'b')
    #plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("A.png")
    plt.show()



    #print "r : ", computeCorrelation(x_train[:,feature_selected], y_train)
    #print "r^2 : ", str(computeCorrelation(x_train[:,feature_selected], y_train) ** 2)
    #print polyfit(x_train[:,feature_selected], y_train, 1)["determination"]
    #ref_lr = linear_model.LinearRegression()
    #ref_lr = linear_model.RANSACRegressor()

    #ref_lr.fit(x_train,y_train)
    #Predict = ref_lr.predict(x_test)
    #index = [i for i in range(len(y_test))]
    #plt.plot(index,y_test,'b',label='True')
    #plt.plot(index,Predict,'r',label="Predict")
    #plt.legend()
    #plt.grid()
    #plt.tight_layout()
    #plt.show()

    #index = [i for i in range(len(Y))]
    #plt.subplot(411)
    #print(Y)
    #B_1 = holtwinters(list(Y), 0.2, 0.1, 0.05, 4)
    #B_1 = Y
    #print(B_1)
    #A_1 = [i for i in range(len(B_1))]
    #B_2 = holtwinters(list(X[:,0]), 0.2, 0.1, 0.05, 4)
    #B_2 = X[:,0]
    #A_2 = [i for i in range(len(B_2))]
    #B_3 = holtwinters(list(X[:,1]), 0.2, 0.1, 0.05, 4)
    #B_3 = X[:,1]
    #A_3 = [i for i in range(len(B_3))]
    #B_4 = holtwinters(list(X[:,2]), 0.2, 0.1, 0.05, 4)
    #B_4 = X[:,2]
    #A_4 = [i for i in range(len(B_4))]
    #plt.plot(A_1,B_1)
    #plt.subplot(412)
    #plt.plot(A_2,B_2)
    #plt.subplot(413)
    #plt.plot(A_3,B_3)
    #plt.subplot(414)
    #plt.plot(A_4,B_4)
    #plt.show()
    #X1 = X [0:3600]
    #Y1 = Y [0:3600]
    #X2 = X [12000:13200]
    #Y2 = Y [12000:13200]
    # rbf regression
    #rbf = RBF(3, 3600, 1)
    #rbf.train(X1, Y1)
    #Z = rbf.test(X2)
    #index = [i for i in range(len(Z))]
    #plt.plot(index,Y2,'b',label='True Soc')
    #plt.plot(index,Z,'g',label='Predicted Soc')
    #plt.legend(fontsize=12)
    #plt.grid()
    #plt.tick_params(labelsize=12)
    #plt.tight_layout()
    #plt.savefig("A.png",dpi=400)
    #plt.show()
    #n = 400
    #x0 = mgrid[-1:1:complex(0, n/2)].reshape(n/2, 1)
    #x = mgrid[-1:1:complex(0, n)].reshape(n/2, 2)
    #x = np.array([[1,2,3,4]])
    #y = np.array([22,12,13,222,33,23,4,2,33,45])
    # set y and add random noise
    #y = sin(1 * (x0 + 0.5) ** 3 - 1)
    # y += random.normal(0, 0.1, y.shape)
    # rbf regression
    #rbf = RBF(2, 10, 1)
    #rbf.train(x, y)
    #z = rbf.test(x)
    # plot original data
    #fig = plt.figure(figsize=(12, 8))
    #X = [i for i in range(len(z))]
    #plt.plot(X,y,'b')
    #plt.plot(X,z,'g')
    #ax = Axes3D(fig)
    #x1,x2 = np.meshgrid(x[:,0], x[:,1])
    #plt.plot(x[:,0], y, 'b-')
    #plt.plot(x[:,1], y, 'g-')
    #ax.plot_surface(x1,x2,y,rstride=1, cstride=1, cmap='rainbow')
    # plot learned model
    #ax.plot_surface(x1,x2,z,rstride=1, cstride=1)
    #plt.plot(x[:,0], z, 'r-', linewidth=2)
    #plt.plot(x[:,1], z, 'k-', linewidth=2)
    # plot rbfs
    #plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
    #for c in rbf.centers:
        # RF prediction lines
        #cx = arange(c - 0.7, c + 0.7, 0.01)
        #cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        #plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
    #plt.xlim(-1.2, 1.2)
    #plt.grid()
    #plt.tick_params(labelsize=12)
    #plt.tight_layout()
    #plt.savefig("A.png",dpi=400)
    #plt.show()
    #print(x)

if __name__ == '__main__':
    Step_2_Run()


