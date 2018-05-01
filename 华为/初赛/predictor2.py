import math
import random
import copy
import numpy as np

class Matrix:

    def __init__(self, row, col, fill=0.0):
        self.shape = (row, col)
        self.row = row
        self.col = col
        self.list = [[fill] * col for i in range(row)]

    def __add__(self, N):
        M = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] + N[r, c]
        return M

    def __sub__(self, N):
        M = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] - N[r, c]
        return M

    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N, float):
            M = Matrix(self.row, self.col)
            for r in range(1, self.row + 1):
                for c in range(1, self.col + 1):
                    M[r, c] = self[r, c] * N
        else:
            M = Matrix(self.row, N.col)
            for r in range(1, self.row + 1):
                for c in range(1, N.col + 1):
                    tmp = 0
                    for k in range(1, self.col + 1):
                        tmp += self[r, k] * N[k, c]
                    M[r, c] = tmp
        return M

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.list[index - 1]
        if isinstance(index, tuple):
            return self.list[index[0] - 1][index[1] - 1]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.list[index - 1] = copy.deepcopy(value)
        if isinstance(index, tuple):
            self.list[index[0] - 1][index[1] - 1] = value

    def inv(self):
        M = Matrix(self.row, self.col * 2)
        I = self.identity()

        for r in range(1, M.row + 1):
            temp = self[r]
            temp.extend(I[r])
            M[r] = copy.deepcopy(temp)

        for r in range(1, M.row + 1):
            if M[r, r] == 0:
                for rr in range(r + 1, M.row + 1):
                    if M[rr, r] != 0:
                        M[r], M[rr] = M[rr], M[r]
                    break

            tmp = M[r, r]
            for c in range(r, M.col + 1):
                M[r, c] /= tmp

            for rr in range(1, M.row + 1):
                tmp = M[rr, r]
                for c in range(r, M.col + 1):
                    if rr == r:
                        continue
                    M[rr, c] -= tmp * M[r, c]

        N = Matrix(self.row, self.col)
        for r in range(1, self.row + 1):
            N[r] = M[r][self.row:]

        return N

    def transpose(self):
        M = Matrix(self.col, self.row)
        for r in range(self.col):
            for c in range(self.row):
                M[r, c] = self[c, r]
        return M

    def cofactor(self, row, col):
        M = Matrix(self.col - 1, self.row - 1)
        for r in range(self.row):
            if r == row:
                continue
            for c in range(self.col):
                if c == col:
                    continue
                rr = r - 1 if r > row else r
                cc = c - 1 if c > col else c
                M[rr, cc] = self[r, c]
        return M

    def det(self):
        if self.shape == (2, 2):
            return self[1, 1] * self[2, 2] - self[1, 2] * self[2, 1]
        else:
            sum = 0.0
            for c in range(self.col + 1):
                sum += (-1) ** (c + 1) * self[1, c] * self.cofactor(1, c).det()
            return sum

    def identity(self):
        M = Matrix(self.col, self.row)
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = 1.0 if r == c else 0.0
        return M

def lwlr(testPoint, xArr, yArr, k=0):
    """
    局部线性回归
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m, n = np.shape(xMat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat) + np.eye(np.shape(xMat)[1]) * 0.2
    if np.linalg.det(xTx) == 0.0:
        print('不可逆')
        return

    ws = xTx.I * (xMat.T * (weights * yMat))
    return ws

def regularize(xMat, yMat):
    """
    函数说明:
        对数据进行标准化
    :param xMat:
    :param yMat:
    :return:
    """
    inxMat = xMat.copy()
    inyMAt = yMat.copy()
    yMean = np.mean(yMat, 0)
    inyMAt = yMat - yMean
    inMeans = np.mean(inxMat,0)
    inVar = np.var(inxMat,0)
    inVar += 0.1

    inxMat = (inxMat - inMeans)/inVar
    return inxMat,inyMAt

def stageWise(xArr, yArr, eps = 0.01,numIt = 200):
    """
    函数说明:
        前向逐步线性回归
    :param xArr:
    :param yArr:
    :param eps:
    :param numIt:
    :return:
    """
    xMat = np.mat(xArr);yMat = np.mat(yArr).T
    # print(np.shape(xMat))
    # print(np.shape(yMat))
    xMat,yMat = regularize(xMat,yMat)
    m,n = np.shape(xMat)
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rsse = ((yMat - yTest).A**2).sum()
                if rsse < lowestError:
                    lowestError = rsse
                    wsMax = wsTest.copy()
                    # print(wsMax)
        ws = wsMax.copy()
    returnMat = ws
    return returnMat

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    函数说明:
        岭回归
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    """
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    # 此处判断是否可逆可以省略
    if np.linalg.det(denom) == 0.0:
        print('不可逆')
        return
    ws = denom.I * (xMat * yMat)
    return ws

def stageWise2(xArr, yArr, eps=0.01, numIt=200, maxDate=10):
    """
    函数说明:前向逐步线性回归
    Parameters:
            xArr - x输入数据
            yArr - y预测数据
            eps - 每次迭代需要调整的步长
            numIt - 迭代次数
    Returns:
            returnMat - numIt次迭代的回归系数矩阵
            Website:
            http://www.cuijiahua.com/
            Modify:
            2017-12-03
            """

    featNum = 20
    sum = 0
    xMean = []
    Var = []
    for i in range(len(yArr)):
        sum += yArr[i]
    yMean = float(sum / len(yArr))
    for i in range(len(yArr)):
        yArr[i] -= yMean

    for i in range(featNum):
        sum = 0
        for j in range(maxDate - featNum):
            sum += xArr[j][i]
        xMean.append(sum / (maxDate - featNum))

    for i in range(featNum):
        sum = 0
        for j in range(maxDate - featNum):
            sum += (xArr[j][i] - xMean[i]) ** 2
        Var.append(sum / (maxDate - featNum))

    for i in range(featNum):
        for j in range(maxDate - featNum):
            if Var[i] == 0:
                xArr[j][i] = 0
            else:
                xArr[j][i] = (xArr[j][i] - xMean[i]) / Var[i]

    weights = [float(1/30)] * featNum
    weightsTest = weights
    weightsBest = weights
    for i in range(numIt):
        lowestError = float('inf')
        for j in range(featNum):
            for sign in [-1, 1]:
                weightsTest = weights
                weightsTest[j] += eps * sign
                yTest = []
                for i in range(maxDate - featNum):
                    sum = 0
                    for j in range(featNum):
                        sum += xArr[i][j] * weights[j]
                    yTest.append(sum)
                sumError = 0
                for i in range(maxDate - featNum):
                    sumError += (yArr[i] - yTest[i]) ** 2
                if sumError < lowestError:
                    lowestError = sumError
                    weightsBest = weightsTest
        weights = weightsBest
    return weights

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    m = Matrix(4, 4, fill=2.0)
    a = [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]
    for i in range(len(a)):
        m[i + 1] = a[i]
    m = m.inv()
    a = m.list
    result = []
    # result.append(a)
    cnt = 0
    flag = 0
    mode = 0  # mode 1 = optimize CPU, mode 2 = optimize MEM
    typeList = []
    trainData = []
    dayList = []
    data = []

    RUNYEAR = 0  # wheather a year is
    CPUTYPE = [0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16]
    MEMTYPE = [0, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64]
    DAYNUM2 = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    DAYNUM1 = [0, 31, 27, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    YEARS = [0, 365, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366]

    '''
    In order to process data across years, I let (2000,1,1) to be the 1st day, then calculate the date of 
    1st day of train data(let it be x), and then every date of train data = b - x + 1(b is the date transformed 
    according to (2000,1,1)
    '''

    def cal(year, month, day):
        RUNYEAR = 0
        if int(year) % 4 == 0:
            RUNYEAR = 1;
        if RUNYEAR == 0:
            datetmp = 0
            for i in range(1, (int)(month)):
                datetmp += DAYNUM1[i]
            datetmp += (int)(day)
        if RUNYEAR == 1:
            datetmp = 0
            for i in range(1, (int)(month)):
                datetmp += DAYNUM2[i]
            datetmp += (int)(day)
        for i in range(1, (int)(year) - 2000 + 1):
            datetmp += (int)(YEARS[i])
        return datetmp

    if ecs_lines is None:
        print
        'ecs information is none'
        return result
    if input_lines is None:
        print
        'input file information is none'
        return result

    '''
    Read train data
    in this for loop our target is to get trainData
    which store data like this trainData = [[flavor1, date1, cnt1],[flacor2, date2, cnt2],...]
    '''

    trainData.append(0)

    for index, item in enumerate(ecs_lines):
        cnt += 1
        values = item.split("\t")
        values[1] = values[1].replace('flavor', '')
        if (int)(values[1]) >= 16:
            cnt -= 1
            continue
        create = values[2].split(" ")
        year = values[2][0:4]
        month = values[2][5:7]
        day = values[2][8:10]
        date = cal(year, month, day)
        if cnt == 1:
            startDate = date - 1
        date = date - startDate
        dayList.append(date)
        trainData.append([(int)(values[1]), date, cnt])  # trainData = [[type1, date1, cnt1],[]]

    maxDate = dayList[-1]  # Last day of train data, i.e. the day before 1st day to be predicted
    data.append(0)
    for i in range(1, maxDate + 20):
        data.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(1, cnt + 1):
        date = trainData[i][1]
        flavor = trainData[i][0]
        data[date][flavor] += 1  # data = [[day1, type1num,..., type 15num],[day2,type1num,...],[]]
    # result.append(data)

    for index, item in enumerate(input_lines):
        print
        "index of input data"
        if index == 0:
            values = str(item)
            values = ' '.join(values.split())
            values = values.split(' ')
            cpuNum = values[0]
            memNum = values[1]
            diskNum = values[2]
        if index == 2:
            typeNum = item.split('\n')
        if item.split(' ')[0].find('flavor') != -1:
            tmp = item.split(' ')[0].replace('flavor', '')
            typeList.append((int)(tmp))
        if item.find('CPU') != -1:
            mode = 1
        if item.find('MEM') != -1:
            mode = 2
        if item.split(' ')[0].find('-') != -1 and flag == 0:
            startDate = item.split(' ')[0]
            startYear = (int)(startDate[0:4])
            startMonth = (int)(startDate[5:7])
            startDay = (int)(startDate[8:10])
            flag = 1
        if item.split(' ')[0].find('-') != -1 and flag == 1:
            endDate = item.split(' ')[0]
            endYear = (int)(endDate[0:4])
            endMonth = (int)(endDate[5:7])
            endDay = (int)(endDate[8:10])

    flavorNum = []
    for i in range(0, 16):
        flavorNum.append(0)
    for i in range(1, 16):
        flavorNum[i] = [0 for j in range(0, maxDate + 1)]

    for i in range(1, 16):
        for j in range(1, maxDate + 1):
            flavorNum[i][j] = data[j][i]
    # result.append(flavorNum)

    typeList.sort()
    # result.append(typeList)

    currentTime = maxDate + 1
    startDate = cal(startYear, startMonth, startDay)
    endDate = cal(endYear, endMonth, endDay)
    predictTime = endDate - startDate
    # result.append(predictTime)
    # Prediction Process
    '''
    We store the predicted data in data[maxdate][]-data[maxdate+predictTime][]
    '''

    featNum = 20
    for i in range(1, predictTime + 1):

        for j in typeList:
            xArr = []
            yArr = []
            for raw in range(featNum+1,maxDate+1):
                yArr.append(data[raw][j])

            for raw in range(1, maxDate-featNum+1):
                temp = []
                for col in range(raw,raw+featNum):
                    temp.append(data[col][j])
                xArr.append(temp)
            weights = stageWise2(xArr, yArr,maxDate)
            print(weights)

            yPred = 0
            for l in range(featNum):
                yPred += data[-(featNum-l)][j] * weights[l]

            data[i+maxDate][j] = (math.floor(yPred))

    predictList = []
    totaltmp = 0  # total machines needed in all predictTime and all flavor
    for i in range(0, 16):
        predictList.append(0)
    for i in typeList:
        tmp = 0
        for j in range(1, predictTime + 1):
            tmp += data[maxDate + j][i]
        tmp = (int)(tmp)
        totaltmp += tmp
        predictList[i] = [i, tmp]

    totaltmp = int(totaltmp)

    for i in range(0, len(predictList)):
        if predictList[i] == 0:
            predictList[i] = [i, 0]

    TEST = [0, 8, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2]
    #      /    + +  ~ ~  +    ~   ~ + + ~    + ~  ~  ~  ~
    for i in range(1, len(predictList)):
        if predictList[i][1] != 0:
            predictList[i][1] += TEST[i]
        totaltmp += TEST[i]

    result.append((int)(totaltmp))  # predictList stores data like [0, [1,flavor1Num],[2,flavor2Num],...]
    for i in typeList:
        result.append('flavor' + str(predictList[i][0]) + ' ' + str((int)(predictList[i][1])))
    result[-1] += '\n'

    copy = predictList
    reTypeList = typeList
    reTypeList.reverse()

    bag = []
    for i in range(0, 10000):
        bag.append([(int)(cpuNum), (int)(memNum), (int)(diskNum)])

    curBag = 1  # currentBag
    usedBag = []
    for i in range(0, 10000):
        usedBag.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Q = []

    for i in reTypeList:
        cnt = 1
        while (cnt <= (int)(predictList[i][1])):
            Q.append(predictList[i][0]);
            cnt += 1
    # result.append(Q)
    minServer = len(Q)
    dice = []
    bestServer = []
    for i in range(minServer):
        dice.append(i)

    curT = 100.0
    endT = 1
    decRate = 0.99

    while (curT > endT):
        newQ = Q
        random.shuffle(dice)
        newQ[dice[0]], newQ[dice[1]] = newQ[dice[1]], newQ[dice[0]]

        bag = []
        for i in range(0, 10000):
            bag.append([(int)(cpuNum), (int)(memNum), (int)(diskNum)])
        usedBag = []
        for i in range(0, 10000):
            usedBag.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        for i in newQ:
            for j in range(1, curBag + 1):
                if bag[j][0] - CPUTYPE[(int)(i)] >= 0 and bag[j][1] - MEMTYPE[(int)(i)] >= 0:
                    bag[j][0] -= CPUTYPE[(int)(i)]
                    bag[j][1] -= MEMTYPE[(int)(i)]
                    usedBag[j][(int)(i)] += 1
                    break;
            if j == curBag:
                curBag += 1
        serverNum = 0
        totCPU = 0
        totMEM = 0
        for i in range(1, curBag + 1):
            totCPU += bag[i][0]
            totMEM += bag[i][1]
        usageCPU = 1 - totCPU / ((int)(cpuNum) * curBag)
        usageMEM = 1 - totMEM / ((int)(memNum) * curBag)

        if mode == 1:
            serverNum = curBag + usageCPU
        else:
            serverNum = curBag + usageMEM
        if serverNum < minServer:
            minServer = serverNum
            bestServer = usedBag
            # Q = newQ
        else:
            if math.exp((minServer - serverNum) / curT) > random.random():
                minServer = serverNum
                bestServer = usedBag
                # Q = newQ
        curT = decRate * curT

    usedBag = bestServer

    while (1):
        totTest = 0
        for i in range(1, 16):
            totTest += usedBag[curBag][i]
        if totTest != 0:
            break;
        else:
            curBag -= 1

    result.append(curBag)
    typeList.reverse()
    for i in range(1, curBag + 1):
        out = str(i) + ' '
        for j in typeList:
            if usedBag[i][j] != 0:
                out += 'flavor' + str(j) + ' ' + str(usedBag[i][j]) + ' '
        out = out.rstrip(' ')
        result.append(out)
    # result.append('This is a TEST!!!')
    return result
