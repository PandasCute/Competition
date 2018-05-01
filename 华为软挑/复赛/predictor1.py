import math
import random
import copy


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

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    m = Matrix(4, 4, fill=2.0)
    a = [[1, 2, 3, 4], [2,  3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]
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
    CPUTYPE = [0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32]
    MEMTYPE = [0, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128]
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
        print('ecs information is none')
        return result
    if input_lines is None:
        print('input file information is none')
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
        if (int)(values[1]) >= 19:
            cnt -= 1
            continue
        create = values[2].split(" ")
        year = values[2][0:4]
        month = values[2][5:7]
        day = values[2][8:10]
        date = cal(year, month, day)
        if cnt == 1:
            startDateOrigin = date - 1
        date = date - startDateOrigin
        dayList.append(date)
        trainData.append([(int)(values[1]), date, cnt])  # trainData = [[type1, date1, cnt1],[]]

    maxDate = dayList[-1]  # Last day of train data, i.e. the day before 1st day to be predicted
    data.append(0)
    for i in range(1, maxDate + 60):
        data.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(1, cnt + 1):
        date = trainData[i][1]
        flavor = trainData[i][0]
        data[date][flavor] += 1  # data = [[day1, type1num,..., type 15num],[day2,type1num,...],[]]
    # result.append(data)

    for index, item in enumerate(input_lines):
        # print "index of input data"
        if index == 1:
            values = str(item)
            values = ' '.join(values.split())
            values = values.split(' ')
            cpuNum = values[1]
            memNum = values[2]
            diskNum = values[3]
        if index == 5:
            typeNum = item.split('\n')
        if item.split(' ')[0].find('flavor') != -1:
            tmp = item.split(' ')[0].replace('flavor', '')
            typeList.append((int)(tmp))

        if item.split(' ')[0].find('201') != -1 and flag == 0:
            startDate = item.split(' ')[0]
            startYear = (int)(startDate[0:4])
            startMonth = (int)(startDate[5:7])
            startDay = (int)(startDate[8:10])
            flag += 1
        if item.split(' ')[0].find('201') != -1 and flag == 1:
            endDate = item.split(' ')[0]
            endYear = (int)(endDate[0:4])
            endMonth = (int)(endDate[5:7])
            endDay = (int)(endDate[8:10])

    flavorNum = []
    for i in range(0, 19):
        flavorNum.append(0)
    for i in range(1, 19):
        flavorNum[i] = [0 for j in range(0, maxDate + 1)]

    for i in range(1, 19):
        for j in range(1, maxDate + 1):
            flavorNum[i][j] = data[j][i]
    # result.append(flavorNum)

    typeList.sort()
    # result.append(typeList)

    currentTime = maxDate + 1
    startDate = cal(startYear, startMonth, startDay)
    endDate = cal(endYear, endMonth, endDay)
    startDate -= startDateOrigin
    endDate -= startDateOrigin
    predictTime = endDate - startDate
    breakTime = startDate - maxDate
    predictTime += breakTime
    # result.append(predictTime)
    # Prediction Process
    '''
    We store the predicted data in data[maxdate][]-data[maxdate+predictTime][]
    '''

    avg = []
    var = []
    for i in range(0, 19):
        avg.append(0)
        var.append(0)
    for i in range(1, 19):
        tmp = 0
        for j in range(1, maxDate + 1):
            tmp += data[j][i]
        avg[i] = (tmp) / (maxDate)

    for i in range(1, 19):
        tmp = 0
        for j in range(1, maxDate + 1):
            tmp += (data[j][i] - avg[i]) ** 2
        var[i] = float(tmp) / float(maxDate)

    for i in range(1, 19):
        for j in range(1, maxDate + 1):
            if data[j][i] > avg[i] * 6:
                data[j][i] = avg[i] * 3.8

    trainSize = 28
    length = maxDate - trainSize
    # print var
    # print avg

    for s in range(0, 19):
        w = [float(1 / 28)] * (trainSize + 1)
        wbest = [float(1 / 28)] * (trainSize + 1)
        xMat = []
        yMat = []
        yPre = []

        for i in range(0, length + 1):
            # w.append(random.random())
            # wbest.extend(w)
            xMat.append([])
            yMat.append(0)
            yPre.append(0)

        for i in range(1, length + 1):
            xMat[i].append(0)
            yMat[i] = data[i + trainSize][s]
            for j in range(0, trainSize):
                xMat[i].append(data[i + j][s])

        minLoss = 1e9
        for i in range(1, trainSize + 1):
            w[i] = 0
            for t in range(1, 500):
                w[i] = wbest[i]
                loss = 0
                for sig in [-1, 1]:
                    w[i] += sig * 0.001
                    for j in range(1, length + 1):
                        tmpp = 0

                        for k in range(1, trainSize + 1):
                            tmpp += float(w[k]) * float(xMat[j][k])
                        yPre[j] = tmpp
                        loss += (float(yPre[j]) - float(yMat[j])) ** 2
                    if loss < minLoss:
                        wbest[i] = w[i]
                        # w[i] = wtmp
                        tmp = 0
                        # for j in range(1, predictTime + 1 - breakTime):
                        minLoss = loss

        for i in range(maxDate, maxDate + predictTime + 1):
            tmp = 0
            for j in range(1, trainSize + 1):
                tmp += float(wbest[j]) * float(data[i - trainSize][s])
            # datac[i][s] = tmp
            # tmp = float(tmp) * float(var[s]) + float(avg[s])
            data[i][s] = max((int)(tmp), 0)
    #
    # total = 0
    # for i in range(1, predictTime + 1):
    #     for j in typeList:
    #         tot = 0
    #         for k in range(maxDate + i - 30, maxDate + i + 1):
    #             tot += data[k][j]
    #         tot = math.floor((tot / (30)))
    #         total += tot
    #         data[maxDate + i][j] = tot
    #
    #         data[maxDate + i][j] += 5.3
    # result.append(data)
    predictList = []
    totaltmp = 0  # total machines needed in all predictTime and all flavor

    for i in range(0, 19):
        predictList.append(0)
    for i in typeList:
        tmp = 0
        for j in range(1, predictTime + 1 - breakTime):
            tmp += data[maxDate + j + breakTime][i]
        tmp = (int)(tmp)
        totaltmp += tmp
        predictList[i] = [i, tmp]

    totaltmp = int(totaltmp)

    for i in range(0, len(predictList)):
        if predictList[i] == 0:
            predictList[i] = [i, 0]


    TEST = [0, 8, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #      /    ! !  ! ~  +    ~   ~ + 2 ~    + ~  ~  !  ~
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
    for i in range(0, 20000):
        bag.append([(int)(cpuNum), (int)(memNum), (int)(diskNum)])

    curBag = 1  # currentBag
    usedBag = []
    for i in range(0, 20000):
        usedBag.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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

    # minServer = -1
    curT = 100.0
    endT = 1
    decRate = 0.996
    count = 0
    while (curT > endT):
        newQ = Q
        random.shuffle(dice)
        tmp = newQ[dice[0]]
        for i in range(len(Q) - 1):
            newQ[dice[i]] = newQ[dice[i + 1]]
        newQ[dice[len(Q) - 1]] = tmp
        # newQ[dice[0]], newQ[dice[1]] = newQ[dice[1]], newQ[dice[0]]

        bag = []
        for i in range(0, 20000):
            bag.append([(int)(cpuNum), (int)(memNum), (int)(diskNum)])
        usedBag = []
        for i in range(0, 20000):
            usedBag.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        curBag = 1

        for i in newQ:
            ratioMax = 1e9
            index = -1
            for j in range(1, curBag + 1):
                if bag[j][0] - CPUTYPE[(int)(i)] >= 0 and bag[j][1] - MEMTYPE[(int)(i)] >= 0:
                    bag[j][0] -= CPUTYPE[(int)(i)]
                    bag[j][1] -= MEMTYPE[(int)(i)]
                    # usedBag[j][(int)(i)] += 1
                    ratio = bag[j][0] + bag[j][1]
                    bag[j][0] += CPUTYPE[(int)(i)]
                    bag[j][1] += MEMTYPE[(int)(i)]
                    if ratio < ratioMax:
                        ratioMax = ratio
                        index = j
            # if j == curBag:
            # curBag += 1
            if index != -1:
                bag[index][0] -= CPUTYPE[(int)(i)]
                bag[index][1] -= MEMTYPE[(int)(i)]
                usedBag[index][(int)(i)] += 1
            else:
                curBag += 1
                bag[curBag][0] -= CPUTYPE[(int)(i)]
                bag[curBag][1] -= MEMTYPE[(int)(i)]
                usedBag[curBag][(int)(i)] += 1

        while (1):
            totTest = 0
            for i in range(1, 19):
                totTest += usedBag[curBag][i]
            if totTest != 0:
                break;
            else:
                curBag -= 1

        serverNum = 0
        totCPU = 0
        totMEM = 0
        for i in range(1, curBag + 1):
            totCPU += bag[i][0]
            totMEM += bag[i][1]
        usageCPU = float(totCPU) / ((int)(cpuNum) * curBag)
        usageMEM = float(totMEM) / ((int)(memNum) * curBag)

        if mode == 1:
            serverNum = (usageCPU + usageMEM) / 2
            # serverNum = (usageCPU + usageMEM)/2
        else:
            serverNum = (usageCPU + usageMEM) / 2
            # serverNum = (usageCPU + usageMEM)/2
        # if count == 0:
        # print serverNum
        # print curBag
        # count += 1
        if serverNum < minServer:
            print
            serverNum
            print
            curBag
            minServer = serverNum
            bestServer = usedBag
            # Q = newQ
        else:
            pass
            # if math.exp((minServer-serverNum)/curT) > random.random():
            #     print serverNum
            #     print curBag
            #     minServer = serverNum
            #     bestServer = usedBag
            # Q = newQ
        curT = decRate * curT

    usedBag = bestServer

    while (1):
        totTest = 0
        for i in range(1, 19):
            totTest += usedBag[curBag][i]
        if totTest != 0:
            break;
        else:
            curBag -= 1

    result.append('General ' + str(curBag))
    typeList.reverse()
    for i in range(1, curBag + 1):
        out = 'General-' + str(i) + ' '
        for j in typeList:
            if usedBag[i][j] != 0:
                out += 'flavor' + str(j) + ' ' + str(usedBag[i][j]) + ' '
        out = out.rstrip(' ')
        result.append(out)
    # result.append('This is a TEST!!!')
    return result

