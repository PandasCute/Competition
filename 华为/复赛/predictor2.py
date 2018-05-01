import random

def exSmooth(maxDate, data, alpha=0.8):
    maxDate = maxDate
    S2_1 = []
    S2_2 = []

    S2_1_new1 = []
    dataEx = []

    for i in range(1, 19):
        x = 0
        for n in range(1, 4):
            x += float(data[n][i])
        x /= 3
        S2_1_empty = []
        S2_1_empty.append(x)
        S2_1.append(S2_1_empty)
        S2_2.append(S2_1_empty)

    for i in range(0, 18):
        S2_1_new = [[]] * 18

        for j in range(maxDate):
            if j == 0:
                S2_1_new[i].append(float(alpha) * float(data[j + 1][i + 1]) + (1 - float(alpha)) * float(S2_1[i][j]))
            else:
                S2_1_new[i].append(
                    float(alpha) * float(data[j + 1][i + 1]) + (1 - float(alpha)) * float(S2_1_new[i][j - 1]))

        S2_1_new1.append(S2_1_new[i])


    S2_2_new1 = []

    info_MSE = []
    for i in range(0, 18):
        S2_2_new = [[]] * 18
        MSE = 0
        for j in range(0, maxDate):
            if j == 0:
                S2_2_new[i].append(float(alpha) * float(S2_1_new1[i][j]) + (1 - float(alpha)) * float(S2_2[i][j]))
            else:
                S2_2_new[i].append(float(alpha) * float(S2_1_new1[i][j]) + (1-float(alpha))*float(S2_2_new[i][j - 1]))

        S2_2_new1.append(S2_2_new[i])
    return S2_1_new1, S2_2_new1

def predict_vm(ecs_lines, input_lines):


    result = []

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

    typeList.sort()

    currentTime = maxDate + 1
    startDate = cal(startYear, startMonth, startDay)
    endDate = cal(endYear, endMonth, endDay)
    startDate -= startDateOrigin
    endDate -= startDateOrigin
    predictTime = endDate - startDate
    breakTime = startDate - maxDate
    predictTime += breakTime


    '''
    We store the predicted data in data[maxdate][]-data[maxdate+predictTime][]
    '''
    #
    # fig = plt.figure()
    for i in range(1, 19):
        for j in range(1, maxDate + 1):
            flavorNum[i][j] = data[j][i]

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
                data[j][i] = int(data[j][i])

    alpha = 0.1
    # print(data)
    # print(maxDate, predictTime)
    # print(data)
    for i in range(1, predictTime+1):

        S2_2_new1, S2_1_new1 = exSmooth(maxDate+i-1, data, alpha)

        for j in range(0, 18):
            At = (float(S2_1_new1[j][len(S2_1_new1[j]) - 1]) * 2 -
                  float(S2_2_new1[j][len(S2_2_new1[j]) - 1]))
            Bt = (float(alpha)/(1 - float(alpha)) * (
                float(S2_1_new1[j][len(S2_1_new1[j]) - 1]) -
                float(S2_2_new1[j][len(S2_2_new1[j]) - 1])
            ))
            sum1 = At + Bt
            if (sum1 - int(sum1)) > 0.5:
                sum1 += 1
            if sum1 < 0:
                sum1 = 0
            data[maxDate+i][j+1] = sum1

    # print(data)
    # total = 0
    # print(maxDate, predictTime)

    predictList = []
    totaltmp = 0  # total machines needed in all predictTime and all flavor

    for i in range(0, 19):
        predictList.append(0)
    # print(typeList)
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


    TEST = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,   0, 0, 0, 0]
    #      /    ! !  ! ~  +    ~   ~ + 2 ~    + ~  ~  !  ~
    for i in range(1, len(predictList)):
        # if predictList[i][1] != 0:
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
            Q.append(predictList[i][0])
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