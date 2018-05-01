import math
import random


def predict_vm(ecs_lines, input_lines):
    # Do your work from here#

    result = []
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

    avg = []
    var = []
    for i in range(0, 16):
        avg.append(0)
        var.append(0)
    for i in range(1, 16):
        tmp = 0
        for j in range(1, maxDate + 1):
            tmp += float(data[j][i])
        avg[i] = float(tmp) / float(maxDate)

    for i in range(1, 16):
        tmp = 0
        for j in range(1, maxDate + 1):
            tmp += (data[j][i] - avg[i]) ** 2
        var[i] = float(tmp) / float(maxDate)

    datac = []
    datac.extend(data)
    for i in range(1, 16):
        for j in range(1, maxDate + 1):
            datac[j][i] = (float(data[j][i]) - float(avg[i])) / float(var[i])

    trainSize = (int)(maxDate / 3)
    length = maxDate - trainSize
    # print var
    # print avg

    for s in range(0, 16):
        w = []
        wbest = []
        xMat = []
        yMat = []
        yPre = []

        for i in range(0, length + 1):
            w.append(random.random())
            wbest.extend(w)
            xMat.append([])
            yMat.append(0)
            yPre.append(0)

        for i in range(1, length + 1):
            xMat[i].append(0)
            yMat[i] = datac[i + trainSize][s]
            for j in range(0, trainSize):
                xMat[i].append(datac[i + j][s])
        minLoss = 1e9
        for i in range(1, trainSize + 1):
            wtmp = -2
            while wtmp <= 2:
                w[i] = wtmp
                loss = 0
                for j in range(1, length + 1):
                    tmpp = 0
                    for k in range(1, trainSize + 1):
                        tmpp += float(w[k]) * float(xMat[j][k])
                    yPre[j] = tmpp
                    loss += (float(yPre[j]) - float(yMat[j])) ** 2
                if loss < minLoss:
                    wbest[i] = wtmp
                    w[i] = wtmp
                    minLoss = loss
                wtmp += 0.01
            w[i] = wbest[i]
        for i in range(maxDate, maxDate + predictTime + 1):
            tmp = 0
            for j in range(1, trainSize + 1):
                tmp += float(wbest[j]) * float(datac[i - trainSize][s])
            # datac[i][s] = tmp
            tmp = float(tmp) * float(var[s]) + float(avg[s])
            data[i][s] = max((int)(tmp), 0)

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
        # newQ[dice[0]], newQ[dice[1]] = newQ[dice[1]], newQ[dice[0]]

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
        usageCPU = 1 - float(totCPU) / ((int)(cpuNum) * curBag)
        usageMEM = 1 - float(totMEM) / ((int)(memNum) * curBag)

        if mode == 1:
            serverNum = curBag + usageCPU
        else:
            serverNum = curBag + usageMEM
        if serverNum < minServer:
            minServer = serverNum
            bestServer = usedBagimport
import random

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#

    result = []
    cnt = 0
    flag = 0
    mode = 0 # mode 1 = optimize CPU, mode 2 = optimize MEM
    typeList = []
    trainData = []
    dayList = []
    data = []

    RUNYEAR = 0 # wheather a year is
    CPUTYPE = [0,1,1,1,2,2,2,4,4,4,8,8,8,16,16,16]
    MEMTYPE = [0,1,2,4,2,4,8,4,8,16,8,16,32,16,32,64]
    DAYNUM2 = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    DAYNUM1 = [0,31,27,31,30,31,30,31,31,30,31,30,31]
    YEARS = [0,365,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366]

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
            for i in range(1,(int)(month)):
                datetmp += DAYNUM1[i]
            datetmp += (int)(day)
        if RUNYEAR == 1:
            datetmp = 0
            for i in range(1,(int)(month)):
                datetmp += DAYNUM2[i]
            datetmp += (int)(day)
        for i in range (1,(int)(year)-2000+1):
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
        values[1] = values[1].replace('flavor','')
        if (int)(values[1]) >= 16:
            cnt -= 1
            continue
        create = values[2].split(" ")
        year = values[2][0:4]
        month = values[2][5:7]
        day = values[2][8:10]
        date = cal(year, month, day)
        if cnt == 1:
            startDate = date-1
        date = date-startDate
        dayList.append(date)
        trainData.append([(int)(values[1]), date, cnt]) # trainData = [[type1, date1, cnt1],[]]




    maxDate = dayList[-1] # Last day of train data, i.e. the day before 1st day to be predicted
    data.append(0)
    for i in range(1, maxDate+20):
        data.append([i,0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0])
    for i in range(1, cnt+1):
        date = trainData[i][1]
        flavor = trainData[i][0]
        data[date][flavor] += 1 # data = [[day1, type1num,..., type 15num],[day2,type1num,...],[]]
    #result.append(data)



    for index, item in enumerate(input_lines):
        print("index of input data")
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
            tmp = item.split(' ')[0].replace('flavor','')
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





    flavorNum=[]
    for i in range (0,16):
        flavorNum.append(0)
    for i in range (1,16):
        flavorNum[i] = [ 0 for j in range(0,maxDate+1)]

    for i in range (1,16):
        for j in range (1,maxDate+1):
            flavorNum[i][j] = data[j][i]
    #result.append(flavorNum)

    typeList.sort()
    #result.append(typeList)

    currentTime = maxDate + 1
    startDate = cal(startYear, startMonth, startDay)
    endDate = cal(endYear, endMonth, endDay)
    predictTime = endDate - startDate
    #result.append(predictTime)
    # Prediction Process
    '''
    We store the predicted data in data[maxdate][]-data[maxdate+predictTime][]
    '''


    avg = []
    var = []
    for i in range(0,16):
        avg.append(0)
        var.append(0)
    for i in range(1,16):
        tmp = 0
        for j in range(1,maxDate+1):
            tmp += float(data[j][i])
        avg[i] = float(tmp)/float(maxDate)

    for i in range(1,16):
        tmp = 0
        for j in range(1,maxDate+1):
            tmp += (data[j][i]-avg[i])**2
        var[i] = float(tmp)/float(maxDate)

    datac = []
    datac.extend(data)
    for i in range(1,16):
        for j in range(1,maxDate+1):
            datac[j][i] = (float(data[j][i])-float(avg[i])) / float(var[i])


    trainSize = (int)(maxDate/3)
    length = maxDate - trainSize
    #print var
    #print avg

    for s in range(0,16):
        w = []
        wbest = []
        xMat = []
        yMat = []
        yPre = []

        for i in range(0,length+1):
            w.append(random.random())
            wbest.extend(w)
            xMat.append([])
            yMat.append(0)
            yPre.append(0)

        for i in range(1,length+1):
            xMat[i].append(0)
            yMat[i] = datac[i+trainSize][s]
            for j in range(0,trainSize):
                xMat[i].append(datac[i+j][s])
        minLoss = 1e9
        for i in range(1,trainSize+1):
            wtmp = -2
            while wtmp <= 2:
                w[i] = wtmp
                loss = 0
                for j in range(1,length+1):
                    tmpp = 0
                    for k in range(1,trainSize+1):
                        tmpp += float(w[k])*float(xMat[j][k])
                    yPre[j] = tmpp
                    loss += (float(yPre[j])-float(yMat[j]))**2
                if loss < minLoss:
                    wbest[i] = wtmp
                    w[i] = wtmp
                    minLoss = loss
                wtmp += 0.01
            w[i] = wbest[i]
        for i in range(maxDate,maxDate+predictTime+1):
            tmp = 0
            for j in range(1,trainSize+1):
                tmp += float(wbest[j])*float(datac[i-trainSize][s])
            #datac[i][s] = tmp
            tmp = float(tmp)*float(var[s])+float(avg[s])
            data[i][s] = max((int)(tmp),0)


    predictList = []
    totaltmp = 0 # total machines needed in all predictTime and all flavor
    for i in range (0,16):
        predictList.append(0)
    for i in typeList:
        tmp = 0
        for j in range(1,predictTime+1):
            tmp += data[maxDate+j][i]
        tmp = (int)(tmp)
        totaltmp += tmp
        predictList[i] = [i,tmp]



    result.append((int)(totaltmp))  #predictList stores data like [0, [1,flavor1Num],[2,flavor2Num],...]
    for i in typeList:
        result.append('flavor'+str(predictList[i][0])+' '+str((int)(predictList[i][1])))
    result[-1] += '\n'

    copy = predictList
    reTypeList = typeList
    reTypeList.reverse()

    bag = []
    for i in range (0,10000):
        bag.append([(int)(cpuNum),(int)(memNum),(int)(diskNum)])

    curBag = 1 # currentBag
    usedBag = []
    for i in range (0,10000):
        usedBag.append([i,0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0])

    Q = []

    for i in reTypeList:
        cnt = 1
        while(cnt <= (int)(predictList[i][1])):
            Q.append(predictList[i][0]);
            cnt += 1
    #result.append(Q)
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
        # newQ[dice[0]], newQ[dice[1]] = newQ[dice[1]], newQ[dice[0]]

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
        usageCPU = 1 - float(totCPU) / ((int)(cpuNum) * curBag)
        usageMEM = 1 - float(totMEM) / ((int)(memNum) * curBag)

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
    # result.append(typeList)
    for i in range(1, curBag + 1):
        out = str(i) + ' '
        for j in typeList:
            if usedBag[i][j] != 0:
                out += 'flavor' + str(j) + ' ' + str(usedBag[i][j]) + ' '
        out = out.rstrip(' ')
        result.append(out)
    # result.append('This is a TEST!')
    return result