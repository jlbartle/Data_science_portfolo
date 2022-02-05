import csv

def delimit_pulse():
    with open('E:\OneDrive\DataScience\IST 659\Project\SamsungHealth\pulse.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if (count == 0):
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\' + row[0] + '.sql' , 'w+')
                count += 1
                continue
            elif (count == 1):
                count += 1
                continue
            else:
                outPut.write('insert into PULSE (PulseDT, Pulse) VALUES (\'' + row[0] + '\', ' + row[6] +')\n')

##read forecast
def delimit_cal_burned():
    with open('E:\OneDrive\DataScience\IST 659\Project\SamsungHealth\cal_burned.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if (count == 0):
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\' + row[0] + '.sql' , 'w+')
                count += 1
                continue
            elif (count == 1):
                count += 1
                continue
            else:
                outPut.write('insert into CAL_BURNED (Cal_BurnedDT, Rest_Cal, Active_Cal) VALUES ' +
                             ' (\'' + row[3] + '\', ' + row[6] + ', ' + row[7] + ')\n')


def delimit_steps():
    with open('E:\OneDrive\DataScience\IST 659\Project\SamsungHealth\steps.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if (count == 0):
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\' + row[0] + '.sql' , 'w+')
                count += 1
                continue
            elif (count == 1):
                count += 1
                continue
            else:
                outPut.write('insert into STEPS(StepsDT, Steps_Count, Speed, Cal_Burned) VALUES ' +
                             ' (\'' + row[0] + '\', ' + row[4] + ', ' + row[6] + ', ' + row[12] + ')\n')


def delimit_food():
    with open('E:\OneDrive\DataScience\IST 659\Project\SamsungHealth\\food_info.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if (count == 0):
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\' + row[0] + '.sql' , 'w+')
                count += 1
                continue
            elif (count == 1):
                count += 1
                continue
            elif row[27].find("'") != -1:
                row[27] = row[27].replace("'", "''")
                x = 31
                while x > 0:
                    if row[x] is None or row[x] == '':
                        row[x] = '0'
                    x -= 1
                outPut.write('insert into FOOD(Name, Serving_Info, Calorie, Carb, Fat, Protein, Fiber, ' +
                             'Cholesterol, VA, Calcium, VC, Sat_Fat, MonoSat_Fat, Potassium, Sodium, Sugars, Iron)' +
                             'VALUES (\'' + row[27] + '\', \'' + row[8] + '\', ' + row[28] + ', ' + row[4] + ', ' +
                             row[0] + ', ' + row[10] + ', ' + row[13] + ', ' + row[11] + ', ' + row[15] + ', ' +
                             row[14] + ', ' + row[18] + ', ' + row[22] + ', ' + row[21] + ', ' + row[1] + ', ' +
                             row[23] + ', ' + row[31] + ', ' + row[29] + ')\n')
            else:
                x = 31
                while x > 0:
                    if row[x] is None or row[x] == '':
                        row[x] = '0'
                    x -= 1
                outPut.write('insert into FOOD(Name, Serving_Info, Calorie, Carb, Fat, Protein, Fiber, ' +
                             'Cholesterol, VA, Calcium, VC, Sat_Fat, MonoSat_Fat, Potassium, Sodium, Sugars, Iron)' +
                             'VALUES (\'' + row[27] + '\', \'' + row[8] + '\', ' + row[28] + ', ' + row[4] + ', ' +
                             row[0] + ', ' + row[10] + ', ' + row[13] + ', ' + row[11] + ', ' + row[15] + ', ' +
                             row[14] + ', ' + row[18] + ', ' + row[22] + ', ' + row[21] + ', ' + row[1] + ', ' +
                             row[23] + ', ' + row[31] + ', ' + row[29] + ')\n')

def delimit_meals():
    with open('E:\OneDrive\DataScience\IST 659\Project\SamsungHealth\meals.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if (count == 0):
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\' + row[0] + '.sql' , 'w+')
                count += 1
                continue
            elif (count == 1):
                count += 1
                continue
            elif row[12].find("'") != -1:
                row[12] = row[12].replace("'", "''")
                outPut.write('INSERT into MEALS(FoodID, MealDT, Amount) VALUES ' +
                             '((SELECT(FoodID) FROM FOOD where Name = \'' + row[12] + '\'), \'' +
                             row[2] + '\', ' + row[0] + ')\n')
            else:
                outPut.write('INSERT into MEALS(FoodID, MealDT, Amount) VALUES ' +
                             '((SELECT(FoodID) FROM FOOD where Name = \'' + row[12] + '\'), \'' +
                             row[2] + '\', ' + row[0] + ')\n')



delimit_pulse()

