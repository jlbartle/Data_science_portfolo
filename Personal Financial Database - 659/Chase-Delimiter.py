import csv



def delimit_credit():
    with open('E:\OneDrive\DataScience\IST 659\Project\Chase\Credit2019.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')


        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if count == 0:
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\\OutPut\\Credit2019.sql' , 'w+')
                count += 1
                continue
            elif (row[3] == 'Shopping'):
                if row[2].find("'") != -1:
                    row[2] = row[2].replace("'", "''")
                outPut.write('INSERT into CREDIT(TransactionDT, Description, CategoryID, Amount) VALUES (\'' + row[0]
                            + '\', \'' + row[2] + '\', 1,' + row[5] + ')\n')
                continue
            elif (row[3] == '' or row[3] is None):
                if row[2].find("'") != -1:
                    row[2] = row[2].replace("'", "''")
                outPut.write('INSERT into CREDIT(TransactionDT, Description, CategoryID, Amount) VALUES (\'' + row[0]
                            + '\', \'' + row[2] + '\', 2,' + row[5] + ')\n')
                continue
            else:
                if row[2].find("'") != -1:
                    row[2] = row[2].replace("'", "''")
                outPut.write('INSERT into CREDIT(TransactionDT, Description, CategoryID, Amount) VALUES (\'' + row[0]
                            + '\', \'' + row[2] + '\', (SELECT(CategoryID) FROM EXPENSE_CATEGORY where Description = \''
                            + row[3] + '\' ), ' + row[5] + ')\n')


def delimit_checking():
    with open('E:\OneDrive\DataScience\IST 659\Project\Chase\\Checking.csv') as csv_file:
        forecast_reader = csv.reader(csv_file, delimiter=',')

        ##initial pointers
        count = 0

        ##Full file read
        for row in forecast_reader:
            if count == 0:
                outPut = open('E:\OneDrive\DataScience\IST 659\Project\OutPut\\Checking.sql' , 'w+')
                count += 1
                continue
            else:
                if row[2].find("'") != -1:
                    row[2] = row[2].replace("'", "''")
                outPut.write('INSERT into CHECKING(TransactionDT, Description, Amount) VALUES (\'' + row[1]
                            + '\', \'' + row[2] + '\',' + row[3] + ')\n')



delimit_credit()
delimit_checking()
