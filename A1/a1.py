'''
COMP9321 Assignment One Code Template 2019T1
Name:Luo Pan
Student ID:z5192086
'''

import pandas as pd
import csv
import re


def q1():
    '''
    Put Your Question 1's code in this function
    '''

    with open("accidents_2017.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            for j in range(0, len(row)):
                row[j] = row[j].strip()
                if j != 8 and j != 9 and j != 10 and j != 12:
                    row[j] = row[j].title()
                    row[j] = re.sub(r"\sDe\b", " de", row[j])
                    row[j] = re.sub(r"\sL'\b", " l'", row[j])
                    row[j] = re.sub(r"\sLa\b", " la", row[j])
                    row[j] = re.sub(r"\sD'\b", " d'", row[j])

                if ' ' in row[j]:
                    row[j] = '"' + row[j] + '"'

            print(' '.join(row))
            if i >= 10:
                break


def q2():
    '''
    Put Your Question 2's code in this function 
    '''

    df = pd.read_csv("accidents_2017.csv")
    df = df[~df["District Name"].isin(["Unknown"])]
    df = df[~df["Neighborhood Name"].isin(["Unknown"])]
    df = df[~df["District Name"].isin(["-"])]
    df = df[~df["Neighborhood Name"].isin(["-"])]
    df = df.dropna(axis=0)

    row = df.values.tolist()
    for j in range(0, len(row) - 1):
        for i in range(0, len(row[j]) - 1):
            if j != 8 and j != 9 and j != 10 and j != 12:
                if isinstance(row[j][i], str):
                    row[j][i] = row[j][i].strip()
                    row[j][i] = row[j][i].title()
                    row[j][i] = re.sub(r"\sDe\b", " de", row[j][i])
                    row[j][i] = re.sub(r"\sL'\b", " l'", row[j][i])
                    row[j][i] = re.sub(r"\sLa\b", " la", row[j][i])
                    row[j][i] = re.sub(r"\sD'\b", " d'", row[j][i])

    df2 = pd.DataFrame(row)
    df2.columns = ["Id", "District Name", "Neighborhood Name", "Street", "Weekday", "Month", "Day", "Hour",
                   "Part of the day", "Mild injuries", "Serious injuries", "Victims", "Vehicles involved", "Longitude",
                   "Latitude"]

    df2.to_csv("result_q2.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)


def q3():
    '''
    Put Your Question 3's code in this function 
    '''
    df = pd.read_csv("result_q2.csv", usecols=[0, 1])
    df = df.drop_duplicates('Id')
    df = df.groupby(df['District Name']).count().sort_values("Id", ascending=False)
    df = df.rename(columns={'Id': 'Total numbers of accidents'})
    pd.set_option('display.width', None)
    df.reset_index()
    df.to_csv("q3.csv")

    with open("q3.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            for j in range(0, len(row)):
                if ' ' in row[j]:
                    row[j] = '"' + row[j] + '"'
            print(' '.join(row))


def q4():
    '''
    Put Your Question 4's code in this function 
    '''
    df = pd.read_csv("air_stations_Nov2017.csv", usecols=[0, 4])
    df = df.to_json(orient='records')
    print(df)

    df2 = pd.read_csv("air_quality_Nov2017.csv", index_col=0)
    df2 = df2[~df2["Air Quality"].isin(["Good"])]
    df2 = df2[~df2["Air Quality"].isin(["--"])]
    df2 = df2.head(10)
    df2.to_csv("q4.csv")

    with open("q4.csv") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            for j in range(0, len(row)):
                if ' ' in row[j]:
                    row[j] = '"' + row[j] + '"'
            print(' '.join(row))

    df3 = pd.read_csv("result_q2.csv")
    l11 = []
    l12 = []
    with open("result_q2.csv") as f1:
        reader = csv.reader(f1)
        day_column = [row[6] for row in reader]
        del day_column[0]
        for i in day_column:
            a = int(i)
            l11.append(a)
    with open("result_q2.csv") as f2:
        reader = csv.reader(f2)
        hour_column = [row[7] for row in reader]
        del hour_column[0]
        for i in hour_column:
            a = int(i)
            l12.append(a)

    with open("air_quality_Nov2017.csv") as f2:
        reader = csv.reader(f2)
        column = [row[13] for row in reader]

    l1 = []
    l2 = []
    result = []
    for i in range(1, len(column)):
        day1 = int(column[i][0:2])
        l1.append(day1)
        hour2 = int(column[i][11:-3])
        l2.append(hour2)

    for i in range(0, len(l11)):
        for j in range(0, len(l1)):
            if l11[i] == l1[j] and l12[i] == l2[j]:
                result.append(i)

    df3 = df3.iloc[result, :]
    df4 = pd.read_csv("air_stations_Nov2017.csv")
    df5 = pd.read_csv("air_quality_Nov2017.csv")
    df5 = df5[~df5["Air Quality"].isin(["Good"])]
    df6 = pd.merge(df4, df5, on=["Station"], how="inner")
    pd.set_option('display.width', None)
    df6 = pd.DataFrame(df6)


    df3 = df3[df3["District Name"].isin(df6["District Name"])]
    df3 = df3[df3["Neighborhood Name"].isin(df6["Neighborhood Name"])]
    df3 = df3[df3["Month"].isin(["November"])]


    '''
    df6["Generated"] = pd.to_datetime(df6["Generated"], format='%d/%m/%Y %H:%M', errors="ignore")

    day = pd.DataFrame(df6["Generated"].dt.day).rename(columns={"Generated": "Day"})
    hour = pd.DataFrame(df6["Generated"].dt.hour).rename(columns={"Generated": "Hour"})

    df3_dict = df3[["Day", "Hour"]].to_dict(orient='records')
    df_dayhour = [day, hour]
    df_dayhour = pd.concat(df_dayhour, axis=1).to_dict(orient='records')

    l = []
    for i in range(len(df3_dict)):
        for j in range(len(df_dayhour)):
            if df3_dict[i] == df_dayhour[j]:
                l.append(i)
    '''
    df3 = df3.drop_duplicates("Id")
    df3.to_csv("result_q4.csv")


def q5():
    '''
    Bonus Question(Optional).
    Put Your Question 5's code in this function.
    '''
    pass



