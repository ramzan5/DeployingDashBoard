from statistics import correlation
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import itertools

# --------------------File Existence Check------------------------------------------------------------
def CheckFileExistence(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")

# print(CheckFileExistence("/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv"))
# ----------------------------------------------------------------------------------------------------

# ------------------Read Data From File ---------------------------------------------------------------

def ReafFile(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")
    DataFrame = pd.read_csv(filePath)
    return DataFrame


# print(ReafFile("/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv"))
# ----------------------------------------------------------------------------------------------------

# ------------------------------Check Missing Data in DataSet -------------------------------------------

def CheckMissingData(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! it is not a DataFrame")
    MissingData = False
    NaNRows = DataFrame.isnull()
    NaNRows = NaNRows.any(axis=1)
    DataFrameWithNaN = DataFrame[NaNRows]
    DataFrameWithNaN.shape[0]
    if DataFrameWithNaN.shape[0] != 0:
        raise Exception("Oops! Some Data is missing in DataSet")
    return MissingData

# -------------------------------------------------------------------------------------------------------

# -----------------------------To Get the Numbers of Column from DataSet
def DisplayNumbersOfColumns(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[1]


DataFrame = ReafFile(
    "/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv")
# print(DisplayNumbersOfColumns(DataFrame))
# -------------------------------------------------------------------------------------------------

# --------------------------------Display the Numbers of rows in DataSet---------------------------
def DisplayNumbersOfRows(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[0]

# print(DisplayNumbersOfRows(DataFrame))
# -------------------------------------------------------------------------------------------------

# ------------------------------Get the Names of Colums from DataSet ------------------------------

def GetColumnsName(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.columns.tolist()

# print(GetColumnsName(DataFrame))

# ---------------------------------------------------------------------------------------------------

# ------------------------------Check the Varience of Columns---------------------------------------

def CheckColumnsVarience(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.var()

# print(CheckColumnsVarience(DataFrame))
# ----------------------------------------------------------------------------------------------------

# -------------------------------Correlation Check of Given Data Set----------------------------------

def CheckCorrelation(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    DataFrameIntegerFloatsColumns = DataFrame.select_dtypes(include=['int64','float64'])
    correlation = {}
    columns = DataFrameIntegerFloatsColumns.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlation[col_a + '__'+ col_b] = pearsonr(DataFrameIntegerFloatsColumns.loc[:,col_a],DataFrameIntegerFloatsColumns.loc[:,col_b])
    FinalResult = pd.DataFrame.from_dict(correlation, orient='index')
    FinalResult.columns = ['PCC', 'p-value']
    return FinalResult.columns

# print(CheckCorrelation(DataFrame))

# ------------------------------------------------------------------------------------------------------

# --------------------------Check the Plauseability of Given DataSet------------------------------------

def CheckPlauseabilityDataSet(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    
    InvalidInstantValue = sum(DataFrame['instant'] < 0)
    # InvalidDateValue = sum(DataFrame['dteday'] )
    InvalidSeasonValue = sum(
        (DataFrame['season'] != 1) & (DataFrame['season'] != 2) & 
        (DataFrame['season'] != 3) & (DataFrame['season'] != 4))
    InvalidYearValue = sum( (DataFrame['yr'] != 0) & (DataFrame['yr'] != 1) )
    InvalidMonthValue = sum(
        (DataFrame['mnth'] != 1) & (DataFrame['mnth'] != 2)&
        (DataFrame['mnth'] != 3) & (DataFrame['mnth'] != 4)&
        (DataFrame['mnth'] != 5) & (DataFrame['mnth'] != 6)&
        (DataFrame['mnth'] != 7) & (DataFrame['mnth'] != 8)&
        (DataFrame['mnth'] != 9) & (DataFrame['mnth'] != 10)&
        (DataFrame['mnth'] != 11) & (DataFrame['mnth'] != 12)
        )
    InvalidHolidayValue = sum(
        (DataFrame['holiday'] != 0) & (DataFrame['holiday'] != 1))
    InvalidWeekDayValue = sum(
        (DataFrame['weekday'] != 0) & (DataFrame['weekday'] != 1)&
        (DataFrame['weekday'] != 2) & (DataFrame['weekday'] != 3)&
        (DataFrame['weekday'] != 4) & (DataFrame['weekday'] != 5)&
        (DataFrame['weekday'] != 6)
    )
    InvalidWorkingDayValue = sum(
        (DataFrame['workingday'] != 0) & (DataFrame['workingday'] != 1)
    )
    InvalidWeathersitValue = sum(
        (DataFrame['weathersit'] != 1) & (DataFrame['weathersit'] != 2)&
        (DataFrame['weathersit'] != 3) & (DataFrame['weathersit'] != 4)
    )
    InvalidTempValue = sum(
        (DataFrame['temp'] > 1) & (DataFrame['temp'] < 0)
    )
    InvalidAtempValue = sum(
        (DataFrame['atemp'] > 1) & (DataFrame['atemp'] < 0)
    )
    InvalidHumidityValue = sum(
        (DataFrame['hum'] > 1) & (DataFrame['hum'] < 0)
    )
    InvalidWindSpeedValue = sum(
        (DataFrame['windspeed'] > 1) & (DataFrame['windspeed'] < 0)
    )
    InvalidCasualValue = sum(DataFrame['casual'] < 0)
    InvalidRegisteredValue = sum(DataFrame['registered'] < 0)
    InvalidCntValue = sum(DataFrame['cnt'] < 0)
    
    PlausibilityValues = {
        'Invalid Value in Instant Column': InvalidInstantValue,
        'Invalid Value in Season Column' : InvalidSeasonValue,
        'Invalid Value in Year Column'   : InvalidYearValue,
        'Invalid Value in Month Column'  : InvalidMonthValue,
        'Invalid Value in Holiday Column': InvalidHolidayValue,
        'Invalid Value in Weekday Column': InvalidWeekDayValue,
        'Invalid Value in Workingday Column': InvalidWorkingDayValue,
        'Invalid Value in WeatherSit Column': InvalidWeathersitValue,
        'Invalid Value in Temp Column '  : InvalidTempValue,
        'Invalid Value in Atemp Column'  : InvalidAtempValue,
        'Invalid Value in Humidity Column': InvalidHumidityValue,
        'Invalid Value in Windspeed Column': InvalidWindSpeedValue,
        'Invalid Value in Casual Column' : InvalidCasualValue,
        'Invalid Value in Registered Column' : InvalidRegisteredValue,
        'Invalid Value in Cnt Column' : InvalidCntValue
    } 

    IncorrectValue = False
    for key in PlausibilityValues:
        if PlausibilityValues[key] > 0:
            IncorrectValue = True
            raise Exception("Oops There is incorrect value are missing in given Dataset")
            break

    return IncorrectValue


# DataFrame = pd.read_csv("DataSet/Bike-Sharing-Dataset/day.csv")
# CheckPlauseabilityDataSet(DataFrame)

# ---------------------------------------------------------------------------------------------------------------
