import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy import stats
import os as os
import json
from statsmodels.tsa.stattools import adfuller
from pandas.errors import ParserError
import re
import pickle
from PyPDF2 import PdfFileMerger, PdfFileReader,PdfMerger,PdfReader
from appp import genai
import string

from datetime import date, datetime

import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import chi2_contingency
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
import time

warnings.filterwarnings("ignore")


def output_df_to_pdf(pdf, df):
    # A cell is a rectangular area, possibly framed, which contains some text
    # Set the width and height of cell
    table_cell_width = 25
    table_cell_height = 6
    # Select a font as Arial, bold, 8
    pdf.set_font('Arial', 'B', 8)

    # Loop over to print column names
    cols = df.columns
    for col in cols:
        pdf.cell(table_cell_width, table_cell_height, col, align='C', border=1)
    # Line break
    pdf.ln(table_cell_height)
    # Select a font as Arial, regular, 10
    pdf.set_font('Arial', '', 10)
    # Loop over to print each data in the table
    for row in range(len(df.columns)):
        for col in range(len(df)):
            # print(df.iloc[row][col])
            value = str("{:.2f}".format(df.iloc[row][col]))  # str(getattr(row, col))
            pdf.cell(table_cell_width, table_cell_height, value, align='C', border=1)
        pdf.ln(table_cell_height)


def subsets(numbers):  # prints all possible subsets of an array
    if numbers == []:
        return [[]]
    x = subsets(numbers[1:])
    return x + [[numbers[0]] + y for y in x]


def ind(categorical_superset):  # function to help to give index as column names
    l = []
    for i in categorical_superset:
        if len(i) == 1:
            for j in i:
                l.append(j)
        else:
            l.append(i)
    return l


# Anova test
def Ftest(df, categorical, numerical):
    # input : dataframe,array of categorical column names, array of numerical column names
    categorical_superset = list(filter(None, subsets(categorical)))
    # F_values = [x[:] for x in [[0] * len(numerical)] * len(categorical_superset)]
    F_values = np.empty((len(categorical_superset), len(numerical)), dtype='int')
    F_alpha = 0.05
    index = []
    counter1 = 0
    # print(categorical_superset)
    for k in range(0, len(numerical)):
        for j in range(0, len(categorical_superset)):
            l = np.array(categorical_superset[j])
            df2 = df[l].astype(str)
            # print(l)
            df2['category'] = df2[categorical_superset[j]].agg('_'.join, axis=1)
            df2.drop(l, axis=1, inplace=True)
            # print(df2.head(3))
            # print("---------------")
            # l = np.append(l, numerical[k])
            df2[numerical[k]] = df[numerical[k]]
            # print(df2.head(3))
            # print("---------------")
            grouper = df2.groupby('category')
            df3 = pd.concat([pd.Series(v.iloc[:, 1].tolist(), name=k) for k, v in grouper], axis=1)
            # print(df3.head(3))
            # df3.boxplot()
            # F,p = stats.f_oneway(*(df3[col] for col in df3.columns))
            if len(df3.columns) > 1:
                data = [df3[col].dropna() for col in df3]
                # print(data)
                Fa, p = stats.f_oneway(*data)
                # print(F,p)
                if p <= 0.05 and p > 0:
                    F_values[j][k] = 1
                else:
                    F_values[j][k] = 0
            else:
                F_values[j][k] = 0
            # print(categorical_superset[j], "::", numerical[k], "::::", p,":::",F_values[j][k])
            # sleep(1000)
    # print(type(F_values))
    df = pd.DataFrame(F_values, columns=numerical)
    df['Categorical_combination'] = categorical_superset
    df.set_index('comb_' + df.index.astype(str))
    return (df)


def chi_square_test(df, categorical):  # input : dataframe,array of categorical column names
    le = LabelEncoder()
    le.fit(categorical)
    comb = list(combinations(categorical, 2))
    comb2 = []
    for i, j in comb:
        comb2.append(le.transform([i, j]))
    n = len(categorical)
    p_values = [x[:] for x in [[0] * n] * n]
    for i in range(0, len(comb)):
        contigency = pd.crosstab(df[comb[i][0]], df[comb[i][1]])
        # sns.heatmap(contigency, annot=True, cmap="YlGnBu")
        # Chi-square test of independence.
        c, p, dof, expected = chi2_contingency(contigency)
        ind_one = comb2[i][0]
        ind_two = comb2[i][1]
        # print("ind", ind_one, ind_two)
        if p <= 0.5:
            p_values[ind_one][ind_two] = 0
            p_values[ind_two][ind_one] = 0
        else:
            p_values[ind_one][ind_two] = 1  # Significant
            p_values[ind_two][ind_one] = 1
    # print(p_values)
    df = pd.DataFrame(p_values, columns=categorical, index=categorical)
    return (df, le)


def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def test_significance(df, categorical, numerical):
    # %matplotlib inline
    print("Starting test_significance function...")
    plt.rcParams['figure.figsize'] = [10, 10]
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    matplotlib.use('Agg')
    filename = "multi.pdf"
    pp = PdfPages(filename)
    # pp.set_font('Arial', 'B', 16)
    dataplot2 = sns.heatmap(df[numerical].corr(), linecolor='white', linewidths=1.3, cmap='Reds', vmin=0, vmax=1,
                            square=True)
    plt.title('Correlation plot for numerical variables 0-Not_Significant 1-Significant', fontweight="bold")
    plt.savefig(pp, format='pdf')
    
    plt.close()

    print(bold_start, "\nRelationship between Numerical and Categorical Variables", bold_end)
    print(categorical)
    categorical_superset = list(filter(None, subsets(categorical)))

    print("Categorical superset:", categorical_superset)

    F_values2 = Ftest(df, categorical, numerical)
    dataplot4 = sns.heatmap(F_values2.loc[:, F_values2.columns != 'Categorical_combination'], linecolor='white',
                            linewidths=1, cmap='Reds', vmin=0, vmax=1)
    plt.title('Anova test for combination of categorical & numerical 0-Not_Significant 1-Significant',
              fontweight="bold")
    plt.savefig(pp, format='pdf')
    plt.close()

    F_values = F_values2[['Categorical_combination']]
    F_values['row_num'] = F_values2.reset_index().index
    # print(F_values.columns)
    df_list = split_dataframe(F_values, 50)
    print(df_list)
    for table in df_list:
        # plt.figure(figsize=(8.27, 11.69))
        tab = plt.table(cellText=table.values, colLabels=table.columns, loc='center',
                        fontsize=3.5, edges='open')
        tab.auto_set_font_size(False)
        tab.auto_set_column_width(col=list(range(len(table.columns))))
        plt.axis('off')
        plt.savefig(pp, format='pdf')
        plt.clf()
        print(".")
    # tab = plt.table(cellText=F_values.values, colLabels="Categorical_combination", cellLoc='center', loc='center')
    # tab.auto_set_font_size(True)
    # plt.savefig(pp, format='pdf')
    plt.close()
    # tab.set_fontsize(30)
    # tab.scale(0.7, 2.5)
    # pdf.savefig(tab)

    # output_df_to_pdf(pp, df[numerical].corr())
    # output_df_to_pdf(pp, F_values)

    p_values, le = chi_square_test(df.sample(n=10), categorical)
    # cmap = sns.color_palette("pastel")
    dataplot3 = sns.heatmap(p_values, linecolor='white', linewidths=1.3, cmap='Reds', vmin=0, vmax=1, square=True)
    plt.title('Chi-square test for categorical variables 0-Not_Significant 1-Significant', fontweight="bold")
    plt.savefig(pp, format='pdf')
    plt.close()

    F_values = F_values2.loc[:, F_values2.columns != 'Categorical_combination']
    indices = ind(categorical_superset)
    cat_num_df = pd.DataFrame(F_values, columns=numerical)
    cat_num_df.index = indices
    # zx = 0
    # print(cat_num_df)
    for i in range(0, F_values.shape[0]):
        for j in range(0, F_values.shape[1]):
            if F_values.iloc[i, j] == 1:  # Significant
                l = np.array(indices[i])
                l = np.append(l, numerical[j])
                df2 = df[l]
                df2.boxplot(column=numerical[j], by=indices[i], rot=45, vert=False)
                plt.title(numerical[j] + ' and ' + str(indices[i]), fontweight="bold")

    fig_nums = plt.get_fignums()
    # print(plt.get_fignums())
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')

    # d = pp.infodict()
    # d['Title'] = 'Cornea-EDA'
    # d['Author'] = 'Kaltics'
    # d['Subject'] = 'Multi-dimensional & combinational EDA'
    # d['Keywords'] = 'Statistics hypothesis p-value EDA'
    # d['CreationDate'] = datetime.today()
    # # d['ModDate'] = datetime.today()
    pp.close()
    print("Function execution completed.")


def test_location(df, categorical, numerical):
    # %matplotlib inline
    plt.rcParams['figure.figsize'] = [10, 10]
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    matplotlib.use('Agg')
    filename = "location.pdf"
    pp = PdfPages(filename)
    # pp.set_font('Arial', 'B', 16)

    print(bold_start, "\nRelationship between Numerical and Location Variables", bold_end)
    categorical_superset = list(filter(None, subsets(categorical)))

    F_values2 = Ftest(df, categorical, numerical)
    dataplot4 = sns.heatmap(F_values2.loc[:, F_values2.columns != 'Categorical_combination'], linecolor='white',
                            linewidths=1, cmap='Reds', vmin=0, vmax=1)
    plt.title('Anova test for combination of Location & numerical 0-Not_Significant 1-Significant',
              fontweight="bold")
    plt.savefig(pp, format='pdf')
    plt.close()

    F_values = F_values2.loc[:, F_values2.columns != 'Categorical_combination']
    


    indices = ind(categorical_superset)
    cat_num_df = pd.DataFrame(F_values, columns=numerical)
    cat_num_df.index = indices
    # zx = 0
    # print(cat_num_df)
    for i in range(0, F_values.shape[0]):
        for j in range(0, F_values.shape[1]):
            if F_values.iloc[i, j] == 1:  # Significant
                l = np.array(indices[i])
                l = np.append(l, numerical[j])
                df2 = df[l]
                df2.boxplot(column=numerical[j], by=indices[i], rot=45, vert=False)
                plt.title(numerical[j] + ' and ' + str(indices[i]), fontweight="bold")

    fig_nums = plt.get_fignums()
    # print(plt.get_fignums())
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')

    # d = pp.infodict()
    # d['Title'] = 'Cornea-EDA'
    # d['Author'] = 'Kaltics'
    # d['Subject'] = 'Multi-dimensional & combinational EDA'
    # d['Keywords'] = 'Statistics hypothesis p-value EDA'
    # d['CreationDate'] = datetime.today()
    # # d['ModDate'] = datetime.today()
    pp.close()


def before_after_summary(unclean_summary, df_summary):
    plt.rcParams['figure.figsize'] = [10, 10]
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    matplotlib.use('Agg')
    filename = "main.pdf"
    main_pdf = PdfPages(filename)

    #Dummy information for the columns
    data = [['Information score', 8.5,"% of data that has value"], ['Data Cleanliness score', 6.5,"% of data that is clean"]]
    dummy_info = pd.DataFrame(data, columns=['Info', 'Score','Additional info'])
    print(dummy_info.head())
    tab = plt.table(cellText=dummy_info.values, colLabels=dummy_info.columns, loc='center',
                    fontsize=3, rowLabels=dummy_info.index)
    tab.auto_set_font_size(False)
    tab.auto_set_column_width(30)
    plt.axis('off')
    plt.title("Summary of your data")
    plt.savefig(main_pdf, orientation='landscape', format='pdf')
    plt.clf()

    #Cleaning summary
    unclean_summary.columns = 'unclean_' + unclean_summary.columns
    #print(unclean_summary.head(2))
    # df_list = pd.dataframe()
    df_list = pd.DataFrame(columns=['clean_data', 'unclean_data'], index=df_summary.index)
    for col_no in range(len(unclean_summary.columns)):
        plt.figure(figsize=(40, 8))
        df_list["clean_data"] = df_summary.iloc[:, [col_no]]
        df_list["unclean_data"] = unclean_summary.iloc[:, [col_no]]
        tab = plt.table(cellText=df_list.values, colLabels=df_list.columns, loc='center',
                        fontsize=2, rowLabels=df_list.index)
        tab.auto_set_font_size(False)
        tab.auto_set_column_width(30)
        plt.axis('off')
        plt.title(df_summary.columns[col_no] + "Transition")
        plt.savefig(main_pdf, orientation='landscape', format='pdf')
        plt.clf()
        df_list[:] = np.nan
    plt.close()


    main_pdf.close()
    return


if __name__ == "__main__":

    start_time = time.time()
    # 1. Set up the PDF doc basics
    # pdf = FPDF()
    # pdf.add_page()
    # pdf.set_font('Arial', 'B', 16)

    # 2. Layout the PDF doc contents
    ## Title
    # pdf.cell(40, 40, 'Daily S&P 500 prices report 1')

    # pdf.ln(20)
    warnings.simplefilter('ignore')
    unclean_summary = pd.read_pickle("./unclean_data_summary.pkl") #./streamlit./
    df_summary = pd.read_pickle("./clean_data_summary.pkl")
    #df_summary.to_csv('file1.csv')
    data = pd.read_pickle("./clean_data.pkl")
    print(df_summary)

    ## Compare clean & unclean data changes


    # getting coulumns in their statistical data type categories
    categorical = []
    for i in range(len(data.columns)):
        if df_summary.loc['Statistical Datatype', data.columns[i]] == 'CATEGORICAL INTEGER' or df_summary.loc[
            'Statistical Datatype', data.columns[i]] == 'CATEGORICAL DECIMAL' or df_summary.loc[
            'Statistical Datatype', data.columns[i]] == 'CATEGORICAL TEXT':
            categorical.append(data.columns[i])

    print(categorical)

    numerical = []
    for i in range(len(data.columns)):
        if df_summary.loc['Statistical Datatype', data.columns[i]] == 'CONTINOUS INTEGER' or \
                df_summary.loc['Statistical Datatype', data.columns[i]] == 'CONTINOUS DECIMAL':
            numerical.append(data.columns[i])
    print(numerical)

    location = []
    for i in range(len(data.columns)):
        if df_summary.loc['Statistical Datatype', data.columns[i]] == 'LOCATION':
            location.append(data.columns[i])
    print(location)

    long_text = []
    for i in range(len(data.columns)):
        if df_summary.loc['Statistical Datatype', data.columns[i]] == "LONG TEXT":
            long_text.append(data.columns[i])
    print(long_text)

    date_time = []
    for i in range(len(data.columns)):
        if df_summary.loc['Statistical Datatype', data.columns[i]] == "DATETIME":
            date_time.append(data.columns[i])
    print(date_time)


    # ['ORDER ID', 'SHIP MODE', 'CUSTOMER ID', 'CUSTOMER NAME', 'SEGMENT', 'PRODUCT ID', 'CATEGORY', 'SUB-CATEGORY', 'QUANTITY', 'DUMMY COLUMN 1']
    # ['ROW ID', 'SALES', 'DISCOUNT', 'PROFIT']
    # ['COUNTRY', 'CITY', 'STATE', 'POSTAL CODE', 'REGION']
    # ['PRODUCT NAME']
    # ['ORDER DATE', 'SHIP DATE']

    # print("Observations of Dickey-fuller test")
    # dftest = adfuller(data['SALES'], autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
    # for key, value in dftest[4].items():
    #     dfoutput['critical value (%s)' % key] = value
    # print(dfoutput)
    #
    test_significance(data.sample(n=100), categorical, numerical)
    test_location(data.sample(n=100), location, numerical)
    before_after_summary(unclean_summary, df_summary)
    # 3. Output the PDF file
    # pdf.ln(20)
    # pdf.output('fpdf_pdf_report.pdf', 'F')

    # 4. Merge all pdf files

    mergedObject = PdfMerger()
    mergedObject.append(PdfReader('main.pdf', 'rb'))
    mergedObject.append(PdfReader('multi.pdf', 'rb'))
    mergedObject.append(PdfReader('location.pdf', 'rb'))

    mergedObject.write("./eda/EDA_report.pdf")
    genai()
    

    print(start_time, time.time())
    print("time elapsed  {:.2f}s".format(time.time() - start_time))
