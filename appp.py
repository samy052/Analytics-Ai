# app.py
import base64
import io
import json
import os
# from parser import ParserError
import re
import sys
import time
import zipfile
import PyPDF2
import pandas as pd
import tempfile
import seaborn as sns
import numpy as np
import streamlit as st
import chardet
import subprocess
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
from reportlab.platypus import Image
from reportlab.lib.units import inch
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import tempfile
import time

# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# def vector_embedding(file_path):
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         documents = []

#         loader = PyPDFLoader(file_path)
#         documents.extend(loader.load())

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         final_documents = text_splitter.split_documents(documents)
#         st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
#         st.session_state.documents = final_documents

# # Specify the file path here

# # specific_file_path ="./EDA_report.pdf"
# specific_file_path = "./us_census./thinkstats2-21-22_compressed.pdf"

# # Call the vector embedding function with the specific file
# vector_embedding(specific_file_path)

# if prompt1 and "vectors" in st.session_state:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': prompt1})
#     st.write(f"Response time: {time.process_time() - start} seconds")
#     st.write(response['answer'])

#     with st.expander("Document Similarity Search"):
#         for doc in response.get("context", []):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
def genai ():
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        st.title("Ask question from GenAi")

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        def vector_embedding(file_path):
            if "vectors" not in st.session_state:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                documents = []

                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(documents)
                st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
                st.session_state.documents = final_documents

        # Specify the file path here

        # specific_file_path ="./EDA_report.pdf"
        specific_file_path = "./us_census/all_of_statistics-wasserman-prepub1_a4-2003.pdf"


        # Call the vector embedding function with the specific file
        vector_embedding(specific_file_path)
        st.write("Vector Store DB Is Ready")

        prompt1 = st.text_input("Enter Your Question From Documents")

        if prompt1 and "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])

            with st.expander("Document Similarity Search"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
                    
def genai2 ():
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        st.title("Ask question from GenAi")

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        def vector_embedding(file_path):
            if "vectors" not in st.session_state:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                documents = []

                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(documents)
                st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
                st.session_state.documents = final_documents

        # Specify the file path here

        # specific_file_path ="./EDA_report.pdf"
        specific_file_path = "./eda/EDA_report.pdf"

        # Call the vector embedding function with the specific file
        vector_embedding(specific_file_path)
        st.write("Vector Store DB Is Ready")

        prompt1 = st.text_input("Enter Your Question For Csv")

        if prompt1 and "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])

            with st.expander("Document Similarity Search"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("--------------------------------")               

def main():
    
    st.set_page_config(page_title="Data Kaltics", page_icon="📊")

    # def local_css(file_name):
    #     with open(file_name) as f:
    #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # local_css("style/style.css")

    st.header("Welcome to Data Kaltics!")

    st.markdown("""
        <div class="header-text animated fadeIn">Upload a CSV file and Ask anything</div>
    """, unsafe_allow_html=True)

    

    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."):
    #             raw_text = get_pdf_text(pdf_docs)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks)
    #             st.success("Done")
    # prompt1 = st.text_input("Enter Your Question From Documents")
   
    
    # with st.sidebar:
        # st.title("Menu:")
    uploaded_file = st.file_uploader("Choose a CSV file (max 10MB)", type=["csv"], accept_multiple_files=False, key="file_uploader")
    df = pd.DataFrame(uploaded_file)
    #filepath2= uploaded_file
    if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            print("path")
            with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            print(path)  
              
    if not uploaded_file:
            # prompt1 = st.text_input("Enter Your Question From Documents")
        
            genai()
            
        
   

    def five_obs(c):
                l = []
                for row in c.head(5):
                    if pd.isna(row):
                        l.append('NULL')
                        print(row)
                    else:
                        l.append(row)
                        print(row)
                return (l)
            
    def float_precision(df):
        df.dropna(inplace=True)
        values = df.copy()
        if values.empty:
            return 'Empty'
        else:
            #         print("here")
            #         print(values.head(2))
            list_values = values.astype(str).str.split(".").str[1:]
            for i in range(0, len(list_values)):
                if list_values[i] == ['0']:
                    list_values[i] = []
                    list_values[i] = ['i']

            valuesb: pd.DataFrame = pd.DataFrame({'a': list_values})

            # print(valuesb['a'][1])
            valuesb = pd.DataFrame({'a': valuesb.a.astype(str).str.split("")})  # .a.astype(str).str.split("")})
            valuesb['a'] = valuesb['a'].apply(lambda row: [val for val in row if val != '['])
            valuesb['a'] = valuesb['a'].apply(lambda row: [val for val in row if val != ']'])
            valuesb['a'] = valuesb['a'].apply(lambda row: [val for val in row if val != "'"])
            valuesb['a'] = valuesb['a'].apply(lambda row: [val for val in row if val != ' '])
            valuesb['a'] = valuesb['a'].apply(lambda row: [val for val in row if val != ''])
            pre_dis = pd.DataFrame(
                {'a': valuesb['a'].value_counts().index.tolist(), 'b': valuesb['a'].value_counts(), 'c': 0})
            pre_dis['c'] = pre_dis['a'].str.len()
            pre_dis.reset_index(drop=True, inplace=True)
            pre_dis.sort_values(['b', 'c'], ascending=False, inplace=True)
            # print(pre_dis.head(2))

        # min format
        if pre_dis.loc[pre_dis['c'].idxmin()]['a'] == ['i']:
            min_format_needed = 0
        else:
            min_format_needed = pre_dis.loc[pre_dis['c'].idxmin()]['c']
        # auto format
        if pre_dis.loc[0]['a'] == ['i']:
            auto_format_needed = 0
        else:
            auto_format_needed = pre_dis.loc[0]['c']

        format_needed = pd.DataFrame({'auto': auto_format_needed, 'max': pre_dis.loc[pre_dis['c'].idxmax()]['c'],
                                    'min': min_format_needed}, index=[0])
        # print(format_needed)
        #     format_needed = ''
        #     if precision == 'uto':
        #         format_needed = pre_dis.loc[0]['c']
        #     if precision == 'Max':
        #         format_needed = pre_dis.loc[pre_dis['c'].idxmax()]['c']
        #     if precision == 'Min':
        #         format_needed = pre_dis.loc[pre_dis['c'].idxmin()]['c']
        #     print(str(format_needed) + ' digits')
        return (format_needed)
    
    def date_precision(df_acf):
        df_acf.dropna(inplace=True)
        values = pd.DataFrame({'a': df_acf, 'b': '', 'c': 0})
        #     print("something to be done here")
        #     df = df.astype(str)
        #     df = df.str.split(r'\D')
        #     values['c'] = values['c'].astype(int)
        for i in range(len(values['a'])):
            # print(i, values['a'][i])
            out1 = []
            if values['a'][i].year:
                out1.append("Year")
            if values['a'][i].month:
                out1.append("Month")
            if values['a'][i].day:
                out1.append("Day")
            if values['a'][i].hour or (
                    values['a'][i].minute > 0 or values['a'][i].second > 0 or values['a'][i].microsecond > 0):
                out1.append("Hour")
            if values['a'][i].minute or (values['a'][i].second > 0 or values['a'][i].microsecond > 0):
                out1.append("Minute")
            if values['a'][i].second or values['a'][i].microsecond > 0:
                out1.append("Second")
            if values['a'][i].microsecond:
                out1.append("Microsecond")
            values.at[i, 'b'] = out1
        #         count1 = 0
        #         for j in out1:
        #             count1 += 1
        # #         print(count1)
        #          values.iat[i,2] = count1

        #     print(values.head())
        date_pr = pd.DataFrame({'precision': values['b'].value_counts().index.tolist(), 'count': values['b'].value_counts(), 'length': 0})
        #     print(date_pr.shape)
        date_pr['length'] = date_pr['precision'].str.len()
        date_pr.sort_values(['count', 'length'], ascending=False, inplace=True)
        date_pr.reset_index(drop=True, inplace=True)
        print(date_pr.head(2))
        format_needed = pd.DataFrame({'auto': [date_pr.loc[0]['precision']], 'max': [date_pr.loc[date_pr['length'].idxmax()]['precision']],
                                    'min': [date_pr.loc[date_pr['length'].idxmin()]['precision']]}, index=[0])
        #     print(format_needed)
        #     if precision == 'Auto':
        #         format_needed = date_pr.loc[0]['a']
        #     if precision == 'Max':
        #         format_needed = date_pr.loc[date_pr['c'].idxmax()]['a']
        #     if precision == 'Min':
        #         format_needed = date_pr.loc[date_pr['c'].idxmin()]['a']

        return format_needed


    def text_precision(values):
        #     values = df.copy()
        values.dropna(inplace=True)
        valuesb = values.str.split("\\s+")
        format_needed = pd.DataFrame({'auto': 'N/A', 'max': str(
            str(valuesb.str.len().min()) + ' words or ' + str(values.astype(str).str.len().min()) + ' characters '),
                                    'min': str(str(valuesb.str.len().min()) + ' words or ' + str(
                                        values.astype(str).str.len().min()) + ' characters ')}, index=[0])
        return format_needed


    def data_summary(data2):
        # initial data processing and summary step 2
        data_type = data2.dtypes
        missing_value = data2.isnull().sum()
        count = data2.count()
        index_1 = []
        for col in data2.columns:
            index_1.append(col)
        index_1 = pd.Series(index_1, index=None)
        max_frequency_observation = []
        min_frequency_observation = []

        for col in data2.columns:
            m = data2.loc[:, col].value_counts().idxmax()
            max_frequency_observation.append(m)
        max_frequency_observation = max_frequency_observation[0:len(max_frequency_observation)]
        max_frequency_observation = pd.Series(max_frequency_observation)
        df_temp_max = pd.concat([index_1, max_frequency_observation], axis=1)

        for col in data2.columns:
            mm = data2.loc[:, col].value_counts().idxmin()
            min_frequency_observation.append(mm)
        min_frequency_observation = min_frequency_observation[0:len(min_frequency_observation)]
        min_frequency_observation = pd.Series(min_frequency_observation)
        df_temp_min = pd.concat([index_1, min_frequency_observation], axis=1)

        col_location = []
        for col in data2.columns:
            m = data2.columns.get_loc(col)
            # m=data2.loc[:,col].head(5)
            col_location.append(m)
        col_location = col_location[0:len(col_location)]
        col_location = pd.Series(col_location)
        df_temp_col_location = pd.concat([index_1, col_location], axis=1)

        df_temp_max = df_temp_max.set_index(0)
        df_temp_max.index.name = None

        df_temp_min = df_temp_min.set_index(0)
        df_temp_min.index.name = None

        df_temp_col_location = df_temp_col_location.set_index(0)
        df_temp_col_location.index.name = None

        df = pd.concat([data_type, missing_value], axis=1)
        df_11 = pd.concat([df_temp_min, df_temp_max, df], axis=1)
        df_11 = pd.concat([df_11, count], axis=1)
        df_11.T
        category = data2.nunique()
        column = ['Min_frequency_observation', 'Max_frequency_observation', 'Data_type', 'missing',
                'Unique Category', 'count']

        df_11 = pd.concat([df_11, category], axis=1)
        df_11 = pd.concat([df_11, df_temp_col_location], axis=1)

        stat_type = []

        location = ['POSTAL', 'CODE', 'ZIP', 'POSTAL CODE', 'ZIPCODE', 'POSTALCODE',
                    'POSTCODE', 'ADD', 'ADDRESS', 'STATE', 'CITY', 'COUNTRY', 'CONTINENT', 'AREA',
                    'TALUKA', 'LONGITUDE', 'LATITUDE', 'LOCATION', 'GEOGRAPHIC', 'REGION', 'LOCALITY', 'LOCALE',
                    'SITE', 'VENUE', 'PRECINCT', 'SECTOR', 'DISTRICT']
        location = '|'.join(location)
        colNames = data2.columns[data2.columns.str.contains(location)]
        # print(colNames)

        for i in range(len(data2.columns)):
            #     md = data2.columns[i]
            #     print(md, md.str.('Postal code'))
            #     if data_type.iloc[i] == 'object': print(data2.iloc[:,i].str.split().str.len().max())
            if data_type.iloc[i] == 'int64' and np.any(data2.columns[i] == colNames):
                m = "LOCATION"
            elif data_type.iloc[i] == 'int64' and not np.any(data2.columns[i] == colNames):
                if data2.iloc[:, i].is_unique:
                    if len(data2.iloc[:, i].unique()) == len(data2.iloc[:, i]):
                        m = "CONTINOUS INTEGER"
                        if data2.iloc[:, i].max() == len(data2.iloc[:, i]):
                            m = 'UNIQUE ROW ID'
                    else:
                        m = 'CATEGORICAL INTEGER'
                else:
                    m = "CONTINOUS INTEGER"
            elif data_type.iloc[i] == 'float64' and not np.any(data2.columns[i] == colNames):
                # if not data2.iloc[:, i].is_unique:
                #     m = 'categorical decimal'
                # else:
                m = "CONTINOUS INTEGER"
            elif data_type.iloc[i] == 'float64' and np.any(data2.columns[i] == colNames):
                m = "LOCATION"
            elif (data_type.iloc[i] == 'datetime64[ns]'):
                m = "DATETIME"
            elif (data_type.iloc[i] == 'object' and np.any(data2.columns[i] == colNames)):
                m = "LOCATION"
            elif (data_type.iloc[i] == 'object' and data2.iloc[:, i].str.split().str.len().max() < 10
                and not np.any(data2.columns[i] == colNames)):
                # Check if email
                regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                print(data2[data2.columns[i]][1])
                if (re.fullmatch(regex, data2[data2.columns[i]][1])):
                    m = 'EMAIL'
                else:
                    if not data2.iloc[:, i].is_unique:
                        m = 'CATEGORICAL INTEGER'
                    else:
                        m = "SHORT TEXT"
            elif (data_type.iloc[i] == 'object' and data2.iloc[:, i].str.split().str.len().max() >= 10):
                regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                if (re.fullmatch(regex, data2[data2.columns[i]][1])):
                    m = 'EMAIL'
                else:
                    m = "LONG TEXT"
            else:
                m = "OTHERS"
            stat_type.append(m)
        df_11['Statistical_Datatype'] = stat_type
        # print(stat_type)
        df_11.columns = ['Min frequency observation', 'Max frequency observation', 'Data type', 'Missing', 'Count',
                        'Unique Category', 'Column location from left', 'Statistical Datatype']
        return (df_11.T)


    # this definition gives added summary of all the columns and recommendations
    def data_summary2(data2, df_summary):
        #     df_summary.T
        l1 = []
        for i in range(len(data2.columns)):
            l1.append(five_obs(data2.iloc[:, i]))
        df_summary['First five observation'] = l1

        l2 = []
        for i in range(len(data2.columns)):
            # print(data2.iloc[:, i].is_unique)
            l2.append(data2.iloc[:, i].nunique())
        df_summary['Has Unique Observatons'] = l2

        l2 = []
        for i in range(len(data2.columns)):
            l2.append(data2.iloc[:, i].nunique())
        df_summary['Unique observation'] = l2

        l3 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Statistical Datatype'][i] != 'location':
                l3.append(data2.iloc[:, i].min())
            else:
                l3.append('NULL')
        df_summary['Min observation'] = l3

        l4 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Statistical Datatype'][i] != 'location':
                l4.append(data2.iloc[:, i].max())
            else:
                l4.append('NULL')
        df_summary['Max observation'] = l4

        l5 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Data type'][i] != 'datetime64[ns]' and \
                    df_summary['Statistical Datatype'][i] != 'location':
                l5.append(data2.iloc[:, i].mean())
            else:
                l5.append('NULL')
        df_summary['Mean observation'] = l5

        l6 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Data type'][i] != 'datetime64[ns]' and \
                    df_summary['Statistical Datatype'][i] != 'location':
                l6.append(data2.iloc[:, i].median())
            else:
                l6.append('NULL')
        df_summary['Median observation'] = l6

        l6 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Data type'][i] != 'datetime64[ns]' and \
                    df_summary['Statistical Datatype'][i] != 'location':
                l6.append(np.std(data2.iloc[:, i], axis=0))
            else:
                l6.append('NULL')
        df_summary['Standard Deviation'] = l6

        l7 = []
        l8 = []
        l9 = []
        for i in range(len(data2.columns)):
            if df_summary['Statistical Datatype'][i] == 'continous decimal':
                format_needed = float_precision(data2.iloc[:, i].copy())
                #         print(format_needed.loc[0]['min'])
                l7.append(format_needed.loc[0]['min'])
                l8.append(format_needed.loc[0]['max'])
                l9.append(format_needed.loc[0]['auto'])
            elif df_summary['Statistical Datatype'][i] == 'datetime':
                # print(df_summary['Column location from left'][i])
                format_needed = date_precision(data2.iloc[:, i].copy())
                #         print(format_needed.loc[0]['min'])
                l7.append(format_needed.loc[0]['min'])
                l8.append(format_needed.loc[0]['max'])
                l9.append(format_needed.loc[0]['auto'])
            elif df_summary['Statistical Datatype'][i] == 'location':
                l7.append('NULL')
                l8.append('NULL')
                l9.append('NULL')
            elif df_summary['Statistical Datatype'][i] == 'short text' or df_summary['Statistical Datatype'][
                i] == 'short text':
                format_needed = text_precision(data2.iloc[:, i].copy())
                l7.append(format_needed.loc[0]['min'])
                l8.append(format_needed.loc[0]['max'])
                l9.append(format_needed.loc[0]['auto'])
            else:
                l7.append('NULL')
                l8.append('NULL')
                l9.append('NULL')

        df_summary['Min Precision'] = l7
        df_summary['Max Precision'] = l8
        df_summary['Recommended Precision'] = l9

        l7 = []
        for i in range(len(data2.columns)):
            if df_summary['Data type'][i] != 'object' and df_summary['Data type'][i] != 'datetime64[ns]' and \
                    df_summary['Statistical Datatype'][i] != 'location':
                stdc = np.std(data2.iloc[:, i])
                threshold = 3
                meanc = np.mean(data2.iloc[:, i])
                outliers = []
                for r in data2.index:
                    y = data2.iloc[r, i]
                    if stdc != 0:
                        z_score = (y - meanc) / stdc
                        if np.abs(z_score) > threshold:
                            outliers.append(y)
                l7.append(len(outliers))

            else:
                l7.append('NULL')

        df_summary['No of Outliers'] = l7

        l7 = []
        for i in range(len(data2.columns)):
            l6 = []
            ## I will keep updating this section
            if df_summary['Data type'][i] != 'object' and df_summary['Data type'][i] != 'datetime64[ns]' and \
                    df_summary['Statistical Datatype'][i] != 'location':
                if df_summary['Count'][i] == df_summary['Unique Category'][i] and df_summary['No of Outliers'][i] == 0:
                    l6.append("This is possibly an unique identifier column")
                if df_summary['No of Outliers'][i] > 0 and df_summary['Missing'][i] > 0:
                    l6.append("Use the auto corrector algorithm. Replace Outliers and Missing with average.")
                if df_summary['Max Precision'][i] != df_summary['Min Precision'][i]:
                    l6.append("We can change the precision in this column")
                if df_summary['Unique Category'][i] == 1:
                    l6.append(
                        "All the values in the column are same. Removing this column is recommended as it wont add any value to the analysis. Else add more data.")
            if df_summary['Data type'][i] == 'datetime64[ns]':
                if df_summary['Max Precision'][i] != df_summary['Min Precision'][i]:
                    l6.append("Date format inconsistencies exist.")
            #         l6.append(df_summary['Data type'][i])
            if df_summary['Data type'][i] == 'object' and df_summary['Statistical Datatype'][i] == 'Email':
                l6.append("All Valid Email")
                regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                for key, value in data2[data2.columns[i]].iteritems():
                    if (re.fullmatch(regex, value)):
                        pass
                    else:
                        l6[0] = "Contains invalid email, manual cleaning needed"
                        break
            l7.append(l6)
            ## I will keep updating this section

        # print(l7)

        df_summary['Key Recommendations'] = l7
        print(df_summary.T)
        return (df_summary.T)
    

    def format_str(format_needed):
        format_string = ''
        if 'Year' in format_needed:
            format_string += '%y-'
        if 'Month' in format_needed:
            format_string += '%m-'
        if 'Day' in format_needed:
            format_string += '%d '
        if 'Hour' in format_needed:
            format_string += '%H'
            if 'Minute' not in format_needed:
                format_string += ':%M'
        if 'Minute' in format_needed:
            format_string += ':%M'
        if 'Second' in format_needed:
            format_string += ':%S'
        if 'Microsecond' in format_needed:
            format_string += '.%f'
        return format_string


    def auto_clean(df_ac, df_sum, precision):
        for i in range(len(df_ac.columns)):
            #         print(df_ac.iloc[:,i].head(1), df_sum['Statistical Datatype'][i])
            if df_sum.loc['Max Precision'][i] != df_sum.loc['Min Precision'][i]:
                if df_sum.loc['Statistical Datatype'][i] == 'datetime':
                    #                     df_ac.iloc[:,i] = pd.to_datetime(df_ac.iloc[:,i])
                    if precision == 'auto':
                        format_need = df_sum.loc['Recommended Precision'][i]
                    elif precision == 'max':
                        format_need = df_sum.loc['Max Precision'][i]
                    elif precision == 'min':
                        format_need = df_sum.loc['Min Precision'][i]
                    format_string = format_str(format_need)
                    #                     print(format_string)
                    #                     print(df_ac.loc[0][i].year)
                    df_ac.iloc[:, i] = df_ac.iloc[:, i].dt.strftime(date_format=format_string)
                    if format_string == '%y-%m-%d ' or format_string == '%y-%m-%d':
                        df_ac.iloc[:, i] = pd.to_datetime(df_ac.iloc[:, i]).dt.date
                    else:
                        df_ac.iloc[:, i] = pd.to_datetime(df_ac.iloc[:, i], format=format_string)

                #                 print(df_ac.iloc[:,i].head(2))
                #                 print(type(df_ac.iloc[:,i]))
                #                 print(df_ac.loc[0][i].year)
                elif df_sum.loc['Statistical Datatype'][i] == 'continous decimal':
                    if precision == 'auto':
                        format_need = df_sum.loc['Recommended Precision'][i]
                    elif precision == 'max':
                        format_need = df_sum.loc['Max Precision'][i]
                    elif precision == 'min':
                        format_need = df_sum.loc['Min Precision'][i]
                    #                     format_string = str("'{:,."+str(format_need)+"f}'")
                    #                     print(format_string)
                    if format_need == 0:
                        df_ac.iloc[:, i] = df_ac.iloc[:, i].astype('int')
                    else:
                        df_ac.iloc[:, i] = df_ac.iloc[:, i].map(str("{:." + str(format_need) + "f}").format)
                        df_ac.iloc[:, i] = df_ac.iloc[:, i].astype('float64')

            else:
                pass
        #     print(df_ac.dtypes)
        return df_ac


    # __Main__ code execution starts from here

    def classify(filepath, create_program):
        file = read_doc(filepath)
        create_program.append('##')
        domain = Predict_Domain(file)
        print('Domain : ', domain)
        sub_domain = Predict_SubDomain(file)
        print('SubDomain : ', sub_domain)
        tags = PredictTags(file)
        print('Tags : ', tags)
        create_program.append('##')
        create_program.append('#' + '# Domain : ' + ' '.join(domain))
        create_program.append('#' + '# SubDomain : ' + ' '.join(sub_domain))
        create_program.append('#' + '# Tags : ' + ' '.join(tags))
        create_program.append('##')
        return create_program


    def step1A(file_name_path):
        create_program = ['#Code generated by Cornea. Please review the code to generate some changes if needed',
                        '#Import Modules', 'import pandas as pd', '', '',
                        '#Importing the File as csv']

        file_path =  file_name_path
        print(os.getcwd())
        print(file_path)
        pd.set_option('display.max_columns', 50)

        file_name, file_extension = os.path.splitext(file_path)
        with open(file_path, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(10000))
            file_encoding = result['encoding']
        # check what the character encoding might be
        # print(result)
        #         data = pd.read_csv(file_path, encoding = file_encoding, float_precision='high' )#,
        create_program.append('##' + file_name)
        if file_extension.lower() == '.csv':
            #     print("Reading CSV file format")
            try:
                data = pd.read_csv(file_path, encoding=file_encoding, float_precision='high')  # , delim_whitespace=True)
                # print('data = pd.read_csv(r"' + file_path + '", encoding = "' + file_encoding + '" , float_precision= "high")')
                create_program.append(
                    'data = pd.read_csv(r"' + file_path + '", encoding = "' + file_encoding + '" , float_precision= "high")')
            except ParserError:
                data = pd.read_csv(file_path, encoding=file_encoding, float_precision='high', delim_whitespace=True)
                # print('data = pd.read_csv(r"' + file_path + '", encoding = "' + file_encoding + '" , float_precision= "high"), delim_whitespace=True')
                create_program.append(
                    'data = pd.read_csv(r"' + file_path + '", encoding = "' + file_encoding + '" , float_precision= "high") , delim_whitespace=True')

        create_program.append('')
        create_program.append('')
        create_program.append('unclean_data_summary = pd.read_pickle("./unclean_data_summary.pkl")')
        create_program.append('clean_data_summary = pd.read_pickle("./clean_data_summary.pkl")')

        data.columns = data.columns.str.upper()
        data.columns = data.columns.str.strip()
        print(data.dtypes)

        for col_name in data.columns:
            # remove white spaces before and after
            if data[col_name].dtype == 'object':
                try:
                    data[col_name] = pd.to_datetime(data[col_name])
                except ValueError:
                    pass
        #     if data[col].dtype == 'float64':
        #         data[col].fillna(0, inplace= True)

        create_program.append('# Clean data with spaces before and after')
        create_program.append('# Classify columns as datetime columns if applicable else pass')
        create_program.extend(('for col in data.columns:',
                            '\t # remove white spaces before and after \n\tdata.columns = data.columns.str.strip()',
                            '\tif data[col].dtype == "object":',
                            '\t\t try: \n\t\t\t data[col] = pd.to_datetime(data[col]) \n'
                            '\t\t except ValueError: \n\t\t\t pass',
                            ''))

        # data.head(1)
        create_program.append('#We have identified the data and classified as below')
        create_program.append('')

        # create_program = classify('./superstore_data.csv',create_program)  # File path needed
        # create_program.append('##' + file_name)

        df_summary = data_summary(data)

        return df_summary, create_program, data


    ########################################################################
    def step1B(df_summary, create_program, data):
        data2 = data.copy()
        df_summary2 = df_summary.copy()
        df_summary = data_summary2(data2, df_summary.T)  # Show this summary of unclean data

        # output in json part one: unclean data summary
        sys.setrecursionlimit(1000000000)
        df_summary_json = df_summary.to_json(orient='index', default_handler=str)
        data2 = data.copy()
        # pass the precision level to get appropriate level of cleaning, min, max or auto

        # output in json part two: clean data for step 2
        data2 = auto_clean(data2, df_summary, 'auto')
        print(type(data2))
        data2_json = data2.to_json(orient='index', default_handler=str)
        create_program.append('##2')
        df_summary2 = data_summary2(data2, df_summary2.T)  # Summary of clean data
        # print(data2_json)

        return df_summary, df_summary2, data, data2


    def step1():
        file_name_path =  path # File path needed
        print(file_name_path)
        print(os.getcwd())
        df_summary, create_program, data = step1A(file_name_path)
        print("done 1")
        # st.header("Summary of DataFrame")
        # st.dataframe(df_summary)
        df_summary_json = df_summary.to_json(orient='index', default_handler=str)
        with open('df_summary.json', 'w') as f:
            json.dump(df_summary_json, f)
        f.close()

        f = open('df_summary.json')
        data_json = json.load(f)
        df_summary = pd.read_json(data_json, orient='index')
        # print(df_summaryj.head(1))
        # df_summaryj.to_csv("dfsummary.csv", sep= '\t')
        df_summary, df_summary2, data, data2 = step1B(df_summary, create_program, data)
        print("done 2")

         # for i in
        for i in range(len(data.columns)):
            print(df_summary.loc['Statistical Datatype', data.columns[i]], data.columns[i], i)

        print(*create_program, sep='\n')
        sourceFile = open('auto_clean_program.py', 'w')
        print(*create_program, sep='\n', file=sourceFile)
        df_summary.to_pickle('./unclean_data_summary.pkl')
        df_summary2.to_pickle('./clean_data_summary.pkl')
        data.to_pickle('./unclean_data.pkl')
        data2.to_pickle('./clean_data.pkl')


        z = zipfile.ZipFile("test2.zip", "w")
        z.write("auto_clean_program.py")
        z.write("unclean_data_summary.pkl")
        z.write("clean_data_summary.pkl")
        z.write("unclean_data.pkl")
        z.write("clean_data.pkl")
        sourceFile.close()

    def extract_first_n_pages(input_pdf, output_pdf, num_pages):
                try:
                    with open(input_pdf, 'rb') as infile:
                        reader = PyPDF2.PdfReader(infile)
                        writer = PyPDF2.PdfWriter()

                        for i in range(min(num_pages, len(reader.pages))):
                            writer.add_page(reader.pages[i])

                        with open(output_pdf, 'wb') as outfile:
                            writer.write(outfile)
                except Exception as e:
                    st.error(f"Error extracting pages: {e}")


    


###########################streamlitttttt########################################################################
    
    if uploaded_file is not None:
        max_size_mb = 10
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  
        if file_size > max_size_mb:
            st.error(f"File size exceeds the maximum limit of {max_size_mb} MB.")
        else:
            st.success("File uploaded successfully!")
            st.info("Now you can review the data and ask anything below that.")
            
            # df = pd.read_csv(uploaded_file)

            # Call step1A to get df_summary
            result_tuple = step1A(path)

            # Extract df_summary from the tuple
            df_summary = result_tuple[0]
            creg_prog = result_tuple[1]
            data = result_tuple[2]

            # Convert df_summary to a DataFrame
            df_summary_dataframe = pd.DataFrame(df_summary)
            


            # Display df_summary in Streamlit as an editable DataFrame
            st.header("Summary of DataFrame")
            #edited_df_summary = st.data_editor(df_summary_dataframe)
            st.dataframe(df_summary_dataframe)


            if st.button("Want to Edit Row?"):
            
                # Add a selectbox to choose the column
                selected_column = st.selectbox("Select a column:", df_summary_dataframe.columns)

                # Display the actual value of the selected column
                default_value = df_summary_dataframe[selected_column].iloc[-1]
                st.write(f"Default value of {selected_column}: {default_value}")

                # Dropdown options including the default value, 'option1', 'option2', and 'option3'
                options = [default_value, 'option1', 'option2', 'option3']

                # Add a dropdown to choose the new value for the selected column
                new_value = st.selectbox(f"Select a new value for {selected_column}:", options, key=f"{selected_column}_selectbox")

                # Add a button to edit another column
                if st.button("Edit Another Column +"):
                    # Filter DataFrame columns to exclude the previously selected column
                    available_columns = [col for col in df_summary_dataframe.columns if col != selected_column]
                    
                    # Display another selectbox to choose another column
                    selected_column_another = st.selectbox("Select another column:", available_columns, key="selected_column_another")
                    
                    # Display the actual value of the selected column
                    default_value_another = df_summary_dataframe[selected_column_another].iloc[-1]
                    st.write(f"Default value of {selected_column_another}: {default_value_another}")
                    
                    # Dropdown options including the default value, 'option1', 'option2', and 'option3'
                    options_another = [default_value_another, 'option1', 'option2', 'option3']
                    
                    # Add a dropdown to choose the new value for the selected column
                    new_value_another = st.selectbox(f"Select a new value for {selected_column_another}:", options_another, key=f"{selected_column_another}_selectbox")

                # Add a button to update the last row of the selected column
                if st.button("Update"):
                    # Update the last row of the selected column with the new value
                    df_summary_dataframe[selected_column].iloc[-1] = new_value
                    
                    # Update the last row of the another selected column with the new value if selected
                    if 'selected_column_another' in locals():
                        df_summary_dataframe[selected_column_another].iloc[-1] = new_value_another
                    
                    st.write("Updated DataFrame:")
                    st.dataframe(df_summary_dataframe)

            result_tuple2 = step1B(df_summary_dataframe, creg_prog, data)

            st.dataframe(result_tuple2[0])

            # if st.button("No Thanks, Proceed"):
            data2 = result_tuple2[3]
            st.dataframe(data2)
            st.write("Summary of Cleaned data")
            df_summary2 = result_tuple2[1]
            st.dataframe(df_summary2)
            st.write("Generating EDA report...")

            step1()

           

            print("Starting step3")
            subprocess.run(["python", "step3_mod.py"])
            print("Ending step3") 
            st.write("EDA report generated!")
            
            # with open("EDA_report.pdf", "rb") as f:
            #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # # Embedding PDF in HTML
            # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

            # # Displaying File
            # st.markdown(pdf_display, unsafe_allow_html=True)  
            # Extract the first 30 pages
            input_pdf = "./eda/EDA_report.pdf"
            output_pdf = "EDA_report_first_30_pages.pdf"
            extract_first_n_pages(input_pdf, output_pdf, 30)

            # Read and encode the new PDF
            try:
                with open(output_pdf, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                # Embedding PDF in HTML
                pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

                # Displaying the PDF
                st.markdown(pdf_display, unsafe_allow_html=True)  
            except FileNotFoundError:
                st.error(f"{output_pdf} not found. Please make sure the file exists and try again.")
            genai2()
                    
            # df_summary = step1A(path)
            # st.header("Summary of DataFrame")
            # st.dataframe(df_summary)



            # Display the df_summary as a table
            # st.write("Summary DataFrame:")
            # st.table(df_summary)

            #step1()
            
            # st.subheader("DataFrame - Head")
            # st.dataframe(df.head())
            # st.subheader("DataFrame - Describe")
            # st.dataframe(df.describe())
        
            
            # numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

            # st.subheader("Select PDF Content:")
            # include_head = st.checkbox("Include Head of DataFrame", value=False)
            # include_describe = st.checkbox("Include Summary Statistics (Describe)", value=False)
            # include_histogram = st.checkbox("Include Histogram", value=False)
            # include_line_plot = st.checkbox("Include Line Plot", value=False)
            # include_scatter_plot = st.checkbox("Include Scatter Plot", value=False)

            # selected_column = st.selectbox("Select a numeric column for plotting", numeric_columns)

            # pdf_buffer = io.BytesIO()
            # doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            # styles = getSampleStyleSheet()
            # elements = []

            # if include_head:
            #     head_text = f"Head of DataFrame - {len(df)} rows"
            #     elements.append(Paragraph(head_text, styles['Title']))
            #     head_table = Table([df.columns] + df.head().values.tolist(), hAlign='LEFT')
            #     head_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            #                                     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            #                                     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            #                                     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            #                                     ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            #                                     ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            #                                     ('GRID', (0, 0), (-1, -1), 1, colors.black)])
            #                         )
            #     elements.append(head_table)
            #     elements.append(Spacer(1, 6))

            # if include_describe:
            #     describe_header = "Summary Statistics (Describe)"
            #     elements.append(Paragraph(describe_header, styles['Heading2']))
            #     describe_transposed = df.describe().T

            #     describe_table_data = describe_transposed.reset_index().values.tolist()
            #     headers = describe_transposed.columns.tolist()
            #     headers.insert(0, "Statistic")
            #     describe_table_data.insert(0, headers)

            #     describe_table = Table(describe_table_data, hAlign='LEFT')

            #     describe_table.setStyle(TableStyle([
            #         ('FONTSIZE', (0, 0), (-1, -1), 8),
            #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            #         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            #         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            #         ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            #         ('GRID', (0, 0), (-1, -1), 1, colors.black)
            #     ]))

            #     describe_table._argW[0] = 0.8 * inch
            #     describe_table._argW[1] = 0.6 * inch

            #     elements.append(describe_table)
            #     elements.append(Spacer(1, 6))

            # if include_histogram:
            #     fig, ax = plt.subplots(figsize=(8, 4))
            #     sns.histplot(df[selected_column], kde=True, ax=ax)
            #     ax.set_xlabel(selected_column)
            #     ax.set_ylabel('Frequency')
            #     plt.title(f'Histogram of {selected_column}')
            #     plt.savefig("histogram.png", format='png', bbox_inches='tight')
            #     plt.close(fig)
            #     elements.append(Paragraph(f'Histogram of {selected_column}', styles['Title']))
            #     elements.append(Image("histogram.png", width=6*inch, height=3*inch))
            #     st.subheader(f'Histogram of {selected_column}')
            #     st.pyplot(fig)

            # if include_line_plot:
            #     fig, ax = plt.subplots(figsize=(8, 4))
            #     sns.lineplot(x=df.index, y=df[selected_column], ax=ax)
            #     ax.set_xlabel('Index')
            #     ax.set_ylabel(selected_column)
            #     plt.title(f'Line Plot of {selected_column}')
            #     plt.savefig("line_plot.png", format='png', bbox_inches='tight')
            #     plt.close(fig)
            #     elements.append(Paragraph(f'Line Plot of {selected_column}', styles['Title']))
            #     elements.append(Image("line_plot.png", width=6*inch, height=3*inch))
            #     st.subheader(f'Line Plot of {selected_column}')
            #     st.pyplot(fig)

            # if include_scatter_plot:
            #     second_column = st.selectbox("Select another numeric column", numeric_columns)
            #     if second_column:
            #         fig, ax = plt.subplots(figsize=(8, 4))
            #         sns.scatterplot(x=df[selected_column], y=df[second_column], ax=ax)
            #         ax.set_xlabel(selected_column)
            #         ax.set_ylabel(second_column)
            #         plt.title(f'Scatter Plot of {selected_column} with {second_column}')
            #         plt.savefig("scatter_plot.png", format='png', bbox_inches='tight')
            #         plt.close(fig)
            #         elements.append(Paragraph(f'Scatter Plot of {selected_column} with {second_column}', styles['Title']))
            #         elements.append(Image("scatter_plot.png", width=6*inch, height=3*inch))
            #         st.subheader(f'Scatter Plot of {selected_column} with {second_column}')
            #         st.pyplot(fig)

            # doc.build(elements)

            # st.subheader("Download PDF Report")
            # st.download_button(
            #     label="Download",
            #     data=pdf_buffer.getvalue(),
            #     file_name="data_analysis_report.pdf",
            #     mime="application/pdf"
            # )

if __name__ == "__main__":
    main()


