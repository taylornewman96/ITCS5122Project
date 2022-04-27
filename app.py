from tkinter import font
from tokenize import group
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from vega_datasets import data

matplotlib.use("Qt4Agg")


def load_clf_model():
    filename = "finalized_model_clf.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def load_label_encoder_location():
    filename = "finalized_label_encoder_location.sav"
    return pickle.load(open(filename, 'rb'))

def load_label_encoder_sector():
    filename = "finalized_label_encoder_sector.sav"
    return pickle.load(open(filename, 'rb'))

def load_label_encoder_title():
    filename = "finalized_label_encoder_title.sav"
    return pickle.load(open(filename, 'rb'))

def load_chart_data():
    df = pd.read_csv('data_cleaned_2021.csv')
    return df

enc_sector, enc_location, enc_title = load_label_encoder_sector(), load_label_encoder_location(), load_label_encoder_title()
clf = load_clf_model()

st.set_page_config(layout="wide")

st.write("""
## Insights for Data Science roles in US
""")

st.markdown("""
Welcome! Here we provide some valuable insights into salaries for Data Science roles nationwide. Our visualizations will help you compare salaries across different industries, locations, and companies while our predictive salary tool will help you determine what you can expect to earn based on your skills!
* **Python libraries:** streamlit, pandas, sklearn
* **Data source:** [data-scientist-salary](https://www.kaggle.com/datasets/nikhilbhathi/data-scientist-salary-us-glassdoor)
""")

st.sidebar.write("""
### User Input Features
""")    
params = {
    "sector": ['Accounting & Legal',
                'Aerospace & Defense',
                'Agriculture & Forestry',
                'Arts, Entertainment & Recreation',
                'Biotech & Pharmaceuticals',
                'Business Services',
                'Construction, Repair & Maintenance',
                'Consumer Services',
                'Education',
                'Finance',
                'Government',
                'Health Care',
                'Information Technology',
                'Insurance',
                'Manufacturing',
                'Media',
                'Mining & Metals',
                'Non-Profit',
                'Oil, Gas, Energy & Utilities',
                'Real Estate',
                'Retail',
                'Telecommunications',
                'Transportation & Logistics',
                'Travel & Tourism'],
"job_location" : ['AL', 'AZ', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'MD', 'MA', 'MI', 'MN', 'MO', 'NE', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OR', 'PA', 'RI', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI'],
"skills": ['Python',
            'spark',
            'aws',
            'excel',
            'sql',
            'sas',
            'keras',
            'pytorch',
            'scikit',
            'tensor',
            'hadoop',
            'tableau',
            'bi',
            'flink',
            'mongo',
            'google_an'],
    "job_title_sim":['Data scientist project manager',
        'analyst',
        'data analitics',
        'data engineer',
        'data modeler',
        'data scientist',
        'director',
        'machine learning engineer',
        'other scientist']
}   

name = st.sidebar.text_input('Enter Name:')
sector = st.sidebar.selectbox('Choose Sector:', params['sector'])
job_location = st.sidebar.selectbox('Choose Job Location:', params["job_location"])
skill_set = st.sidebar.multiselect('Skills', params['skills'])
job_title_sim = st.sidebar.selectbox('Choose Job Title:', params['job_title_sim'])
predict_btn = st.sidebar.button('Predict')


test_data = pd.DataFrame({
    'Sector' : list(enc_sector.transform([sector])),
    'Job Location' : list(enc_location.transform([job_location])),
    'Python' : [1 if 'Python' in skill_set else 0],
    'spark' : [1 if 'spark' in skill_set else 0],
    'aws' : [1 if 'aws' in skill_set else 0], 
    'excel' : [1 if 'excel' in skill_set else 0],
    'sql' : [1 if 'sql' in skill_set else 0],
    'sas' : [1 if 'sas' in skill_set else 0],
    'keras': [1 if 'keras' in skill_set else 0],
    'pytorch': [1 if 'pytorch' in skill_set else 0],
    'scikit' : [1 if 'scikit' in skill_set else 0],
    'tensor' : [1 if 'tensor' in skill_set else 0],
    'hadoop' : [1 if 'hadoop' in skill_set else 0],
    'tableau' : [1 if 'tableau' in skill_set else 0],
    'bi' : [1 if 'bi' in skill_set else 0],
    'flink' : [1 if 'flink' in skill_set else 0],
    'mongo' : [1 if 'mongo' in skill_set else 0],
    'google_an' : [1 if 'google_an' in skill_set else 0],
    'job_title_sim' : list(enc_title.transform([job_title_sim]))
})

if predict_btn:
    pred = clf.predict(test_data)
    if pred and len(pred) > 0:
        st.sidebar.write("Prediction result: You can expect a salary of $" + str(round(pred[0], 2))+ "k")


df = pd.DataFrame(load_chart_data())
print(df.head(5))

#map
states = alt.topo_feature(data.us_10m.url, 'states')

df_states = pd.DataFrame(df['Job Location'].value_counts())
#df_states['states'] = df_states.index 

map = alt.Chart(states).mark_geoshape().encode(
    color = 'Job Location:Q',
    tooltip=['id:O', 'Job Location:Q']
).transform_lookup(
    lookup='index',
    from_=alt.LookupData(df_states, 'index', ['Job Location']),
).project(
    type='albersUsa'
).properties(
    width=1000,
    height=500
)
st.write(map)



col1, col2 =  st.columns(2)
# revenue chart
df = df[df.Revenue != 'Unknown / Non-Applicable']

revenue_chart = alt.Chart(df, title = 'Number of Companies Per Revenue Bracket').mark_bar().encode(
    alt.X('Revenue'),
    alt.Y('count()', title = 'Company Count'),
    tooltip=['count():O']
).configure_axis(
    titleFontSize=12
).configure_title(
    fontSize=15
).properties(
     width=500,
    height=500
).interactive()
col1.write(revenue_chart)
# st.write(revenue_chart)

# Industries With The Most Available Jobs - pie chart
df_industry = df['Industry'].value_counts()
top_5 = df_industry[0:5]
fig1, ax1 = plt.subplots()
ax1.pie(top_5, labels=top_5.index, autopct='%1.1f%%',)
# ax1.title.set_text('Industries With The Most Available Jobs')
# st.pyplot(fig1)
plt.title('\n Industries With The Most Available Jobs  \n', size=10, color='black')
# col2.write(""" ###### Industries With The Most Available Jobs  """)
col1.pyplot(fig1)

top10_max_job_posting_df = pd.DataFrame({
    'Company name': df["company_txt"].value_counts().sort_values(ascending=False).head(10).index,
    'Company count': df["company_txt"].value_counts().sort_values(ascending=False).head(10)
}) 

# Top 10 Companies with Maximum Number of Job Postings
top10_max_job_posting_chart = alt.Chart(top10_max_job_posting_df, title = 'Top 10 Companies with Maximum Number of Job Postings').mark_bar().encode(
    alt.X('Company name'),
    alt.Y('Company count', title = 'Job Posting Count'),
    tooltip=['Company count'],
    color=alt.Color('Company name', scale=alt.Scale(scheme='dark2'), legend=None)
).configure_axis(
    titleFontSize=12
).configure_title(
    fontSize=15
).properties(
     width=500,
    height=500
).interactive()
col2.write(top10_max_job_posting_chart)

# Top 10 states with avg annual minimal and maximal salaries
sort_ind = df["Location"].value_counts().sort_values(ascending=False).index
ind = df.groupby("Location")["Lower Salary","Upper Salary"].mean().sort_values("Location",ascending=False)
ind = ind.reset_index()
ind["Location"] = ind["Location"].astype("category")
ind["Location"].cat.set_categories(sort_ind, inplace=True)
ind = ind.sort_values(["Location"]).reset_index()
ind = ind.drop("index",axis=1)
ind.head(2)
lab=[]
for i in sort_ind[0:10]:
  lab.append(i)
x = np.arange(len(lab))
width = 0.35
fig2, ax2 = plt.subplots()
rects1 = ax2.bar(x - width/2, ind["Lower Salary"][0:10], width, label='Min avg Salary')
rects2 = ax2.bar(x + width/2, ind["Upper Salary"][0:10], width, label='Max avg Salary')
plt.title('\n Top 10 locations with Average Annual Minimal and Maximal Salaries \n', size=10, color='black')
plt.xticks(fontsize=8)
plt.xticks(rotation=75,ha='right', rotation_mode='anchor')
plt.yticks(fontsize=8)
plt.xlabel('\n Locations \n', fontsize=8, color='black')
plt.ylabel('\n Salary (K) \n', fontsize=8, color='black')
fig2.tight_layout()
ax2.set_xticks(x)
ax2.set_xticklabels(lab)
ax2.legend(loc="upper right")
# fig2.set_figwidth(6)
# fig2.set_figheight(4)

col2.write(fig2)
