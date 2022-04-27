import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from vega_datasets import data

# load clf model
def load_clf_model():
    filename = "finalized_model_clf.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# encoder for location
def load_label_encoder_location():
    filename = "finalized_label_encoder_location.sav"
    return pickle.load(open(filename, 'rb'))

# encoder for sector
def load_label_encoder_sector():
    filename = "finalized_label_encoder_sector.sav"
    return pickle.load(open(filename, 'rb'))

# encoder for title
def load_label_encoder_title():
    filename = "finalized_label_encoder_title.sav"
    return pickle.load(open(filename, 'rb'))

# load chart data
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
# input widgets
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

# Map - US Revenue by State
df_states = pd.DataFrame(df['Job Location'].value_counts())
states = alt.topo_feature(data.us_10m.url, 'states')
source = data.income.url

map = alt.Chart(source).mark_geoshape().encode(
    shape='geo:G',
    color='total:Q',
    tooltip=['name:N', 'total:Q'],
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=states, key='id'),
    as_='geo'
).properties(
    title="US Revenue by State",
    width=1000,
    height=400,
).project(
    type='albersUsa'
)
st.write(map)


# Create two columns
col1, col2 =  st.columns(2)

# Revenue chart
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

# Industries With The Most Available Jobs - pie chart
df_industry = df['Industry'].value_counts()
top_5 = df_industry[0:5]
fig1, ax1 = plt.subplots()
ax1.pie(top_5, labels=top_5.index, autopct='%1.1f%%',)
plt.title('\n Industries With The Most Available Jobs  \n', size=10, color='black')
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

# Top 10 states with avg annual minimum and maximum salaries
sort_index = df["Location"].value_counts().sort_values(ascending=False).index
loc_index = df.groupby("Location")[["Lower Salary","Upper Salary"]].mean().sort_values("Location",ascending=False)
loc_index = loc_index.reset_index()
loc_index["Location"] = loc_index["Location"].astype("category")
loc_index["Location"].cat.set_categories(sort_index, inplace=True)
loc_index = loc_index.sort_values(["Location"]).reset_index()
loc_index = loc_index.drop("index",axis=1)
loc_index.head(2)
label_index=[]
for i in sort_index[0:10]:
  label_index.append(i)
x = np.arange(len(label_index))
width = 0.35
fig2, ax2 = plt.subplots()
rects1 = ax2.bar(x - width/2, loc_index["Lower Salary"][0:10], width, label='Min avg Salary')
rects2 = ax2.bar(x + width/2, loc_index["Upper Salary"][0:10], width, label='Max avg Salary')
plt.title('\n Top 10 locations with Average Annual Minimum and Maximum Salaries \n', size=10, color='black')
plt.xticks(fontsize=8)
plt.xticks(rotation=75,ha='right', rotation_mode='anchor')
plt.yticks(fontsize=8)
plt.xlabel('\n Locations \n', fontsize=8, color='black')
plt.ylabel('\n Salary (K) \n', fontsize=8, color='black')
fig2.tight_layout()
ax2.set_xticks(x)
ax2.set_xticklabels(label_index)
ax2.legend(loc="upper right")

col2.write(fig2)
