from bokeh.models.annotations import Legend, Title
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from bokeh.io import curdoc, show
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Slider, RangeSlider, CustomJS, TextInput, CheckboxGroup, PreText, Dropdown, MultiChoice, FileInput, Panel, Tabs, Select
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.models.tools import HoverTool


df =  pd.read_excel('dataset.xlsx')

#Making lists
patient = df['Patient ID'].to_list()
age = df['Patient age quantile'].to_list()
covid = df['SARS-Cov-2 exam result'].to_list()
age_unique = df['Patient age quantile'].unique()
age_unique = sorted(age_unique)
patient.sort()

MIN_AGE = 0
MAX_AGE = 19
k = 0

covid_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
covid_pos_list = covid_pos['SARS-Cov-2 exam result'].count()

viruses = ['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'Rhinovirus/Enterovirus', 'Adenovirus']



def calcDFICU():
    ageStart = range_slider.value[0]
    ageEnd = range_slider.value[1]

    if(patientType.value == 'Positive'):
        covid_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
    elif(patientType.value == 'Negative'):
        covid_pos = df[df['SARS-Cov-2 exam result'] == 'negative']
    elif(patientType.value == 'All'):
        covid_pos = df

    selected = covid_pos[(covid_pos['Patient age quantile'] >= ageStart) & 
        (covid_pos['Patient age quantile'] <= ageEnd)]

    dfICU = selected[['Patient age quantile', 
        'Patient addmited to regular ward (1=yes, 0=no)', 
        'Patient addmited to semi-intensive unit (1=yes, 0=no)',
        'Patient addmited to intensive care unit (1=yes, 0=no)']]

    dfICU.columns = ['Age', 'Regular Ward', 'Semi-ICU', 'ICU']

    regWard = dfICU.groupby(['Age', 'Regular Ward']).size().unstack(fill_value=0).reset_index()
    regWard['% of 1'] = (regWard[1]/(regWard[1]+regWard[0])).round(2) 

    SICU = dfICU.groupby(['Age', 'Semi-ICU']).size().unstack(fill_value=0).reset_index()
    SICU['% of 1'] = (SICU[1]/(SICU[1]+SICU[0])).round(2)

    ICU = dfICU.groupby(['Age', 'ICU']).size().unstack(fill_value=0).reset_index()  
    ICU['% of 1'] = (ICU[1]/(ICU[1]+ICU[0])).round(2)

    dfICU = dfICU.groupby('Age').count()
    dfICU['Regular Ward'] = regWard['% of 1']*100
    dfICU['Semi-ICU'] = SICU['% of 1']*100
    dfICU['ICU'] = ICU['% of 1']*100

    return dfICU

# Set up data

def getDf():

    return df

#Algorithm to count cases
def countCases():
    ageStart = range_slider.value[0]
    ageEnd = range_slider.value[1]

    if(patientType.value == 'Positive'):
        temp = df[df['SARS-Cov-2 exam result'] == 'positive']
    elif(patientType.value == 'Negative'):
        temp = df[df['SARS-Cov-2 exam result'] == 'negative']
    else:
        temp = df

    color = []
    color.append(selectColor.value)
    for i in range(1,20):
        color.append(None)

    selected = temp[(temp['Patient age quantile'] >= ageStart) & 
        (temp['Patient age quantile'] <= ageEnd)]
    
    k=0
    count = []
    regular_count = []
    sicu_count = []
    icu_count = []

    for l in range(0,20):
        x = selected[selected['Patient age quantile'] == l].copy()
        covid = x['SARS-Cov-2 exam result'].to_list()
        regular = x['Patient addmited to regular ward (1=yes, 0=no)'].to_list()
        sicu = x['Patient addmited to semi-intensive unit (1=yes, 0=no)'].to_list()
        icu = x['Patient addmited to intensive care unit (1=yes, 0=no)'].to_list()
        k = 0
        k1 = 0
        k2 = 0
        k3 = 0

        count.append(x.shape[0])

        for i in regular:
            if i == int('1'):
                k1 = k1 + 1
        regular_count.append(k1)

        for i in sicu:
            if i == int('1'):
                k2 = k2 + 1
        sicu_count.append(k2)

        for i in icu:
            if i == int('1'):
                k3 = k3 + 1
        icu_count.append(k3)

    
    return [count, age_unique, regular_count, sicu_count, icu_count, color]


#Making widgets
range_slider = RangeSlider(start = MIN_AGE, end = MAX_AGE, value = (MIN_AGE, MAX_AGE), step = 1, title = "Age")
patientType = Select(value = "Positive", options = ["All", "Positive", "Negative"])
multi_choice = MultiChoice(value=[], options=viruses)

colors = ["blue", "red", "green", "black", "yellow", "orange", "purple"]
selectColor = Select(title="Choose the color of the plot:", value="blue", options=colors)


pre = PreText(text="""Covid-19 status:""", width=500, height=10)
pre2 = PreText(text="""Choose other viruses:""", width=500, height=10)
pre3 = PreText(text="""Upload file:""", width=500, height=10)

file_input = FileInput()


# Hover tool for Abstract/Explore (V)
hover = """
    <div>

    <div><strong>Number of people: </strong>@count</div>
    <div><strong>Number of people admitted to Regular Ward: </strong>@regular_count</div>
    <div><strong>Number of people admitted to Semi-ICU: </strong>@sicu_count</div>
    <div><strong>Number of people admitted to ICU: </strong>@icu_count</div>

    </div>
    """

hover2 = """
    <div>

    <div><strong>Age: </strong>@Age</div>

    </div>
    """

TOOLS = "pan,box_select,wheel_zoom,save,reset"

#Plots
p1Source = ColumnDataSource(data = dict(count = [], age_unique = [], regular_count = [], sicu_count = [], icu_count = [], color = []))

p1 = figure(plot_width=800,
    plot_height=600,
    title = "Number of positive cases per age",
    x_axis_label = "Age of the patient",
    y_axis_label = "Number of positive corona cases",
    tools = TOOLS,
    tooltips = hover)

color = []
color.append('blue')

for i in range(1,20):
    color.append(None)

p1.vbar(
    x = 'age_unique',
    top = 'count',
    bottom = 0,
    width = 0.5,
    color = 'color',
    fill_alpha = 0.5,
    source = p1Source
)

p1.xaxis.ticker = list(range(0, 20))

tab1 = Panel(child=p1, title="Age Distribution")


p2Source = ColumnDataSource()

p2 = figure(plot_width = 800,
    plot_height = 600,
    title = 'Admission Rate by Age',
    x_range = p1.x_range,
    y_range = (0, 50),
    y_axis_label = 'Percentage of Admissions',
    x_axis_label = 'Age of the Patient',
    tools = TOOLS,
    tooltips = hover2
)

p2.vbar(x=dodge('Age', -0.25), top='Regular Ward', width=0.2, source=p2Source,
       color="blue", legend_label="Regular Ward", fill_alpha = 0.8)

p2.vbar(x=dodge('Age',  0.0), top='Semi-ICU', width=0.2, source=p2Source,
       color="orange", legend_label="Semi-ICU", fill_alpha = 0.8)

p2.vbar(x=dodge('Age',  0.25), top='ICU', width=0.2, source=p2Source,
       color="green", legend_label="ICU", fill_alpha = 0.8)


p2.xaxis.ticker = list(range(0, 20))

tab2 = Panel(child = p2, title = "Admission Rate")

def calcTitle():

    t = Title()

    if(patientType.value == 'Positive'):
        t.text = 'Number of Positive Cases by Age'
    elif(patientType.value == 'Negative'):
        t.text = 'Number of Negative Cases by Age'
    else:
        t.text = 'All Patients by Age'

    return t

def update():
    p1SourceList = countCases()
   
    p1Source.data = dict(
        count = p1SourceList[0],
        age_unique = p1SourceList[1],
        regular_count = p1SourceList[2],
        sicu_count = p1SourceList[3],
        icu_count = p1SourceList[4],
        color = p1SourceList[5]
    )

    p1.title = calcTitle()

    p2Source.data = calcDFICU()

    pass

controls = [patientType, range_slider, multi_choice, selectColor]

for control in controls:
    control.on_change('value', lambda attr, old, new: update())
    


layout = column(pre, patientType, range_slider, pre2, multi_choice, pre3, file_input, selectColor)
grid = gridplot([[layout, Tabs(tabs=[tab1, tab2])]])

update()  # initial load of the data

curdoc().add_root(grid)
curdoc().title = "COVID-19 Data Visualizations"

#show(grid)
#bokeh serve --show Vis2.py