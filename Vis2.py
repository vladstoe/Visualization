from bokeh.models.annotations import Legend
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from bokeh.io import curdoc, show
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Slider, RangeSlider, CustomJS, TextInput, CheckboxGroup, PreText, Dropdown, MultiChoice, FileInput, Panel, Tabs, Button
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

MIN_AGE = 0
MAX_AGE = 20

covid_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
covid_pos_list = covid_pos['SARS-Cov-2 exam result'].count()

viruses = ['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'Rhinovirus/Enterovirus', 'Adenovirus']



def calcDFICU():
    ageStart = range_slider.start
    ageEnd = range_slider.end

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
    ageStart = range_slider.start
    ageEnd = range_slider.end

    selected = df[(df['Patient age quantile'] >= ageStart) & 
        (df['Patient age quantile'] <= ageEnd)]
    print(range_slider.value)
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
        for i in covid:
            if i == 'positive':
                k = k + 1
        count.append(k)

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

    
    return [count, age_unique, regular_count, sicu_count, icu_count]


#Making widgets
range_slider = RangeSlider(start = MIN_AGE, end = MAX_AGE, value = (MIN_AGE, MAX_AGE), step = 1, title = "Age")
text_input = TextInput(value=patient[0], title="Patient:")
checkbox_group = CheckboxGroup(labels=["positive", "negative"], active=[0, 1])
multi_choice = MultiChoice(value=[], options=viruses)
button = Button(label="Update", button_type="success")

pre = PreText(text="""Covid-19 status:""", width=500, height=10)
pre2 = PreText(text="""Choose other viruses:""", width=500, height=10)
pre3 = PreText(text="""Upload file:""", width=500, height=10)

file_input = FileInput()


# Hover tool for Abstract/Explore (V)
hover = """
<div>

<div><strong>Number of infected people: </strong>@count</div>
<div><strong>Number of infected people admitted to Regular Ward: </strong>@regular_count</div>
<div><strong>Number of infected people admitted to Semi-ICU: </strong>@sicu_count</div>
<div><strong>Number of infected people admitted to ICU: </strong>@icu_count</div>

</div>
"""

hover2 = """
<div>

<div><strong>Age: </strong>@Age</div>

</div>
"""

TOOLS = "pan,box_select,wheel_zoom,save,reset"

#Plots
p1Source = ColumnDataSource(data = dict(count = [], age_unique = [], regular_count = [], sicu_count = [], icu_count = []))

p1 = figure(plot_width=800,
    plot_height=600,
    title = "Number of positive cases per age",
    x_axis_label = "Number of positive corona cases",
    y_axis_label = "Age of the patient",
    tools = TOOLS,
    tooltips = hover)
color = 'blue'
p1.vbar(
    x = 'age_unique',
    top = 'count',
    bottom = 0,
    width = 0.5,
    color = color,
    fill_alpha = 0.5,
    source = p1Source
)

p1.xaxis.ticker = list(range(0, 20))

tab1 = Panel(child=p1, title="Plot 1")


p2Source = ColumnDataSource()

p2 = figure(plot_width = 800,
    plot_height = 600,
    title = 'Admission Rate by Age',
    x_range = p1.x_range,
    y_range = p1.y_range,
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

tab2 = Panel(child = p2, title = "Plot 2")

def update():
    p1SourceList = countCases()
   

    p1Source.data = dict(
        count = p1SourceList[0],
        age_unique = p1SourceList[1],
        regular_count = p1SourceList[2],
        sicu_count = p1SourceList[3],
        icu_count = p1SourceList[4]
    )

    p2Source.data = calcDFICU()

    pass


button.on_click(update)


controls = [range_slider, text_input, multi_choice]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())
    

controls.append(checkbox_group)

layout = column(range_slider, text_input, pre, checkbox_group, pre2, multi_choice, pre3, file_input, button)
grid = gridplot([[layout, Tabs(tabs=[tab1, tab2])]])

update()  # initial load of the data

curdoc().add_root(grid)
curdoc().title = "COVID-19 Data Visualizations"

#show(grid)
