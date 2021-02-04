from bokeh.models.annotations import Legend, Title
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from bokeh.io import curdoc, show
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Slider, RangeSlider, CustomJS, TextInput, CheckboxGroup, PreText, Dropdown, MultiChoice, FileInput, Panel, Tabs, Select, Text, LinearAxis
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.models.tools import HoverTool
from bokeh.models.plots import Plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

df =  pd.read_excel('dataset.xlsx')

#Making lists
patient = df['Patient ID'].to_list()
age = df['Patient age quantile'].to_list()
covid = df['SARS-Cov-2 exam result'].to_list()
age_unique = df['Patient age quantile'].unique()
age_unique = sorted(age_unique)

MIN_AGE = 0
MAX_AGE = 19
k = 0
counter = 0

covid_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
covid_pos_list = covid_pos['SARS-Cov-2 exam result'].count()

viruses = ['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'Rhinovirus/Enterovirus', 'Adenovirus']

number = []
for j in viruses:
    counter = 0
    for i in df[df[j] == 'detected']:
        counter = counter + 1
    number.append(str(counter))

def calcDFICU():
    ageStart = range_slider.value[0]
    ageEnd = range_slider.value[1]

    if(patientType.value == 'Positive'):
        covid_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
    elif(patientType.value == 'Negative'):
        covid_pos = df[df['SARS-Cov-2 exam result'] == 'negative']
    else:
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

    temp = df[(df['Patient age quantile'] >= ageStart) & 
        (df['Patient age quantile'] <= ageEnd)]

    if(patientType.value == 'Positive'):
        selected = temp[temp['SARS-Cov-2 exam result'] == 'positive']
    elif(patientType.value == 'Negative'):
        selected = temp[temp['SARS-Cov-2 exam result'] == 'negative']
    else:
        selected = temp

    color = []
    color.append(selectColor.value)


    
    text = []

    n = df['SARS-Cov-2 exam result'][df['Patient ID'] == patient_select.value].to_list()
    text.append(n[0])

    for i in range(1,20):
        color.append(None)
    
    for i in range(1,9):
        text.append(None)

    count = []
    regular_count = []
    sicu_count = []
    icu_count = []
    normalizedCount = []
    selectedPeople = []
    allPeople = []
    testResult = []


    for l in range(0,20):

        x = selected[selected['Patient age quantile'] == l].copy()
        regular = x['Patient addmited to regular ward (1=yes, 0=no)'].to_list()
        sicu = x['Patient addmited to semi-intensive unit (1=yes, 0=no)'].to_list()
        icu = x['Patient addmited to intensive care unit (1=yes, 0=no)'].to_list()

        testResult.append(patientType.value)

        k1 = 0
        k2 = 0
        k3 = 0

        count.append(x.shape[0])

        selectedPeople.append(x.shape[0])
        allPeople.append((temp[temp['Patient age quantile'] == l]).shape[0])

        if(allPeople[l] == 0):
            normalizedCount.append(0)
        else:
            normalizedCount.append(round((selectedPeople[l] / allPeople[l] * 100), 1))
        

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

    
    return [count, age_unique, regular_count, sicu_count, icu_count, color, text, normalizedCount, selectedPeople, allPeople, testResult]

def calcPCA(fig):
    selected = df[df.columns[3:20]]
    #selected['Patient ID'] = df[['Patient ID']]
    selected.dropna(inplace=True)
    selected.reset_index(inplace=True)
        
    selected['Patient addmited to regular ward (1=yes, 0=no)'].replace(0, 'No', inplace=True)
    selected['Patient addmited to regular ward (1=yes, 0=no)'].replace(1, 'Yes', inplace=True)

    selected['Patient addmited to semi-intensive unit (1=yes, 0=no)'].replace(0, 'No', inplace=True)
    selected['Patient addmited to semi-intensive unit (1=yes, 0=no)'].replace(1, 'Yes', inplace=True)

    selected['Patient addmited to intensive care unit (1=yes, 0=no)'].replace(0, 'No', inplace=True)
    selected['Patient addmited to intensive care unit (1=yes, 0=no)'].replace(1, 'Yes', inplace=True)

    features = selected[selected.columns[4:18]]

    features = StandardScaler().fit_transform(features)
    pcaFeatures = PCA(n_components = 2)
    pcaCovid = pcaFeatures.fit_transform(features)

    dfCovidPca = pd.DataFrame(data = pcaCovid, columns = ['Principal Component 1', 'Principal Component 2'])

    targets = ['No', 'Yes']

    target_colors = {
        'Yes': '#1FCC00',
        'No': '#FF0000'
    }

    for target in targets:

        index = ((selected['Patient addmited to regular ward (1=yes, 0=no)'] == target) | 
            (selected['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == target) |
            (selected['Patient addmited to intensive care unit (1=yes, 0=no)'] == target))


        fig.circle(dfCovidPca.loc[index, 'Principal Component 1'], 
            dfCovidPca.loc[index, 'Principal Component 2'], 
            color = mpl.colors.rgb2hex(target_colors[target]),
            alpha = 0.9,
            line_color = 'black',
            line_width = 0.5,
            size = 7,
            legend_label = target
        ) 
        
    return fig

    

#Making widgets
range_slider = RangeSlider(start = MIN_AGE, end = MAX_AGE, value = (MIN_AGE, MAX_AGE), step = 1, title = "Age")
patientType = Select(value = "Positive", options = ["All", "Positive", "Negative"])
multi_choice = MultiChoice(value=[], options=viruses)
patient_select = Select(title="Patient", value=patient[0], options=patient)

colors = ["blue", "red", "green", "black", "yellow", "orange", "purple"]
selectColor = Select(title="Choose the color of the plot:", value="blue", options=colors)


pre = PreText(text="""Covid-19 Status:""", width=500, height=10)
pre2 = PreText(text="""Choose other viruses:""", width=500, height=10)
pre3 = PreText(text="""Upload file:""", width=500, height=10)
pre4 = PreText(text="""COVID-19 test result of the selected patient:""", width=500, height=10)

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

hover3 = """
    <div>

    <div><strong>Age: </strong>@age_unique</div>
    <div><strong>Percentage of People: </strong>%@normalizedCount{0.0}</div>
    <div><strong>Number of People That Are @selection: </strong>@selectedPeople</div>
    <div><strong>Number of People: </strong>@allPeople</div>

    </div>
    """

TOOLS = "pan,box_select,tap,wheel_zoom,save,reset"

#Plots
p1Source = ColumnDataSource(data = dict(count = [], age_unique = [], regular_count = [], sicu_count = [], icu_count = [], color = [], normalizedCount = [], selectedPeople = [], allPeople = [], selection = []))

p1 = figure(plot_width=800,
    plot_height=600,
    title = "Number of Cases by Age",
    x_axis_label = "Age",
    y_axis_label = "Number of Cases",
    tools = TOOLS,
    tooltips = hover
)

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
    fill_alpha = 0.7,
    source = p1Source
)

p1.xaxis.ticker = list(range(0, 20))

tab1 = Panel(child=p1, title="Patient Distribution")



p2 = figure(plot_width=800,
    plot_height=600,
    x_range = p1.x_range,
    title = "Normalized Cases by Age",
    x_axis_label = "Age",
    y_axis_label = "Percentage of People",
    tools = TOOLS,
    tooltips = hover3
)

p2.vbar(
    x = 'age_unique',
    top = 'normalizedCount',
    bottom = 0,
    width = 0.5,
    color = 'color',
    fill_alpha = 0.7,
    source = p1Source
)

p2.xaxis.ticker = list(range(0, 20))

tab2 = Panel(child = p2, title = "Normalized Patient Distribution")

p3Source = ColumnDataSource()

p3 = figure(plot_width = 800,
    plot_height = 600,
    title = 'Admission Rate by Age',
    x_range = p1.x_range,
    y_range = (0, 60),
    y_axis_label = 'Percentage of Admissions',
    x_axis_label = 'Age',
    tools = TOOLS,
    tooltips = hover2
)

p3.vbar(x=dodge('Age', -0.25), top='Regular Ward', width=0.2, source=p3Source,
       color="blue", legend_label="Regular Ward", fill_alpha = 0.7)

p3.vbar(x=dodge('Age',  0.0), top='Semi-ICU', width=0.2, source=p3Source,
       color="orange", legend_label="Semi-ICU", fill_alpha = 0.7)

p3.vbar(x=dodge('Age',  0.25), top='ICU', width=0.2, source=p3Source,
       color="green", legend_label="ICU", fill_alpha = 0.7)


p3.xaxis.ticker = list(range(0, 20))

tab3 = Panel(child = p3, title = "Admission Rate")


fig1 = figure(plot_width = 600,
    plot_height = 600,
    x_axis_label = "Principal Component 1",
    y_axis_label = "Principal Component 2",
    title = "PCA Analyis for Hospital Admission Based on Blood Test Results"
)

tab4 = Panel(child = calcPCA(fig1), title = "PCA for Admission")

N = 9
x = np.linspace(-2, 2, N)
y = x**2
a = "test"
text = [a for i in range(1)]
for i in range(1,9):
    text.append(None)
source = ColumnDataSource(dict(x=x, y=y, text=text))

plot = Plot(
    title=None, plot_width=500, plot_height=600,
    min_border=0, toolbar_location=None)
    
glyph = Text(x="x", y="y", text="text", angle=0.0, text_color="black")
plot.add_glyph(source, glyph)
plot.outline_line_color = None

def update():
    p1SourceList = countCases()
   
    p1Source.data = dict(
        count = p1SourceList[0],
        age_unique = p1SourceList[1],
        regular_count = p1SourceList[2],
        sicu_count = p1SourceList[3],
        icu_count = p1SourceList[4],
        color = p1SourceList[5],
        normalizedCount = p1SourceList[7],
        selectedPeople = p1SourceList[8], 
        allPeople = p1SourceList[9],
        selection = p1SourceList[10]
    )


    source.data = dict(x = x, y = y, text = p1SourceList[6])

    p3Source.data = calcDFICU()


controls = [patient_select, patientType, range_slider, multi_choice, selectColor]

for control in controls:
    control.on_change('value', lambda attr, old, new: update())
    


layout = column(pre, patientType, range_slider, patient_select, selectColor, pre4, plot)
grid = gridplot([[layout, Tabs(tabs=[tab1, tab2, tab3, tab4])]])

update()  # initial load of the data    

curdoc().add_root(grid)
curdoc().title = "COVID-19 Data Visualizations"

#show(grid)
#bokeh serve --show Vis2.py