import pandas as pd
import warnings
from dash import Dash, Input, Output, dcc, html, State, dash_table
import plotly.express as px
import numpy as np
from pymongo import MongoClient
import os
import re
from datetime import datetime,timedelta,date
from dotenv import load_dotenv
from pprint import pprint
import dash_mantine_components as dmc
from difflib import SequenceMatcher
import calendar


warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client['Flights']
arrival_collection = db['arrivals_barcelona']
departure_collection = db['departures_barcelona']

def get_data(type,start_str=None, end_str=None):
    
    if start_str and end_str:
        query = {'date' : {"$gt" : start_str, "$lte" : end_str }}
    else:
        query = {}
        
    if type == 'arrivals':
        df_query = pd.DataFrame(list(arrival_collection.find(query).sort("date", 1)))
    elif type == 'departures':
        df_query = pd.DataFrame(list(departure_collection.find(query).sort("date", 1)))   
    
    df = pd.DataFrame()
    
    for document in df_query['flights']:
        df = df.append(pd.DataFrame(document))
        
    df.drop(df[(df['status'] == 'Scheduled') | (df['iata'] == 'MMM')].index, inplace = True)
            
    df['airline'] = df['airline'].str.replace(r' [0-9]*', '')   
    df['status'] = df['status'].str.replace(r'Cancelled.*', 'Cancelled')
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x,'%d %b %Y').date())
    df.reset_index(inplace=True,drop=True)
    return df

def clean_departures(df):
    df = df[df['status'] != 'Departed']
    return df

def get_time(time_str):
    try:
        match = re.search(r'([0-2][0-9]:[0-5][0-9])', time_str)
        return match.group(0)
    except:
        return None

def pos_check(x, zero_value=pd.Timedelta(0,'s')):
    return pd.Timedelta(x) >= zero_value

def get_series_range(series,start,end):
    sum = 0
    for i in range(start,end):
        sum += series.get(i,0)
    return sum

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def normalize_df(df):
    df = df[['time','weekday','iata','arrival/departure','airline']].copy()
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df = encode(df,'weekday',7)
    df.drop('weekday', axis=1, inplace=True)
    return df

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def similarity(row_df, compare_df):
    ponderation = {
        'time' : .3,
        'weekday' : .5,
        'iata' : .1,
        'airline' : .05,
    }
    similar_df = pd.DataFrame(columns=['time','weekday','iata','arrival/departure','airline'])
    similar_df['time'] = compare_df['time'].apply(lambda x: abs(x - row_df['time'].iloc[0]))/1440*ponderation['time']
    similar_df['weekday_cos'] = compare_df['weekday_cos'].apply(lambda x: abs(x - row_df['weekday_cos'].iloc[0]))
    similar_df['weekday_sin'] = compare_df['weekday_sin'].apply(lambda x: abs(x - row_df['weekday_sin'].iloc[0]))
    similar_df['weekday'] = (similar_df['weekday_cos'] + similar_df['weekday_sin'])/2.740174*ponderation['weekday']
    similar_df['iata'] = compare_df['iata'].apply(lambda x: 1 - similar(x, row_df['iata'].iloc[0])*ponderation['iata'])//1
    similar_df['airline'] = compare_df['airline'].apply(lambda x: 1 - similar(x, row_df['airline'].iloc[0]))*ponderation['airline']//1
    similar_df['iata'] = similar_df['iata']*ponderation['iata']
    similar_df['airline'] = similar_df['airline']*ponderation['airline']
    similar_df.drop(['weekday_cos','weekday_sin'], axis=1, inplace=True)
    similar_df.drop_duplicates(inplace=True)
    similar_df['score'] = similar_df.sum(axis=1)
    
    similar_df.sort_values(by='score', ascending=True, inplace=True)
    return similar_df.index[:10]

def airport_graphs(df_arrivals,df_departures):
    #Concurrency Logic
    #Reduce dataframe dimensions to make computation faster.
    df_arrivals = df_arrivals[['time','arrival/departure','airline','status']].copy()
    df_departures = df_departures[['time','arrival/departure','airline','status']].copy()

    #Get the hour string from status and convert to datetime object.
    df_arrivals['status_time'] = pd.to_datetime(df_arrivals['status'].apply(lambda x: get_time(x)))
    df_departures['status_time'] = pd.to_datetime(df_departures['status'].apply(lambda x: get_time(x)))
    df_departures['time'] = pd.to_datetime(df_departures['time'])
    #Add to Dataframe for visualization
    df_concurrency = pd.DataFrame()
    df_concurrency['Arrivals'] = df_arrivals.groupby(df_arrivals['status_time'].dt.hour).size()
    df_concurrency['Departures'] = df_departures.groupby(df_departures['status_time'].dt.hour).size()
    fig_concurrency = px.bar(
        df_concurrency,
        x=df_concurrency.index,
        y=['Arrivals','Departures'],
        title='Airport Concurrency',
        labels={'value':'Number of Flights','status_time':'Hour of the Day','variable':'Type of Flight'},
        color_discrete_sequence=[px.colors.sequential.Aggrnyl[0],px.colors.sequential.Aggrnyl[3]]
        )
    
    #Delay graph logic
    time_df = df_departures[['time','status_time']].dropna()
    time_df['time_dif'] = time_df['status_time'] - time_df['time']
    time_df['isLate']  = time_df['time_dif'].apply(lambda x: pos_check(x, zero_value=pd.Timedelta(0,'s')))
    min_late = time_df[time_df['isLate'] == True]['time_dif'].dt.total_seconds()/60
    dec_late = min_late//10
    dec_late = dec_late.value_counts().sort_index()
    
    dec_late_dict = {
    '0-10 minutes' : dec_late[0],
    '10-20 minutes' : dec_late[1],
    '20-30 minutes' : dec_late[2],
    '30-60 minutes': dec_late[3:5].sum(),
    '60-120 minutes' : dec_late[6:12].sum(),
    '120+ minutes' : dec_late[13:].sum()
    }   
    
    dec_late_df = pd.DataFrame(dec_late_dict.items(), columns=['Time', 'Count'])
    fig_delays = px.pie(
        dec_late_df,values='Count',
        names='Time',
        hole=0.3,
        title='Departure Delays',
        color_discrete_sequence=px.colors.sequential.Aggrnyl
        ,)

    #Airline Volume Graph Logic
    airline_df = pd.DataFrame()
    airline_df['Arrivals'] = df_arrivals['airline'].value_counts()
    airline_df['Departures'] = df_arrivals['airline'].value_counts()
    
    fig_airlines=px.bar(
        airline_df.iloc[:10],
        x=airline_df.iloc[:10].index,
        y=['Arrivals','Departures'],
        title='Popular Airlines',
        labels={'value':'Number of Flights','x':'Airline','variable':'Type of Flight'},
        color_discrete_sequence=[px.colors.sequential.Aggrnyl[0],px.colors.sequential.Aggrnyl[3]]
    )
    
    #Location Volume Graph Logic
    location_df = pd.DataFrame()
    location_df['Arrivals'] = df_arrivals['arrival/departure'].value_counts()
    location_df['Departures'] = df_departures['arrival/departure'].value_counts()
    
    fig_location = px.bar(
        location_df.iloc[:10],
        x=location_df.index[:10],
        y=['Arrivals','Departures'],
        title='Popular Locations',
        labels={'value':'Number of Flights','x':'Location','variable':'Type of Flight'},
        color_discrete_sequence=[px.colors.sequential.Aggrnyl[0],px.colors.sequential.Aggrnyl[3]]
        )
    
    return html.Div([
        html.Div([
            html.Div([
                    html.H2(f'Total Flights: {len(df_departures)}'),
                    html.H2(f'Cancelled Flights: {len(df_departures[df_departures["status"]=="Cancelled"])}'),
                    html.H2(f'Average Delay (min): {round(min_late.mean(),2)}'),
                ],className='airport-stats-info-div'),
            dcc.Graph(
                figure = fig_concurrency,
                responsive=True,
                className='airport-concurrency-graph'
                ),
            dcc.Graph(
                figure=fig_delays,
                responsive=True,
                className='airport-delay-graph'
                ),
        ], className='airport-delay-div'),
        html.Div([
            dcc.Graph(
                figure=fig_location,
                responsive=True,
                className='airport-location-graph'
                ),
            dcc.Graph(
                figure=fig_airlines,
                responsive=True,
                className='airport-airline-graph'
                ),
        ],className='airport-popular-div'),
    ], style={})
    
data_arrivals = get_data('arrivals')
data_departures = clean_departures(get_data('departures'))

airlines = sorted(data_arrivals['airline'].unique())

external_stylesheets = [{
}]

app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
app.title = 'Barcelona Flight Dashboard'
server = app.server

app.layout = html.Div([
    html.Div([
            html.Div([
                    html.Img(src="https://cdn4.iconfinder.com/data/icons/aiga-symbol-signs/586/aiga_departingflights_inv-512.png", className="logo"),
                ], className="header-logo-container"),
            html.Div([
                html.H1(
                    children="Barcelona Flight Dashboard", className="header-title"
                ),
                html.P(
                    children=(
                        "This dashboard is a tool to visualize the flights that arrive and depart from Barcelona Airport."
                    ),
                    className="header-description",
                ),
            ],className="header-title-container"),
            ], className="header",
            ),
    html.Div([
        
        html.H1('El Prat Aiport Statistics'),
        airport_graphs(data_arrivals,data_departures),
    
        ],className='airport-stat-div'),
    
    html.Div([
        html.H1('Flight Recommendation'),
        html.Div([
            html.Div('Airline'),
            dcc.Dropdown(
                            id="airline-filter",
                            options=[
                                {"label": airline, "value": airline}
                                for airline in airlines
                                #Need to get airlines values from database
                            ],
                            value="Ryanair",
                            clearable=False,
                            className="dropdown",
                        ),
            html.Div('Location'),
            dcc.Dropdown(
                            id="location-filter",
                            value="Paris",
                            clearable=False,
                            className="dropdown",
                        ),
            html.Div('Airport'),
            dcc.Dropdown(
                    id='location-airport',
                    value='BVA',
                    clearable=False,
                    className='dropdown',
                ),
            html.Div(
                        children="Travel Date", className="menu-title"
                        ),
            dcc.DatePickerSingle(
                id='travel_date',
                min_date_allowed=date.today(),
                max_date_allowed=date(2024, 12, 30),
                initial_visible_month=date.today(),
                date= date.today(),
                clearable=False,
                className='date-picker',
            ),
            html.Div(
                        children="Travel Time", className="menu-title"
                        ),
            dmc.TimeInput(
                format="24",
                id="travel_hour",
                clearable=False,
                value=datetime.now().hour,
                className='time-input',
            ),
            html.Button(
                id='submit-button',children='Submit',n_clicks=0,className='submit-button'
                ),
        ], className='airline-config-div'),
        html.Div([            
            html.Div([
                html.Div('Recommended Flights:'),
                dash_table.DataTable(
                    id='datatable-recommendation',
                    columns=[
                        {"name": i.capitalize(), "id": i} for i in ['weekday','to','airport','airline','time']
                    ],
                    page_current=0,
                    page_size=10,
                ),
            ],className='datatable-recommendation'),
            dcc.Graph(id='late-pie',className='late-pie-graph'),
        ],className='recommendation-graphs-div'),
        html.Div([
                html.Div('Past Flights:'),
                dash_table.DataTable(
                    id='datatable-paging',
                    columns=[
                        {"name": i.capitalize(), "id": i} for i in sorted(data_arrivals.columns)
                    ],
                    page_current=0,
                    page_size=6,
                )
            ],className='datatable-flights'),
        
    ],className='recommendation-div')

#End
],className='main-div')

#Inputs
#Start date
#End Date
#Airline


@app.callback(
    Output('location-filter', 'options'),
    Input('airline-filter', 'value'))
def set_location_options(airline):
    options = sorted(data_departures[data_departures['airline'] == airline]['arrival/departure'].unique())
    return [{'label': i, 'value': i} for i in options]

@app.callback(
    Output('location-airport', 'options'),
    Input('location-filter', 'value'),
    Input('airline-filter', 'value'))

def set_location_airport_options(location,airline):
    options = sorted(data_departures[(data_departures['arrival/departure'] == location) & (data_departures['airline'] == airline)]['iata'].unique())
    return [{'label': i, 'value': i} for i in options]


@app.callback(
    [Output(component_id='datatable-paging',component_property='data'),
    Output(component_id='late-pie',component_property='figure'),
    Output(component_id='datatable-recommendation',component_property='data')],
    [Input(component_id='submit-button',component_property='n_clicks'),
    State(component_id='airline-filter',component_property='value'),
    State(component_id='location-filter',component_property='value'),
    State(component_id='location-airport',component_property='value'),
    State(component_id='travel_date',component_property='date'),
    State(component_id='travel_hour',component_property='value'),
    ]
    #,prevent_initial_call=True
)
def update_graph(submit,airline,location,airport_iata,travel_date,travel_hour):
    graph_data = data_departures.query(f"airline == @airline and `arrival/departure` == @location")
    
    time_df = graph_data[['time','status']].copy()
    time_df['status_time'] = time_df['status'].apply(lambda x: get_time(x))
    #This sends message of copy of slice warning
    time_df['time'] = pd.to_datetime(time_df['time'])
    time_df['status_time'] = pd.to_datetime(time_df['status_time'])
    time_df['dif'] = time_df['status_time'] - time_df['time']
    time_df['isLate'] = time_df['dif'].apply(lambda x: pos_check(x))
    late_time = time_df[time_df['isLate'] == True]['dif'].dt.total_seconds()/60//10
    late_time = late_time.value_counts().sort_index()
        
    dec_late_dict = {
        '0-10 minutes' : late_time.get(0,0),
        '10-20 minutes' : late_time.get(1,0),
        '20-30 minutes' : late_time.get(2,0),
        '30-60 minutes': get_series_range(late_time,3,6),
        '60-120 minutes' : get_series_range(late_time,6,12),
        '120+ minutes' : get_series_range(late_time,12,int(late_time.index.max()))
    }
    
    dec_late_df = pd.DataFrame(dec_late_dict.items(), columns=['Time', 'Count'])
    pie_fig = px.pie(dec_late_df,values='Count', names='Time',hole=0.3, title='Airline Delays',color_discrete_sequence=px.colors.sequential.Aggrnyl)
    
    #travel date from hour string to datetime
    travel_date = datetime.strptime(travel_date, '%Y-%m-%d').date()
    
    user_selection_df = pd.DataFrame(columns=['time','weekday','iata','arrival/departure','airline'])
    user_selection_df.loc[0] = [travel_hour,travel_date.weekday(),airport_iata,location,airline]
    
    recommendation_df = data_departures.query(f"`arrival/departure` == @location").copy()
    recommendation_df['weekday'] = recommendation_df['date'].apply(lambda x: x.weekday())  
    
    similarity_indexes = similarity(normalize_df(user_selection_df),normalize_df(recommendation_df))
    recommendation_df = recommendation_df.loc[similarity_indexes]
    recommendation_df = recommendation_df[['time','weekday','iata','arrival/departure','airline']]
    recommendation_df['weekday'] = recommendation_df['weekday'].apply(lambda x: calendar.day_name[x])
    recommendation_df.rename(columns={'iata':'airport','arrival/departure':'to'},inplace=True)
    
    return graph_data.to_dict('records'), pie_fig, recommendation_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)