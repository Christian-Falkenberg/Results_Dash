from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import pickle



with open('results_sentiment_full.pkl', 'rb') as f:
    df_sentiment = pickle.load(f)                         #Dataset 	Data_size 	Method 	Accuracy 	Coverage

with open('results_spam_full.pkl', 'rb') as f:
    df_spam = pickle.load(f) 

with open('results_QA_full.pkl', 'rb') as f:
    df_QA = pickle.load(f) 


df = pd.concat([df_sentiment,df_spam,df_QA], ignore_index=True)


app = Dash()

app.layout = [
    html.H1(children='Results', style={'textAlign': 'center'}),
    html.P("Select Dataset"),
    dcc.Dropdown(id="dataset",options=df.Dataset.unique(),value="Sentiment",clearable=False),

    dcc.Checklist(df.Method.unique(), df.Method.unique(), id='method-selection', inline=True),
    dcc.Checklist(['Show Grid','Cropped X-Axis'], ['Cropped X-Axis'], id='settings', inline=True),
    dcc.Graph(id='graph-content', style={'width': '1000px', 'height': '800'}),

    html.P('Range Slider:', style={'font-weight': 'bold'}), 
    dcc.RangeSlider(id='range_slider',min=100,max=10000,step=None,
    marks={100: '100', 250: '250', 500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=[8000,10000]),

    html.P('Slider:', style={'font-weight': 'bold'}),
    dcc.Slider(id='slider',min=100,max=10000,step=None,
    marks={100: '100', 250: '250', 500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=10000),

    dcc.RadioItems(['Range Slider', 'Slider'], 'Slider',id='slider_choice')
]


@app.callback(
    Output('graph-content', 'figure'),
    Input('method-selection', 'value'),
    Input('settings', 'value'),
    Input('range_slider', 'value'),
    Input('slider_choice', 'value'),
    Input('slider', 'value'),
    Input('dataset','value')
)
def update_graph(methods, settings, range_slider,slider_choice,slider,dataset): #ich schätze die sind nach reihenfolge der "values" bei callback
    dff = df[df['Dataset'] == dataset]
    dff = dff[dff.Method.isin(methods)]
    
    if slider_choice=='Range Slider':
        dff = dff[(dff['Data_size'] >= range_slider[0]) & (df['Data_size'] <= range_slider[1])]
    else:
        dff = dff[dff['Data_size'] == slider]
    #fig = px.line(dff, x="Coverage", y="Accuracy", color='Method')

    if 'Cropped X-Axis' in settings:
        dff = dff[dff['Coverage'] > 0.5]
        #fig['layout']['xaxis'] = {'range': (1, 0.5)}

    fig = px.line(dff, x="Coverage", y="Accuracy", color='Method', line_group='Data_size',line_dash='Data_size')
    fig.update_xaxes(autorange='reversed')  # Invert y-axis if desired
    fig.update_layout(plot_bgcolor='white')
    
    # Update grid lines based on checkbox selection
    show_grid = 'Show Grid' in settings
    fig.update_xaxes(showgrid=show_grid, gridcolor='grey')  # X-axis grid color
    fig.update_yaxes(showgrid=show_grid, gridcolor='grey')  # Y-axis grid color


    #fig['layout']['xaxis'] = {'range': (1, 0.5)}
    #ymin, ymax = min(dff["Accuracy"]), max(dff["Accuracy"])
    #print(ymin,ymax)
    #fig['layout']['yaxis'] = {'range': (ymin, ymax)}



    return fig

if __name__ == '__main__':
    app.run(debug=True)