from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import pickle


with open('results_full.pkl', 'rb') as f:
    df = pickle.load(f) 


with open('results_time_full.pkl', 'rb') as f:
    df_time = pickle.load(f) 



app = Dash()

app.layout = [
    html.H1(children='Results', style={'textAlign': 'center'}),
    html.P("Select Dataset"),
    dcc.Dropdown(id="dataset",options=df.Dataset.unique(),value="Sentiment Interpolated",clearable=False),
    dcc.Checklist(df.Method.unique(), df.Method.unique(), id='method-selection', inline=True),
    dcc.Checklist(["SVM","Random Forest","Logistic Regression","Simple NN","generic"], ["Logistic Regression","generic"],id='slider_classifier',inline=True),

    html.P('Range Slider:', style={'font-weight': 'bold'}), 
    dcc.RangeSlider(id='range_slider',min=100,max=10000,step=None,
    marks={500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=[8000,10000]),
    html.P('Slider:', style={'font-weight': 'bold'}),
    dcc.Slider(id='slider',min=100,max=10000,step=None,
    marks={500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=10000),
    dcc.RadioItems(['Range Slider', 'Slider'], 'Slider',id='slider_choice', inline=True),

    dcc.Graph(id='graph-content', style={'width': '1000px', 'height': '800'}),
    dcc.Graph(id='graph-content_2', style={'width': '800px', 'height': '400'}),
    dcc.Checklist(['Show Grid','Cropped X-Axis'], ['Cropped X-Axis'], id='settings', inline=True),


]


@app.callback(
    Output('graph-content', 'figure'),
    Output('graph-content_2', 'figure'),
    Input('method-selection', 'value'),
    Input('settings', 'value'),
    Input('range_slider', 'value'),
    Input('slider_choice', 'value'),
    Input('slider', 'value'),
    Input('dataset','value'),
    Input('slider_classifier', 'value')
)
def update_graph(methods, settings, range_slider,slider_choice,slider,dataset,classifier): #ich schätze die sind nach reihenfolge der "values" bei callback
    dff = df[df['Dataset'] == dataset]
    dff = dff[dff.Method.isin(methods)]
    dff = dff[dff.Classifier.isin(classifier)]
    
    if slider_choice=='Range Slider':
        dff = dff[(dff['Data_size'] >= range_slider[0]) & (df['Data_size'] <= range_slider[1])]
    else:
        dff = dff[dff['Data_size'] == slider]
    #fig = px.line(dff, x="Coverage", y="Accuracy", color='Method')

    if 'Cropped X-Axis' in settings:
        dff = dff[dff['Coverage'] >= 0.5]
        #fig['layout']['xaxis'] = {'range': (1, 0.5)}

    fig = px.line(dff, x="Coverage", y="Accuracy", color='Method_Classifier', line_group='Data_size',line_dash='Data_size')
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


    dff_time = df_time[df_time['Dataset'] == dataset]
    #dff_time = dff_time[dff_time.Method.isin(methods)]
    fig2 = px.scatter(dff_time, x="Score", y="Inference Time",color='Method')

    return fig,fig2
    
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)