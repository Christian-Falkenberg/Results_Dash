from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
from plotly.subplots import make_subplots
import pickle


with open('results_full.pkl', 'rb') as f:
    df = pickle.load(f) 


with open('results_time_full.pkl', 'rb') as f:
    df_time = pickle.load(f) 



app = Dash()

app.layout = [
    html.H1(children='Results', style={'textAlign': 'center'}),
    html.P("Select Dataset"),
    dcc.Dropdown(id="dataset",options=df[~df['Dataset'].str.contains("Interpolated")]['Dataset'].unique(),value="Sentiment",clearable=False),
    dcc.Checklist(df.Method.unique(), [method for method in df.Method.unique() if method not in ["Base Embeddings", "Base Outlier"]], id='method-selection', inline=True),
    dcc.Checklist(["SVM","Random Forest","Logistic Regression","Simple NN","generic"], ["Logistic Regression","generic"],id='slider_classifier',inline=True),

    html.P('Range Slider:', style={'font-weight': 'bold'}), 
    dcc.RangeSlider(id='range_slider',min=100,max=10000,step=None,
    marks={500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=[8000,10000]),
    html.P('Slider:', style={'font-weight': 'bold'}),
    dcc.Slider(id='slider',min=100,max=10000,step=None,
    marks={500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'},value=10000),
    dcc.RadioItems(['Range Slider', 'Slider'], 'Slider',id='slider_choice', inline=True),
    dcc.RadioItems(['Interpolated', 'Raw'], 'Interpolated',id='Interpolated', inline=True),

    html.Div(
        style={'display': 'flex', 'margin-top': '20px'},
        children=[
            dcc.Graph(id='graph-content', style={'width': '1000px', 'height': '400px'}),
            html.Div(
                id='explanation-window',
                style={
                    'border': '1px solid black',
                    'padding': '10px',
                    'margin-left': '20px',
                    'width': '200px',
                    'height': '150px',
                    'overflow-y': 'auto'
                }
            )
        ]
    ),
    dcc.Graph(id='graph-content_2', style={'width': '1600px', 'height': '600'}),
    dcc.Checklist(['Show Grid','Cropped X-Axis'], ['Cropped X-Axis'], id='settings', inline=True),
    #html.Div(id='explanation-window', style={'border': '1px solid black', 'padding': '10px', 'margin-top': '10px'})


]


@app.callback(
    Output('graph-content', 'figure'),
    Output('graph-content_2', 'figure'),
    Output('explanation-window', 'children'),
    Input('method-selection', 'value'),
    Input('settings', 'value'),
    Input('range_slider', 'value'),
    Input('slider_choice', 'value'),
    Input('slider', 'value'),
    Input('dataset','value'),
    Input('slider_classifier', 'value'),
    Input('Interpolated', 'value'),
)
def update_graph(methods, settings, range_slider,slider_choice,slider,dataset,classifier,format): #ich schätze die sind nach reihenfolge der "values" bei callback

    if dataset == "Sentiment":
        explanation = "Multiclass (pos,neutral,neg) Sentiment Analysis Dataset"
    elif dataset == "Question Answering":
        explanation = "Context-Question-Answer (SQUAD) Dataset"
    elif dataset == "AG_News":
        explanation = "AG News Classification Dataset"
    elif dataset == "Spam":
        explanation = "Email Spam Classification Dataset mit Email TITEL als Input"
    elif dataset == "Spam_text":
        explanation = "Email Spam Classification Dataset mit Email INHALT als Input"
    elif dataset == "Regression":
        explanation = "Movie Review Sentiment als Regression - Model predicted Sternenanzahl als float value"
    elif dataset == "Transformation":
        explanation = "Spell Checker - Model korrigert Rechtschreibung"
    elif dataset == "Time Series Regression":
        explanation = "Periodic Time Series data - Input: 5 Values, Predict the next - Correct if distance to label is close enough"
    elif dataset == "Merged":
        explanation = "Alle Interpolated Datasets Merged, ohne RunTime Merge"
        format = "Raw" # Merged ist eigentlich Interpolated Merge, aber für Anzeige irrelevant 
    else:
        explanation = "Erklärung kommt noch"

    if format == "Interpolated":
        dataset = dataset + " Interpolated"

    dff = df[df['Dataset'] == dataset]
    dff = dff[dff.Method.isin(methods)]
    dff = dff[dff.Classifier.isin(classifier)]
    
    if slider_choice=='Range Slider':
        dff = dff[(dff['Data_size'] >= range_slider[0]) & (df['Data_size'] <= range_slider[1])]
    else:
        dff = dff[dff['Data_size'] == slider]

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

    #Run Time Graphen 
    #----------------------------------------------------------------------------
    if format == "Raw":
        dff_time = df_time[df_time['Dataset'] == dataset+ " Interpolated"] #Scores und Runtime Values innerhalb Interpolated Data gespeichert, nur zum Zugreifen
    else:
        dff_time = df_time[df_time['Dataset'] == dataset]
        
    if slider_choice=='Range Slider':
        dff_time = dff_time[(dff_time['Data_size'] >= range_slider[0]) & (dff_time['Data_size'] <= range_slider[1])]
    else:
        dff_time = dff_time[dff_time['Data_size'] == slider]
    

    dff_time = dff_time[dff_time.Method.isin(methods)]
    dff_time = dff_time[dff_time.Classifier.isin(classifier)]


    # Create subplots: 1 row, 2 columns
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Inference Time vs Score", "Initial Time vs Score"))

    # Create scatter plots
    inference_fig = px.scatter(dff_time, x="Score", y="Inference Time", color='Method', symbol='Data_size')
    initial_fig = px.scatter(dff_time, x="Score", y="Initial Time", color='Method', symbol='Data_size')

    # Add traces to subplots
    for trace in inference_fig.data:
        fig2.add_trace(trace, row=1, col=1)

    for trace in initial_fig.data:
        fig2.add_trace(trace, row=1, col=2)

    fig2.update_layout(title_text="Score X Inference and Initial Time X Data sizes(Range Slider auswählen)",)
    fig2.update_xaxes(title_text="Score", row=1, col=1)
    fig2.update_yaxes(title_text="Inference Time in seconds", row=1, col=1)
    fig2.update_xaxes(title_text="Score", row=1, col=2)
    fig2.update_yaxes(title_text="Initial Time in seconds", row=1, col=2)

    return fig,fig2,explanation
    
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)