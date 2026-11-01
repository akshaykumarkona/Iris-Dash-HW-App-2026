import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from dash import Dash, dcc, html, Input, Output

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = pd.Categorical.from_codes(iris.target, iris.target_names)
features = list(iris.feature_names)

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Iris Dataset Explorer'),
    html.Div([
        html.Div([
            html.Label('X-axis'),
            dcc.Dropdown(features, features[0], id='x-axis', clearable=False),
            html.Br(),
            html.Label('Color by Species'),
            dcc.Checklist(options=[{'label': 'Color by Species', 'value': 'yes'}],
                          value=[], id='color-by-species'),
        ], style={'width': '48%'}),
        html.Div([
            html.Label('Y-axis'),
            dcc.Dropdown(features, features[1], id='y-axis', clearable=False),
            html.Br(),
            html.Label('Marker Size'),
            dcc.Slider(5, 20, 1, value=10,
                       marks={i: str(i) for i in (5, 10, 15, 20)}, id='marker-size'),
        ], style={'width': '48%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px'}),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram'),
], style={'padding': '24px', 'maxWidth': '1400px', 'margin': 'auto'})

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('histogram', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('color-by-species', 'value'),
    Input('marker-size', 'value'),
)
def update_plots(x_axis, y_axis, color_choice, marker_size):
    color = 'target' if 'yes' in color_choice else None
    scatter = px.scatter(df, x=x_axis, y=y_axis, color=color,
                         title=f'Iris Dataset: {x_axis} vs {y_axis}')
    scatter.update_traces(marker={'size': marker_size})
    histogram = px.histogram(df, x=y_axis, color=color,
                             title=f'Iris Dataset Histogram: {y_axis}',
                             barmode='group')
    return scatter, histogram

if __name__ == '__main__':
    app.run(debug=True)
