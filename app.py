import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io
import base64
from sklearn.preprocessing import StandardScaler

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.DataFrame()
model = None
feature_order = []

app.layout = html.Div([
    html.H4("Upload File"),
    dcc.Upload(id='upload-data', children=html.Button('Upload CSV'), multiple=False),
    html.Div(id='upload-status', children="Please upload a CSV file to begin."),

    html.Br(),
    html.Div([
        html.Label("Select Target:"),
        dcc.Dropdown(id='target-dropdown')
    ]),

    html.Div([
        html.Div([dcc.RadioItems(id='categorical-radio')], style={'width': '100%', 'margin-bottom': '20px'}),
        html.Div([
    dcc.Graph(id='combined-graph', style={'width': '100%', 'maxWidth': '1200px', 'height': '700px'})
], style={'display': 'flex', 'justifyContent': 'center'})
    ]),

    html.Hr(),
    html.Div([
        html.Label("Select features:"),
        dcc.Checklist(id='feature-checklist')
    ]),
    html.Button("Train", id='train-button'),
    html.Div(id='r2-score-output'),

    html.Br(),
    html.Div([
        dcc.Input(id='predict-input', type='text', size = "100", placeholder='Input comma separated values for each paramater. e.g. params = area, basement: 1000000, yes'),
        html.Button('Predict', id='predict-button'),
        html.Div(id='predict-output')
    ])
])

@app.callback(
    Output('target-dropdown', 'options'),
    Output('feature-checklist', 'options'),
    Output('categorical-radio', 'options'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    global df
    if contents is None:
        return [], [], [], "Please upload a CSV file to begin."

    content_type, content_string = contents.split(',')
    decoded = io.BytesIO(base64.b64decode(content_string))
    df = pd.read_csv(decoded)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    return (
        [{'label': col, 'value': col} for col in num_cols],
        [{'label': col, 'value': col} for col in df.columns],
        [{'label': col, 'value': col} for col in cat_cols],
        f"Loaded file: {filename}"
    )

@app.callback(
    Output('combined-graph', 'figure'),
    Input('target-dropdown', 'value'),
    Input('categorical-radio', 'value')
)
def update_combined_graph(target, categorical):
    if target is None or df.empty or categorical is None:
        placeholder = go.Figure().add_annotation(
            text="Please upload a CSV and select a target + categorical column.",
            xref="paper", yref="paper", showarrow=False, font=dict(size=16)
        )
        placeholder.update_layout(height=400)
        return placeholder

    # Subplot structure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Average {target} by {categorical}", f"{target} Correlations")
    )

    # Bar average
    avg_data = df.groupby(categorical)[target].mean().reset_index()
    fig.add_trace(
        go.Bar(x=avg_data[categorical], y=avg_data[target], name='Group Average', marker_color='lightblue'),
        row=1, col=1
    )

    # Correlation bar
    corr_vals = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=corr_vals.index, y=corr_vals.values, name='Correlation Strength', marker_color='blue'),
        row=1, col=2
    )

    fig.update_layout(height=700, width=1200, showlegend=False)
    return fig

@app.callback(
    Output('r2-score-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('target-dropdown', 'value'),
    State('feature-checklist', 'value')
)
def train_model(n_clicks, target, features):
    global model, feature_order
    if n_clicks is None or target is None or not features:
        return ""

    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
    categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
    preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])



    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    y_pred = model.predict(X)
    feature_order = features

    return f"The R2 score is: {r2_score(y, y_pred):.2f}"

@app.callback(
    Output('predict-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('target-dropdown', 'value')

)
def predict_target(n_clicks, value, target):
    if n_clicks is None or model is None or not value:
        return ""

    try:
        input_vals = value.split(',')
        if len(input_vals) != len(feature_order):
            return "Invalid input format."

        data = pd.DataFrame([input_vals], columns=feature_order)
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
            except:
                pass

        pred = model.predict(data)[0]
        return f"Predicted {target} is : {pred:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
