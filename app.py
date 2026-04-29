import pandas as pd
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# 1. Load Data
df = pd.read_csv("wb_election_data.csv")

# 2. Setup App (Basic)
app = dash.Dash(__name__)
app.title = "Govt of WB - Exit Poll Data"
# Suppress dash callback exceptions if needed
app.config.suppress_callback_exceptions = True

# Colors mapping
party_colors = {
    'AITC': '#20C646',
    'BJP': '#FF9933',
    'CPI(M)': '#DE1100',
    'INC': '#19AAED',
    'Others': '#808080'
}
parties = [p for p in df['Party'].unique() if p != 'Others']

# 3. Layout (Amateur gov style)
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'color': '#000000'}, children=[
    html.Div(style={'borderBottom': '3px solid #000080', 'marginBottom': '20px'}, children=[
        html.H1("Election Data Analysis", style={'color': '#000080', 'margin': '0'}),
        html.H2("West Bengal Assembly Election Predictor", style={'color': '#000000', 'marginTop': '5px'})
    ]),
    
    html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
        html.Tr([
            # Left Sidebar
            html.Td(style={'width': '30%', 'verticalAlign': 'top', 'borderRight': '2px solid #000000', 'paddingRight': '20px'}, children=[
                html.Div(style={'border': '1px solid #000000', 'padding': '15px', 'backgroundColor': '#F0F0F0'}, children=[
                    html.H3("Vote Share Adjustments", style={'marginTop': '0'}),
                    html.P(html.I("Enter manual swing offsets based on field reports:")),
                    
                    html.Div([
                        html.Div([
                            html.Label(f"{p} Swing (%): ", style={'fontWeight': 'bold'}),
                            dcc.Slider(
                                id=f'swing-{p}',
                                min=-40, max=40, step=0.5, value=0,
                                marks={i: str(i) for i in range(-40, 41, 10)}
                            )
                        ], style={'marginBottom': '30px'}) for p in parties
                    ])
                ]),
                
                html.Div(style={'border': '2px solid #000000', 'padding': '15px', 'backgroundColor': '#FFFFCC', 'marginTop': '20px'}, children=[
                    html.H3("Prediction Result", style={'marginTop': '0'}),
                    html.P([html.B("Leading Party: "), html.Span(id="winner-name", style={'fontWeight': 'bold', 'fontSize': '18px'})]),
                    html.P([html.B("Status: "), html.Span(id="winner-status")])
                ])
            ]),
            
            # Right Content
            html.Td(style={'width': '70%', 'verticalAlign': 'top', 'paddingLeft': '20px'}, children=[
                html.H3("Projected 2026 Outcome"),
                html.Div(style={'display': 'flex'}, children=[
                    dcc.Graph(id='pred-seats', style={'width': '50%'}),
                    dcc.Graph(id='pred-votes', style={'width': '50%'})
                ]),
                
                html.H3("Historical Data Analysis", style={'marginTop': '30px', 'borderTop': '1px dashed #000', 'paddingTop': '20px'}),
                html.Div(style={'display': 'flex'}, children=[
                    dcc.Graph(id='hist-seats', style={'width': '50%'}),
                    dcc.Graph(id='hist-votes', style={'width': '50%'})
                ])
            ])
        ])
    ]),
    
    html.Div(style={'borderTop': '1px solid #000000', 'marginTop': '30px', 'paddingTop': '10px'}, children=[
        html.P("CONFIDENTIAL - Data is based on historical trends.", style={'fontSize': '12px'})
    ])
])

# 4. Callbacks for interactivity
@app.callback(
    [Output('pred-seats', 'figure'),
     Output('pred-votes', 'figure'),
     Output('hist-seats', 'figure'),
     Output('hist-votes', 'figure'),
     Output('winner-name', 'children'),
     Output('winner-status', 'children'),
     Output('winner-name', 'style')],
    [Input(f'swing-{p}', 'value') for p in parties]
)
def update_dashboard(*swings):
    swing_factors = {p: v for p, v in zip(parties, swings) if v != 0}
    
    # ML Logic
    predictions = []
    for party in df['Party'].unique():
        party_data = df[df['Party'] == party]
        X = party_data[['Year']].values
        y_seats = party_data['Seats'].values
        y_votes = party_data['Vote_Share'].values
        
        # Simple Linear Regression for seats
        model_seats = LinearRegression()
        model_seats.fit(X, y_seats)
        
        # Simple Linear Regression for vote share
        model_votes = LinearRegression()
        model_votes.fit(X, y_votes)
        
        pred_seats = model_seats.predict([[2026]])[0]
        pred_votes = model_votes.predict([[2026]])[0]
        
        # Apply swing if provided
        if party in swing_factors:
            swing = swing_factors[party]
            pred_votes += swing
            pred_seats += (swing * 3)
            
        pred_seats = max(0, int(round(pred_seats)))
        pred_votes = max(0.0, round(pred_votes, 2))
        
        predictions.append({
            'Party': party,
            'Seats': pred_seats,
            'Vote_Share': pred_votes
        })
        
    pred_df = pd.DataFrame(predictions)
    
    # Normalize seats to sum to 294
    total_seats = pred_df['Seats'].sum()
    if total_seats > 0:
        pred_df['Seats'] = (pred_df['Seats'] / total_seats * 294).round().astype(int)
    
    # Fix rounding errors
    diff = 294 - pred_df['Seats'].sum()
    if diff != 0:
        max_idx = pred_df['Seats'].idxmax()
        pred_df.loc[max_idx, 'Seats'] += diff
        
    # Normalize vote share
    total_votes = pred_df['Vote_Share'].sum()
    if total_votes > 0:
        pred_df['Vote_Share'] = (pred_df['Vote_Share'] / total_votes * 100).round(2)
        
    pred_df = pred_df.sort_values(by="Seats", ascending=False)
    
    # Create Figures with white background to match gov theme
    plotly_layout = {
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'font': {'family': 'Arial'},
        'margin': dict(t=40, b=40, l=40, r=20)
    }

    # Hist Seats
    fig_hist_seats = px.line(df, x="Year", y="Seats", color="Party", color_discrete_map=party_colors, markers=True, title="Historical Seats")
    fig_hist_seats.update_xaxes(type='category', gridcolor='#e0e0e0')
    fig_hist_seats.update_yaxes(gridcolor='#e0e0e0')
    fig_hist_seats.update_layout(**plotly_layout)
    
    # Hist Votes
    fig_hist_votes = px.line(df, x="Year", y="Vote_Share", color="Party", color_discrete_map=party_colors, markers=True, title="Historical Vote Share (%)")
    fig_hist_votes.update_xaxes(type='category', gridcolor='#e0e0e0')
    fig_hist_votes.update_yaxes(gridcolor='#e0e0e0')
    fig_hist_votes.update_layout(**plotly_layout)
    
    # Pred Seats
    pred_df_pie = pred_df[pred_df['Seats'] > 0]
    fig_pred_seats = px.pie(pred_df_pie, values="Seats", names="Party", color="Party", color_discrete_map=party_colors, title="Predicted Seats (Total: 294)")
    fig_pred_seats.update_traces(textinfo='value+percent', textposition='inside')
    fig_pred_seats.update_layout(**plotly_layout)
    
    # Pred Votes
    fig_pred_votes = px.bar(pred_df, x="Party", y="Vote_Share", color="Party", color_discrete_map=party_colors, text="Vote_Share", title="Predicted Vote Share (%)")
    fig_pred_votes.update_traces(textposition='outside')
    fig_pred_votes.update_xaxes(gridcolor='#e0e0e0')
    fig_pred_votes.update_yaxes(gridcolor='#e0e0e0')
    fig_pred_votes.update_layout(**plotly_layout)
    
    # Winner Box Logic
    winner = pred_df.iloc[0]['Party']
    max_seats = pred_df.iloc[0]['Seats']
    winner_color = party_colors.get(winner, 'black')
    
    if max_seats >= 148:
        status = f"Majority Win ({max_seats} seats)"
    else:
        status = f"Hung Assembly (Largest party with {max_seats} seats)"
        
    return fig_pred_seats, fig_pred_votes, fig_hist_seats, fig_hist_votes, winner, status, {'color': winner_color, 'fontWeight': 'bold', 'fontSize': '18px'}

if __name__ == '__main__':
    print("Starting Dash app... open http://127.0.0.1:8000 in your browser.")
    app.run(debug=True, dev_tools_ui=False, port=8000)
