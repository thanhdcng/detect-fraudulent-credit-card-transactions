import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load the saved all_metrics DataFrame
all_results = pd.read_csv('all_results.csv')
# Load the saved best_results DataFrame
best_results = pd.read_csv('best_results.csv')

# Round the specific columns to 2 decimal places
best_results['Precision'] = best_results['Precision'].round(2)
best_results['Recall'] = best_results['Recall'].round(2)
best_results['F1 Score'] = best_results['F1 Score'].round(2)
best_results['ROC AUC'] = best_results['ROC AUC'].round(2)
best_results['Jaccard Score'] = best_results['Jaccard Score'].round(2)
all_results['Precision'] = all_results['Precision'].round(2)
all_results['Recall'] = all_results['Recall'].round(2)
all_results['F1 Score'] = all_results['F1 Score'].round(2)
all_results['ROC AUC'] = all_results['ROC AUC'].round(2)
all_results['Jaccard Score'] = all_results['Jaccard Score'].round(2)

# Create a list of available metrics
metrics = ['Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Jaccard Score', 'Net Savings ($)']

# Create a list of unique model names for selection
model_names = best_results['Model Name'].unique()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Model Evaluation Dashboard on Unseen Test Data"),
    
    # Dropdown for selecting a metric to display
    dcc.Dropdown(
        id='metric-selector',
        options=[{'label': metric, 'value': metric} for metric in metrics],
        value='Precision',  # Default value
        clearable=False
    ),
    
    # Bar chart to show the selected metric vs. Threshold
    dcc.Graph(id='metric-graph'),
    
    # Div to hold the two pie charts side by side
    html.Div([
        dcc.Graph(id='confusion-matrix-pie', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='cost-pie-chart', style={'display': 'inline-block', 'width': '49%'})
    ]),
    
    # Table to display the detailed metrics for the top models
    dash_table.DataTable(
        id='model-details-table',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'minWidth': '100px',
            'width': '100px',
            'maxWidth': '100px',
            'whiteSpace': 'normal'
        }
    ),
    
    # Dropdown for selecting a model to display its metrics
    html.H2("Select a Model to View Detailed Metrics"),
    dcc.Dropdown(
        id='model-selector',
        options=[{'label': name, 'value': name} for name in model_names],
        value=model_names[0],  # Default value is the first model
        clearable=False
    ),
    
    # Table to display the selected model's metrics
    dash_table.DataTable(
        id='selected-model-details-table',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'minWidth': '100px',
            'width': '100px',
            'maxWidth': '100px',
            'whiteSpace': 'normal'
        },
        page_size=10  # Allows pagination
    )
])

# Callback to update the graph and table based on selected metric
@app.callback(
    Output('metric-graph', 'figure'),
    Output('confusion-matrix-pie', 'figure'),
    Output('cost-pie-chart', 'figure'),
    Output('model-details-table', 'data'),
    Output('model-details-table', 'columns'),
    Output('selected-model-details-table', 'data'),
    Output('selected-model-details-table', 'columns'),
    Input('metric-selector', 'value'),
    Input('model-selector', 'value')
)
def update_graph_and_table(selected_metric, selected_model):
    if not selected_metric or not selected_model:
        raise dash.exceptions.PreventUpdate

    # Filter the DataFrame to get the top 3 models based on the selected metric
    top_models = best_results.nlargest(3, selected_metric)
    
    # Create the bar plot for the selected metric
    fig_metric = px.bar(top_models, x='Threshold', y=selected_metric, color='Model Name', 
                        title=f"Top 3 Models by {selected_metric}", barmode='group')
    
    # Select the best model to display confusion matrix and cost pie
    best_model = top_models.iloc[0]
    
    # Create a pie chart for the confusion matrix without True Negatives
    labels = ['True Positives (TP)', 'False Positives (FP)', 'False Negatives (FN)']
    values = [best_model['True Positives (TP)'], best_model['False Positives (FP)'], best_model['False Negatives (FN)']]
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig_pie.update_layout(
        title=f"Confusion Matrix for {best_model['Model Name']} at Threshold {best_model['Threshold']:.2f}",
        title_font_size=14
    )
    
    # Create a pie chart for cost of operation and net savings
    cost_labels = ['Cost of Operation ($)', 'Net Savings ($)']
    cost_values = [best_model['Cost of Operation ($)'], best_model['Net Savings ($)']]
    
    fig_cost_pie = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_values, hole=.3)])
    fig_cost_pie.update_layout(
        title=f"Total Cost for {best_model['Model Name']} at Threshold {best_model['Threshold']:.2f}",
        title_font_size=14
    )
    
    # Prepare data and columns for the detailed metrics table
    model_details_data = top_models.to_dict('records')
    model_details_columns = [{"name": i, "id": i} for i in top_models.columns]
    
    # Filter the DataFrame to get the metrics for all records of the selected model
    selected_model_metrics = all_results[all_results['Model Name'] == selected_model]
    selected_model_data = selected_model_metrics.to_dict('records')
    selected_model_columns = [{"name": i, "id": i} for i in selected_model_metrics.columns]
    
    return (fig_metric, fig_pie, fig_cost_pie, 
            model_details_data, model_details_columns, 
            selected_model_data, selected_model_columns)

# Run the app on a different port to avoid conflicts
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
