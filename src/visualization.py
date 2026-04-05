import plotly.graph_objects as go

def plot_results(real, predicted, title="Forecast vs Actual"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=real, name='Actual'))
    fig.add_trace(go.Scatter(y=predicted, name='Predicted'))
    fig.update_layout(title=title)
    fig.show()