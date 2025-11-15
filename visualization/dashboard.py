"""
Interactive dashboard helpers using Plotly.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_dashboard(
    price_df: pd.DataFrame,
    predictions: Dict[str, pd.Series],
    metrics_table: pd.DataFrame,
) -> go.Figure:
    """
    Compose a multi-panel interactive dashboard summarising predictions.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Historical Close",
            "Predictions",
            "Return Distribution",
            "Model Metrics",
        ),
        specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df["Close"],
            mode="lines",
            name="Close",
        ),
        row=1,
        col=1,
    )

    for model_name, series in predictions.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=model_name,
            ),
            row=1,
            col=1,
        )

    returns = price_df["Close"].pct_change().dropna()
    fig.add_trace(
        go.Histogram(x=returns, name="Daily Returns", opacity=0.75, nbinsx=50),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=list(metrics_table.columns), fill_color="paleturquoise"),
            cells=dict(values=[metrics_table[col] for col in metrics_table.columns]),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, width=1100, title_text="Stock Prediction Dashboard")
    return fig




