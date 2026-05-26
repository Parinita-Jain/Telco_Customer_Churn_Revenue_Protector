import pandas as pd

def business_kpi(df, predictions):

    df["pred"] = predictions

    # Assume:
    churn_cost = 5000      # loss per churn
    retention_cost = 500   # cost to retain

    actual_loss = df[df["Churn"] == 1].shape[0] * churn_cost

    saved_customers = df[(df["Churn"] == 1) & (df["pred"] == 1)].shape[0]

    saved_revenue = saved_customers * churn_cost
    retention_spend = saved_customers * retention_cost

    net_savings = saved_revenue - retention_spend

    return {
        "actual_loss": actual_loss,
        "saved_revenue": saved_revenue,
        "retention_spend": retention_spend,
        "net_savings": net_savings
    }