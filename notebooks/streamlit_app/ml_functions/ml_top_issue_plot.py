import pandas as pd
from io import StringIO
import plotly.express as px

# csv_text = """content_clean,score,issue_cluster,issue_label
# "the lack of support for octopus intelligent go is frustrating i cant understand why this has been removed its the most popular ev tariff in the uk this change has cost me money and time trying to remedy it and for what what was the upside to bmw from making this change",1,issue_1,charging_issue
# "its not working",1,issue_4,app_stability_issue
# "not able to login at all says incorrect password when i am able login via browser and ios app",1,issue_0,login_issue
# "remote not working after reset",1,issue_4,app_stability_issue
# """

# df = pd.read_csv(StringIO(csv_text))

df = pd.read_csv("data/My_BMW_en_raw_clean_negative_issues.csv", encoding="utf-8-sig")

df["category"] = df["issue_label"].fillna("").astype(str).str.strip()
df.loc[df["category"] == "", "category"] = df.loc[df["category"] == "", "issue_cluster"]

plot_df = (
    df.groupby("category", as_index=False)
      .size()
      .rename(columns={"size": "count"})
      .sort_values("count", ascending=False)
)

fig = px.bar(
    plot_df,
    x="category",
    y="count",
    title="Reviews by Issue Category",
    labels={"category": "Issue Category", "count": "Count"},
    text="count"
)

fig.update_traces(textposition="outside")
fig.update_layout(xaxis_title="Issue Category", yaxis_title="Count")

fig.show()