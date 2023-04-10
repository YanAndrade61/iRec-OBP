import pandas as pd
from collections import Counter
import json
import numpy as np

ll = np.array([np.arange(20) for i in 20])


x = pd.read_csv("ml.csv", sep="|").head(50)

x["freq"] = [v if v <= 50 else 50 for v in x["freq"]]
print(f"sum = {x['freq'].sum()}, max={x['freq'].max()}")

x.to_csv("test.csv", sep="|", index=False)
