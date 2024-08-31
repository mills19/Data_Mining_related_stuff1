import pandas as pd
import numpy as np
#creating synthetic dataset with outliers
data={'value':[10, 15, 8, 22, 7, 30, 5, 18, 25, 12, 100, 105, 98, 110]}
pd.DataFrame(data)

d=pd.DataFrame(data)
bins=5
bound=np.linspace(d['value'].min(),d['value'].max(),bins +1)

d['Smoothed_Value'] = d['value'].apply(lambda x: min(bound, key=lambda boundary: abs(boundary - x)))
smoothed_data = smooth_by_bin_boundaries(d, bins)
