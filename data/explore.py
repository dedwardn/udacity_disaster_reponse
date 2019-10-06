import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategories', engine)

category_ratio =  df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()/df.shape[0]
category_names = category_ratio.index.values


print(category_ratio.sort_values(ascending=False), category_names)