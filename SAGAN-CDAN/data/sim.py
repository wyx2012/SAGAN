import pandas as pd
import numpy as np
import os
from rdkit.Chem import AllChem, Draw
from rdkit import rdBase, Chem, DataStructs
ChChMiner_path = './ChChMiner.csv'
ZhangDDI_path = './ZhangDDI_re_sim.csv'
# 读取两个CSV文件
df1 = pd.read_csv(ChChMiner_path)
df2 = pd.read_csv(ZhangDDI_path)
column_name1 = 'he'
column_name2 = 'drugbank_id_2'

ch_column1 = df1[[column_name1]]
ch_column2 = df1[[column_name2]]
zhang_column1 = df2[[column_name1]]
zhang_column2 = df2[[column_name2]]
# 使用merge找出两个数据框中相同的行，基于'column_to_compare'列
# 这里我们只关心df2中存在的行，所以使用indicator=True和之后的查询
merged = df1[[column_name1]].merge(df2[[column_name1]], on=column_name1, how='right', indicator=True)

# 找出在两个数据框中都存在的行
to_delete = merged[merged['_merge'] == 'both'][column_name1]

# 基于上面的结果，从df2中删除这些行
df2_filtered = df2[~df2[column_name1].isin(to_delete)]

# 保存修改后的df2
df2_filtered.to_csv('ZhangDDI_re_sim.csv', index=False)