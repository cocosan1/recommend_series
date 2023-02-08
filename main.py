import numpy as np
import pandas as pd
import streamlit as st

#st
st.set_page_config(page_title='recommend_series')
st.markdown('#### レコメンド　アプリ')

#データ読み込み
df_zenkoku = pd.read_pickle('df_zenkoku7879.pickle')

#得意先範囲の設定
sales_max = st.number_input('分析対象得意先の売上上限を入力', key='sales_max', value=70000000)
sales_min = st.number_input('分析対象得意先の売上下限を入力', key='sales_min', value=2000000)

#salesによる絞込み
df_zenkoku2 = df_zenkoku.copy()
df_zenkoku2 = df_zenkoku2[(df_zenkoku2['sales'] >= sales_min) & (df_zenkoku2['sales'] <= sales_max)]

#target選定用リスト
st.write('target得意先の選択')
df_kita = pd.read_pickle('df_kita7879.pickle')
target_sales_min = st.number_input('得意先絞込みの為、売上下限を入力', key='target_sales_min', value=2000000)
df_target = df_kita[df_kita['sales'] >= target_sales_min].index

# selectbox target ***
target_list = list(df_target)
target = st.selectbox(
    'target得意先:',
    target_list,   
) 

#dfをアイテム列だけに絞る
df_zenkoku3 = df_zenkoku2.drop(['sales', 'a_price'], axis=1)

#相関係数 相関なし0-0.2 やや相関あり0.2-0.4 相関あり 0.4-0.7 強い相関 0.7-
df_corr = df_zenkoku3.corr()
st.dataframe(df_corr)

#targetの売上を整理
df_zenkoku_temp = df_zenkoku.drop(['sales', 'a_price'], axis=1)
target_sales = df_zenkoku_temp.loc[target]
st.write(target_sales)

#展示品の選択


