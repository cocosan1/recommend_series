import numpy as np
import pandas as pd
import streamlit as st

#st
st.set_page_config(page_title='recommend_series')
st.markdown('#### レコメンド アプリ')

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
cust_text = st.text_input('得意先名の一部を入力 例）東京イ')

target_list = []
for cust_name in df_kita.index:
    if cust_text in cust_name:
        target_list.append(cust_name)

if target_list != '':
    # selectbox target ***
    target = st.selectbox(
        'target得意先:',
        target_list,   
    ) 

if target != '':

    #dfをアイテム列だけに絞る
    df_zenkoku3 = df_zenkoku2.drop(['sales', 'a_price'], axis=1)

    #相関係数 相関なし0-0.2 やや相関あり0.2-0.4 相関あり 0.4-0.7 強い相関 0.7-
    df_corr = df_zenkoku3.corr()

    #targetの売上を整理
    df_zenkoku_temp = df_zenkoku.drop(['sales', 'a_price'], axis=1)
    df_target_sales = df_zenkoku_temp.loc[target]

    #recomenndリスト作成のための計算
    #相関係数×シリーズ売上の一覧
    sim_candidates = pd.Series()
    for i in range(0, len(df_target_sales.index)):
        # print('adding sims for ' + target_sales.index[i] + '...')
        #シリーズ毎に相関表作成
        sims = df_corr[df_target_sales.index[i]]
        #相関係数×シリーズ金額
        sims = sims.map(lambda x: round(x * df_target_sales[i]))
        sim_candidates = sim_candidates.append(sims)

    #シリーズ毎に合計
    sim_candidates = sim_candidates.groupby(sim_candidates.index).sum()
    sim_candidates.sort_values(ascending=False, inplace=True)

    #pointsとsalesをmerge
    df_simcan = pd.DataFrame(sim_candidates)
    df_sales = pd.DataFrame(df_target_sales)
    df_merge = df_simcan.merge(df_sales, left_index=True, right_index=True, how='right')
    df_merge = df_merge.sort_values(0, ascending=False)
    df_merge.columns = ['points', 'sales']
    st.markdown('###### 全シリーズ')
    st.write(df_merge)

    #展示品の指定
    tenji_series = st.multiselect(
        '展示品を選択',
        df_target_sales.index)

    #展示している商品は削る
    sim_candidates2 = df_merge.drop(index=tenji_series)
    st.markdown('###### お薦め展示シリーズ')
    st.write(sim_candidates2)



