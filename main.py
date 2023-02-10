import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)

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

#***************************lgbm********************************************
with st.form('lgbmによる推測'):

    #lgbmで予測するシリーズの選択
    target_lgbm = st.selectbox(
            'target series:',
            sim_candidates2.index,   
        ) 
    submitted = st.form_submit_button("Submit")
    if submitted:    

        #相関係数でカラムを絞る
        df_corr_all = df_zenkoku.corr()
        df_corr_all2 = df_corr_all.loc[target_lgbm]
        df_corr_all2 = df_corr_all2[(df_corr_all2 >=0.4) | (df_corr_all2 <= -0.4)]

        #単回帰分析で外れ値（得意先）を削除
        #target_lgbmを抜いたカラムリスト作成
        col_list = list(df_corr_all2.index)
        col_list.remove(target_lgbm)

        #単回帰モデル作成
        # model_lr = LinearRegression()

        # for col in col_list:
        #     model_lr.fit(df_zenkoku[[col]], df_zenkoku[[target_lgbm]])

        #     df_target = df_zenkoku[[target_lgbm]]
        #     df_target['lr_value'] = model_lr.predict(df_zenkoku[[col]])
        #     df_target['lr_value'] = df_target['lr_value'].map(lambda x: round(x))
        #     df_target['value/lr'] = df_target['lr_value'] / df_target[target_lgbm]

        #     df_target = df_target[(df_target['value/lr'] <= 4) & (df_target['value/lr'] >= 0.25)]

        # #外れ値削除後（得意先を絞った）のdf_zenkoku
        # df_zenkoku5 = df_zenkoku.loc[df_target.index]

        df_zenkoku5 = df_zenkoku.copy()


        #target(対象得意先)のデータを外す
        if target in df_zenkoku5.index:
            df_zenkoku6 = df_zenkoku5.drop(target, axis=0)
        else:
            df_zenkoku6 = df_zenkoku5.copy()

        #target_seriesが10万円以上のデータに絞る
        df_zenkoku6 = df_zenkoku6[df_zenkoku6[target_lgbm] >= 100000]                   

        #相関の高いカラムに絞ってモデルに渡す
        corr_col_list = df_corr_all2.index

        df_zenkoku6 = df_zenkoku6[corr_col_list]

        #過去のデータを分析するモデル作成　lgbm

        #データの分割
        X = df_zenkoku6.drop(target_lgbm, axis=1)
        y = df_zenkoku6[target_lgbm]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        #lgbmの実装
        #ハイパーパラメータ調整
        estimator = lgb.LGBMRegressor()

        #ハイパーパラメータの範囲
        param_grid = [{
            'num_leaves': [50, 60, 70, 80, 90, 100, 110, 120, 130], #[50, 60, 70, 80, 90, 100, 110, 120, 130]
            'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130],#def100
            'boosting_type':['gbdt', 'dart', 'goss', 'rf'], #def 'gbdt'['gbdt', 'dart', 'goss', 'rf']
            'min_data_in_leaf': [10, 20, 30, 40, 50], #[10, 20, 30, 40, 50]
            'class_weight': ['balanced'], #['balanced', 'None']
            'max_depth': [2, 3, 4, 5, 6], #[2, 3, 4, 5, 6]
            # 'learning_rate': [0.1],
            'random_state': [100]
        }]

        #データセットの分割数
        cv = 5

        tuned_model = GridSearchCV(estimator=estimator,
                                param_grid= param_grid,
                                cv= cv,
                                return_train_score=False) #return_train_score　trainデータの検証を返すか

        tuned_model.fit(X_train, y_train)

        y_pred_train = tuned_model.predict(X_train)
        y_pred_test = tuned_model.predict(X_test)

        df_tuned_model = pd.DataFrame(tuned_model.cv_results_)

        r2_train = r2_score(y_train,y_pred_train)
        r2_test = r2_score(y_test,y_pred_test)
        st.write('R2_train :',r2_train)
        st.write('R2_test :',r2_test)

        #パラメータチューニングの結果
        df_tuned_model.sort_values('mean_test_score', ascending=False)

        #ベストモデルの作成
        best_model = tuned_model.best_estimator_
        st.caption('ベストモデルのスコア')
        st.write(best_model.score(X_test, y_test))

        y_pred_best = best_model.predict(X_test)

        # 真値と予測値の表示
        df_pred = pd.DataFrame({'real': y_test, 'pred': y_pred_best})
        df_pred['pred'] = df_pred['pred'].map(lambda x: round(x))
        df_pred['rate'] = df_pred['pred'] / df_pred['real']

        st.caption('予測状況')
        st.dataframe(df_pred)

        # モデル評価
        # rmse : 平均二乗誤差の平方根

        mse = mean_squared_error(y_test, y_pred_test) # MSE(平均二乗誤差)の算出
        rmse = np.sqrt(mse) # RSME = √MSEの算出
        st.write('RMSE :',rmse)

        # r2 : 決定係数
        r2 = r2_score(y_test,y_pred_test)
        st.write('R2 :',r2)

        importances = pd.DataFrame({'features': X_train.columns, \
                            'importances': best_model.feature_importances_})\
                            .sort_values('importances', ascending=False)
        st.caption('feature_importances')               
        st.dataframe(importances)

        #*****************target得意先の推測*******************************
        df_target = pd.DataFrame(df_zenkoku.loc[target])
        df_target = df_target.T

        #相関の強いカラムへの絞込み
        df_target2 = df_target[corr_col_list]

        X_target = df_target2.drop(target_lgbm, axis=1)
        y_target = df_target2[target_lgbm]

        #ベストモデルで推測
        best_model2 = tuned_model.best_estimator_
        st.caption('ベストモデルのスコア')
        st.write(best_model2.score(X_target, y_target))

        y_pred_best = best_model2.predict(X_target)

        # 真値と予測値の表示
        df_pred = pd.DataFrame({'real': y_target, 'pred': y_pred_best})
        df_pred['pred'] = df_pred['pred'].map(lambda x: round(x))
        df_pred['rate'] = df_pred['pred'] / df_pred['real']
        st.caption('推測結果')
        st.table(df_pred)








