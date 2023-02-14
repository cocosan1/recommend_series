import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openpyxl

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)

#st
st.set_page_config(page_title='recommend_series')
st.markdown('### レコメンド アプリ')

#データ読み込み
df_zenkoku = pd.read_pickle('df_zenkoku7879.pickle')

st.markdown('###### １．分析対象得意先の絞込み')
#得意先範囲の設定
sales_max = st.number_input('分析対象得意先の売上上限を入力', key='sales_max', value=70000000)
sales_min = st.number_input('分析対象得意先の売上下限を入力', key='sales_min', value=2000000)

#salesによる絞込み
df_zenkoku2 = df_zenkoku.copy()
df_zenkoku2 = df_zenkoku2[(df_zenkoku2['sales'] >= sales_min) & (df_zenkoku2['sales'] <= sales_max)]
st.caption(f'対象得意先数: {len(df_zenkoku2)}')

img_yajirusi = Image.open('矢印.jpeg')
st.image(img_yajirusi, width=20)

#target選定用リスト
st.markdown('######  ２．target得意先の選択')
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
st.image(img_yajirusi, width=20)    

if target != '':

    #dfをアイテム列だけに絞る
    df_zenkoku3 = df_zenkoku2.drop(['sales', 'a_price'], axis=1)

    #相関係数 相関なし0-0.2 やや相関あり0.2-0.4 相関あり 0.4-0.7 強い相関 0.7-
    df_corr = df_zenkoku3.corr()

    #targetの売上を整理
    df_zenkoku_temp = df_zenkoku.drop(['sales', 'a_price'], axis=1)
    df_target_sales = df_zenkoku_temp.loc[target]

    #******************アイテムベース*******************************
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

    st.markdown('######  ３．展示品の選択')

    #展示品の指定
    tenji_series = st.multiselect(
        '展示品を選択',
        df_target_sales.index)

    

    st.image(img_yajirusi, width=20) 

    st.markdown('######  ４．動きのよくない展示品の抽出')

    # ***ファイルアップロード 今期***
    uploaded_file_now = st.file_uploader('今期', type='xlsx', key='now')

    if uploaded_file_now:
        df_now = pd.read_excel(
            uploaded_file_now, sheet_name='受注委託移動在庫生産照会', usecols=[3, 15, 42, 50])  # index　ナンバー不要　index_col=0
    else:
        st.info('今期のファイルを選択してください。')
        st.stop()

    #targetから79などを削除
    target2 = target[:-2]
    
    #LD分類
    df_now_target = df_now[df_now['得意先名']==target2]
    cate_list = []
    for cate in df_now_target['商品分類名2']:
        if cate in ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']:
            cate_list.append('d')
        elif cate in ['リビングチェア', 'クッション', 'リビングテーブル']:
            cate_list.append('l')
        else:
            cate_list.append('none') 

    df_now_target['category'] = cate_list 

    #noneを削除
    df_now_target = df_now_target[df_now_target['category']!='none']

    #シリーズ名＋LD分類
    df_now_target['series2'] = df_now_target['シリーズ名'] + '_' + df_now_target['category']

    #seiries2を品番に変換

    index_list = []
    htsd_list = []
    kxd_list = []
    sgd_list = []
    kdd_list = []
    snd_list = []
    vzd_list = []
    sld_list = []
    fxd_list = []
    rkd_list = []
    psd_list = []

    snl_list = []
    hkl_list = []
    wkl_list = []
    kdl_list = []
    wql_list = []
    wnl_list = []
    fxl_list = []
    psl_list = []
    sdl_list = []

    sales_list = []

    for cust in df_now_target['得意先名'].unique():
        index_list.append(cust)
        df = df_now_target[df_now_target['得意先名']==cust]
        sales = df['金額'].sum()
        sales_list.append(sales)

        for series in  ['侭 JIN_d', 'SEOTO-EX_d', 'クレセント_d', 'SEOTO_d', '森のことば_d', 'TUGUMI_d', 'YURURI_d',\
            '風のうた_d', 'ALMO (ｱﾙﾓ)_d', 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d',\
            '森のことば_l', '穂高_l', 'CHIGUSA(ﾁｸﾞｻ）_l', 'SEOTO_l', 'SEION 静穏_l', 'VIOLA (ｳﾞｨｵﾗ)_l',\
            '風のうた_l', 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l', 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l']:

                    if len(df_now_target[df_now_target['series2']==series]) == 0:
                        if series == '侭 JIN_d':
                            htsd_list.append(0)
                        elif series == 'SEOTO-EX_d':  
                            kxd_list.append(0)
                        elif series == 'クレセント_d':
                            sgd_list.append(0)
                        elif series == 'SEOTO_d':
                            kdd_list.append(0)
                        elif series == '森のことば_d':
                            snd_list.append(0)
                        elif series == 'TUGUMI_d':
                            vzd_list.append(0) 
                        elif series == 'YURURI_d':
                            sld_list.append(0)
                        elif series == '風のうた_d':
                            fxd_list.append(0)
                        elif series == 'ALMO (ｱﾙﾓ)_d':
                            rkd_list.append(0)
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d':
                            psd_list.append(0)        

                        elif series == '森のことば_l':
                            snl_list.append(0)
                        elif series == '穂高_l':
                            hkl_list.append(0)
                        elif series == 'CHIGUSA(ﾁｸﾞｻ）_l':
                            wkl_list.append(0)       
                        elif series == 'SEOTO_l':
                            kdl_list.append(0)
                        elif series == 'SEION 静穏_l':
                            wql_list.append(0) 
                        elif series == 'VIOLA (ｳﾞｨｵﾗ)_l':
                            wnl_list.append(0) 
                        elif series == '風のうた_l':
                            fxl_list.append(0) 
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l':
                            psl_list.append(0)
                        elif series == 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l':
                            sdl_list.append(0)          
                            
                    else:
                        sales = df_now_target[df_now_target['series2']==series]['金額'].sum()
                        if series == '侭 JIN_d':
                            htsd_list.append(sales)
                        elif series == 'SEOTO-EX_d':  
                            kxd_list.append(sales)
                        elif series == 'クレセント_d':
                            sgd_list.append(sales)
                        elif series == 'SEOTO_d':
                            kdd_list.append(sales)
                        elif series == '森のことば_d':
                            snd_list.append(sales)
                        elif series == 'TUGUMI_d':
                            vzd_list.append(sales) 
                        elif series == 'YURURI_d':
                            sld_list.append(sales)
                        elif series == '風のうた_d':
                            fxd_list.append(sales)
                        elif series == 'ALMO (ｱﾙﾓ)_d':
                            rkd_list.append(sales)
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d':
                            psd_list.append(sales)        
                        elif series == '森のことば_l':
                            snl_list.append(sales)
                        elif series == '穂高_l':
                            hkl_list.append(sales)
                        elif series == 'CHIGUSA(ﾁｸﾞｻ）_l':
                            wkl_list.append(sales)       
                        elif series == 'SEOTO_l':
                            kdl_list.append(sales)
                        elif series == 'SEION 静穏_l':
                            wql_list.append(sales) 
                        elif series == 'VIOLA (ｳﾞｨｵﾗ)_l':
                            wnl_list.append(sales) 
                        elif series == '風のうた_l':
                            fxl_list.append(sales) 
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l':
                            psl_list.append(sales)
                        elif series == 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l':
                            sdl_list.append(sales)

    df_now_target2 =pd.DataFrame(list(zip(sales_list, htsd_list, kxd_list, sgd_list, kdd_list, snd_list,\
                                  vzd_list, sld_list, fxd_list, rkd_list, psd_list,\
                                  snl_list, hkl_list, wkl_list, kdl_list, wql_list, wnl_list, fxl_list,\
                                  psl_list, sdl_list)), \
                         index=['売上'],\
                          columns=['sales', 'hts_d', 'kx_d', 'sg_d', 'kd_d', 'sn_d', 'vz_d', 'sl_d',\
                                   'fx_d', 'rk_d', 'ps_d',\
                                   'sn_l', 'hk_l', 'wk_l', 'kd_l', 'wq_l', 'wn_l', 'fx_l', 'ps_l', 'sd_l']).T

    #展示品に絞る
    df_now_tenji = df_now_target2.loc[tenji_series]

    #展示品の売り上げ下限を入力
    min_line = st.number_input('展示品の売上下限を入力', key='min_line', value=0)

    #売上下限以下のdfを作成
    df_problem_series = df_now_tenji[df_now_tenji['売上'] <= min_line]

    df_problem_series = df_problem_series.sort_values('売上')

    st.write('動きの良くない展示シリーズ')
    st.table(df_problem_series)

    #******************ユーザーベース*******************************
    #データの正規化
    df_zenkoku3_norm = df_zenkoku3.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)),axis=1)
    #axis=1 行方向
    df_zenkoku3_norm = df_zenkoku3_norm.fillna(0)

    # コサイン類似度を計算

    #成分のほとんどが0である疎行列はリストを圧縮して表す方式
    from scipy.sparse import csr_matrix 

    #2つのベクトルがなす角のコサイン値のこと。1なら「似ている」を、-1なら「似ていない」
    from sklearn.metrics.pairwise import cosine_similarity

    zenkoku3_sparse = csr_matrix(df_zenkoku3_norm.values)
    user_sim = cosine_similarity(zenkoku3_sparse)
    user_sim_df = pd.DataFrame(user_sim,index=df_zenkoku3_norm.index,columns=df_zenkoku3_norm.index)

    # ユーザーtarget_nameと類似度の高いユーザー上位3店を抽出する
    sim_users = user_sim_df.loc[target].sort_values(ascending=False)[0:5]
    

    # ユーザーのインデックスをリストに変換
    sim_users_list = sim_users.index.tolist()
    #listからtarget_nameを削除

    #targetと同じ得意先のデータを削除
    sim_users_list2 = []
    for name in sim_users_list:
        if name[:3] not in target:
            sim_users_list2.append(name)


    # 類似度の高い上位得意先のスコア情報を集めてデータフレームに格納
    sim_df = pd.DataFrame()
    count = 0
    
    for i in df_zenkoku3_norm.iloc[:,0].index: #得意先名
        if i in sim_users_list2:
            #iloc[数字]でindexのみ指定可
            sim_df = pd.concat([sim_df,pd.DataFrame(df_zenkoku3_norm.iloc[count]).T])
        count += 1

    # ユーザーtarget_nameの販売シリーズを取得する
    df_target_sales = df_zenkoku3[df_zenkoku3.index==target].T

    # 未販売リストを作る
    nontenji_list = list(set(df_zenkoku3_norm.columns) - set(tenji_series))
    

    sim_df = sim_df[nontenji_list]

    #各シリーズの評点の平均をとる
    score = []
    for i in range(len(sim_df.columns)):
        #series毎に平均を出す
        mean_score = sim_df.iloc[:,i].mean()
        #seriesのname取り出し　columnで絞った場合はcolumns名
        name = sim_df.iloc[:,i].name
        #scoreにlist形式でシリーズ名とスコアを格納
        score.append([name, mean_score])
    # 集計結果からスコアの高い順にソートする
    #１つのlistに２カラム分入っている為list(zip())不要　
    #カラム名指定していない為0 1
    df_score_data = pd.DataFrame(score, columns=['cust', 'points']).sort_values('points', ascending=False)
    df_score_data = df_score_data.set_index('cust')

    st.markdown('######  ４．展示候補シリーズ')

    col1, col2 = st.columns(2)

    with col1:
        #展示している商品は削る
        sim_candidates2 = df_merge.drop(index=tenji_series)
        sim_candidates2 = sim_candidates2['points']
        st.markdown('###### アイテムベース分析')
        st.table(sim_candidates2[:5])

        st.caption('※参考　展示品のpoints')
        st.table(df_merge[:5])

    with col2:
        st.markdown('###### ユーザーベース分析')
        st.table(df_score_data[:5])
        st.caption('※一番売上が高いシリーズを1とした時の予測数値')

        df_temp1 = df_zenkoku3.loc[target].sort_values(ascending=False)[:1]
        st.write(df_temp1)

        st.write('類似得意先')
        st.write(sim_users.loc[sim_users_list2])

#***************************lgbm********************************************
st.markdown('## 展示後の売上の推測（機械学習）')
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








