# 乗車車両の推定
k-NNとSVMを用いた乗車車両の推定

## 各スクリプトの要約
- ML_predict_evaluation.ipynb
-- 周船寺駅で録音した電車走行音声に対して，
SNMFとSEDNNを使って磁励音を分離し，特徴量を算出して機械学習で乗車車両を推定するプログラムです．
逐一，分離後のデータをnpyで保存し，それを読み出しながら特徴量と機械学習についてチューニングを行なっています．
一点注意事項として，データが少ないことが原因でハイパーパラメータのチューニングで少しdata leakageしています．
（そのためおおまかなハイパーパラメータの探索に留めています．）
また，CNNやDNNベースの推定を行う予定でしたが，これ以上の性能は出ませんでした．
(optunaでアーキテクチャまで探索をかけて最高で60%くらいでした)
おそらく，データの拡張やアーキテクチャの変更でもう少し上がると思いますが，なによりもデータ数が致命的でした．
- そのほかのpythonファイル
SEDNN用の関数を定義してあるファイルです．

## 注意点
SNMFに関しては基底行列の初期値データと処理後の学習音声データ，SEDNNに関しては学習済みモデルが必要です．

## 感想
- 特徴量算出後のデータについては，多くても7~12くらいの特徴量で済んでいるので，機械学習手法の違いはあまり性能に差を及ばさないと感じます．
- 光来出君の論文でもあったように，MFCCを取る際に，各時間フレームで平均してあげて12次元に落としてあげるっていうのは結構有効に働く様子です(理論的な理由はわからない)