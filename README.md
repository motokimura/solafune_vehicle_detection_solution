# motokimura's solution for Solafune "Vehicle Detection in Multi-Resolution Images" competition

Solafuneの[「マルチ解像度画像の車両検出」コンテスト](https://solafune.com/ja/competitions/25012781-b1e8-499e-9c8c-1f9b284d483e?menu=about&tab=overview)
に対するmotokimuraの解法です。

> cf. @solafune (https://solafune.com)
> コンテストの参加以外の目的とした利用及び商用利用は禁止されています。商用利用・その他当コンテスト以外で利用したい場合はお問い合わせください。(https://solafune.com)

## 1. Final Approach

### サマリ
* 500x375解像度の画像は2倍に拡大し、すべての画像で同じくらいの地上分解能になるようにした
* YOLOX-Lと、2通りのハイパラで学習したYOLOX-Xを[WBF (Weighted Boxes Fusion)](https://arxiv.org/abs/1910.13302)によってアンサンブル
* YOLOX-Xを学習する際には、YOLOX-Lによるpseudo labelを学習データに混ぜ、強めのaugmentationをかけた（[Noisy Student](https://arxiv.org/abs/1911.04252)の要領）
* 大型トラックなどの誤検出を防ぐ狙いで、学習の後半ではrandom resizeの範囲を小さくする（OC-Costが0.001~0.002ほど改善）
* WBF・NMSのIoU閾値をやや小さめ（0.4）に設定することで、OC-Costが0.003ほど改善
* WBF後のbboxの座標を、Pythonのround関数で一番近い整数に丸めるとOC-Costが大きく改善した（floatに比べて0.005ほど改善）

### 前処理
* 学習データをrandom splitで5 foldに分割
* 500x375解像度の画像を縦横2倍に拡大（Lanczos）し、600x500解像度の画像と同程度の地上分解能になるようにした

### データ拡張
* random crop（480x480）
* horizontal and vertical flip
* rotate90/180/270
* 上記に加え、モデルごとに追加のaugmentationを適用（以下で説明）

### モデル
* モデルA
    * YOLOX-L
    * 100 epoch 学習
    * 50 epochまで0.667〜1.333の倍率でrandom resize、以降の50 epochでは0.887〜1.113にresizeの範囲を狭める
* モデルB
    * YOLOX-X
    * 130 epoch 学習
    * モデルAによるpseudo label（テストデータへの推論結果）をオリジナルの学習データに混ぜて学習
    * 0.667〜1.333の倍率でrandom resize
    * Noisy Studentの要領で、モデルAよりも強めのaugmentationをかけた（最後の50 epochではオフ）
        * Mosaic
        * Mixup
        * RandomAffine（scale:0.9〜1.1, rotate: ±10 deg, shear: ±2 deg）
* モデルC
    * YOLOX-X（アーキテクチャはモデルBと同じ。augmentationをやや弱めにしたもの）
    * 100 epoch 学習
    * モデルAによるpseudo label（テストデータへの推論結果）をオリジナルの学習データに混ぜて学習
    * 50 epochまで0.667〜1.333の倍率でrandom resize、以降の50 epochでは0.887〜1.113にresizeの範囲を狭める
    * Noisy Studentの要領で、モデルAよりも強めのaugmentationをかけた（最後の50 epochではオフ）
        * Mosaic
        * Mixup
* モデルA〜Cに共通の学習設定
    * batch size: 4
    * optimizer: SGD(momentum=0.9, weight_decay=5e-4, nesterov=True)
    * lr: 6.25e-4で固定（annealing, stepも試したが改善せず）
    * COCO pretrainedのweightで初期化
    * fp16 (amp) で single gpu training（使用メモリは10GBくらい）
    * valに対するOC-Costが最小となるepochを選択
    * 実装には[mmdetection](https://github.com/open-mmlab/mmdetection)を使用

### 推論・後処理
* horizontal/vertical flip TTA + rotate90/180/270 TTA の5通りのTTA
* n_arch x n_fold x (1 + n_tta) = 3 x 5 x (1 + 5) = 90 の推論結果を、WBF (Weighted Boxes Fusion)でアンサンブル
* NMS・WBFのiou_thresh=0.4
* 検出閾値は0.5
* WBF後のbboxの座標を、Pythonのround関数で一番近い整数に丸める

### Ablation

Model | CV | Public LB | Private LB
-- | -- | -- | --
A (yolox-l) | 0.13870 | 0.13717 | 0.14110
B (yolox-x) | 0.13627 | 0.13760 | 0.14146
C (yolox-x, weaker augmentation) | 0.13555 | 0.13776 | 0.14120
A + B | 0.13532 | 0.13664 | 0.14031
A + B + C | 0.13469 | 0.13700 | 0.13996

* LBでは、単体モデルとしてはYOLOX-Lが最も性能が良く（表の1行目）、YOLOX-Xとアンサンブルしなくてもprivateで1st
* YOLOX-X（2・3行目）は、YOLOX-L（1行目）に比べてCVは良いが、LBは改善していない
* YOLOX-LとYOLOX-XをアンサンブルするとCV・LBともに改善したので、これらを最終サブとした（4・5行目）

### 補足
* YOLOX-Xを学習する際には、pseudo label + 強めのaugmentation が重要であった
    * pseudo labelを加えずにYOLOX-Xを学習した場合、YOLOX-Lに比べてCV・LBともに劣る結果となった
    * また、pseudo labelを加えてYOLOX-Xを学習した場合でも、強めのaugmentation（mosaic, mixup など）をかけないと、YOLOX-Lに比べてCV・LBともに劣る結果となった
    * pseudo label + 強めのaugmentationで学習したYOLOX-**L**よりも、同じ条件で学習したYOLOX-Xの方がCVは良かった（pseudo label + 強めのaugmentationは、大きいモデルほど有効）
* NMS・WBFのIoU閾値を調整するのが効いた
    * CVによりiou_thresh=0.4とした
    * デフォルト（NMS: 0.65, WBF: 0.55）に比べて、CV・LBともに0.003ほど改善
* 学習の後半でrandom resizeの範囲を狭めるのが地味に効いた（LBで0.001〜0.002ほど改善）
    * YOLO系のモデルは、特に学習序盤にはrandom resizeを強めにかけた方が最終的に精度が上がりやすい（あくまで自分の経験上）
    * 一方で、今回は大型トラックなどは検出対象に含まれないため、random resizeをやりすぎると誤検出につながると考えた
    * 学習前半ではrandom resizeを強めにかけ、後半ではresizeの範囲を狭めるというアプローチをとった
* TTAによる改善は小さい（LBでは0.001程度）
* アノテーションが整数で与えられているためか、予測したbboxの座標を一番近い整数に丸めるとCV・LBともに0.005改善した
    * コンペ終了2日前に気づき、public LBで3位→1位に上がった
* 分解能を近づける（500x375解像度の画像を縦横2倍に拡大する）効果は、ちゃんと定量評価できていない
    * アノテーションされたbboxのサイズ分布を見ると、2.3倍ほどの拡大率が最適に見えたが、拡大率2.3倍よりも2倍の方がCVは良かった
    * 対象物体が小さいので、画像を整数倍で拡大しないと物体の境界が曖昧になってしまうのが原因？

## 2. Solution Development

* 1週間ほどかけてmmdetectionのCOCO APが高いモデルを色々試し、OC-Costが最も良かったYOLOX-Lをベースラインモデルとして選定
    * [DetectoRS](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors)や[UniverseNet](https://github.com/shinya7y/UniverseNet/tree/master/configs/universenet)などを試したが、YOLOX-Lより精度・速度の両面で劣っていたので採用せず
    * mmdetectionを選んだ理由としては、多少使い慣れていたのと、色々なモデルがサポートされているのでアンサンブルが容易そうだと考えたため（結局YOLOXしか使わず）
* 2週間ほどかけて、YOLOX-Lのハイパラ（主にデータ拡張）を最適化：
    * random resizeの範囲
    * horizontal/vertical flip
    * rotate90/180/270
    * mixup（YOLOX-Lでは採用せず）
    * mosaic（YOLOX-Lでは採用せず）
    * color jitter（精度が上がらなかったので採用せず）
    * lr schedule（annealing, stepなど試したが、6.25e-4で固定がベストだった）
    * 画像の拡大（車両がかなり小さいので、入力画像のサイズを3倍以上に拡大するというのを試したが、train lossが収束せず断念）
* YOLOX-Lでの精度が頭打ちとなったところで、1週間ほどかけて推論まわりのハイパラを最適化：
    * WBF・NMSのIoU閾値の調整
    * TTA（horizontal/vertical flip + rotate90/180/270）
* その後、1~2種間ほどかけてpseudo labelingを検証
    * 最初は定石通りに、YOLOX-Lをpseudo labelを加えてfinetuneするなどしていたが、OC-Costは改善せず
    * Noisy Studentの論文を参考に、YOLOX-Xに強いaugmentaion（mixup, mosaic, randon affine）を加えて学習し、YOLOX-LとアンサンブルするとOC-Costが改善したのでこの方法を採用した
* 最後の数日間は、後処理の最適化に取り組んだ：
    * 検出したbboxの面積に対して閾値を設定し、範囲外のbboxは削除する（OC-Costは改善せず）
    * 色々試したがほとんど効果がなく、唯一、bboxの座標値を一番近い整数に丸めることでOC-Costを大きく改善できることに気づいた
