# MaAI-SDS 用の FST。公式のサンプル集（mmdagent-ex/example）の中に maai_sds フォルダーごと置いて使う。
# アバターはサンプル集のジェネ。待機モーションは体の揺れが小さい 01_Wait.vmd、カメラは上半身。
# 頷き・まばたきは MaAI-SDS が頭のボーンと表情を結び付けて動かし、口パクは MMDAgent-EX が音声から作る。
0 10 <eps> STAGE|../images/floor_block.png,../images/back_white.png
10 20 <eps> MODEL_ADD|gene|../gene/gene.pmd
20 30 MODEL_EVENT_ADD|gene MOTION_ADD|gene|base|../motions/wait/01_Wait.vmd|FULL|LOOP|OFF|OFF
30 40 <eps> CAMERA|0,16.0,0|0,0,0|27.0|30.0
