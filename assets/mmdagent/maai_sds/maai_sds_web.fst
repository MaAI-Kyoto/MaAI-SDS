# MaAI-SDS 用の FST（ブラウザ版 MMDAgent-EX Web 用）。中身は maai_sds.fst と同じ。
# MMDAgent-EX は MDF と同じ名前の FST を読むので、maai_sds_web.mdf に合わせてこの名前で置く。
# アバターはサンプル集のジェネ。頷き・まばたきは MaAI-SDS が頭のボーンと表情を結び付けて動かし、口パクは Julius が音声から作る。
0 10 <eps> STAGE|../images/floor_block.png,../images/back_white.png
10 20 <eps> MODEL_ADD|gene|../gene/gene.pmd
20 30 MODEL_EVENT_ADD|gene MOTION_ADD|gene|base|../motions/wait/01_Wait.vmd|FULL|LOOP|OFF|OFF
30 40 <eps> CAMERA|0,16.5,0|0,0,0|20.2|30.0
