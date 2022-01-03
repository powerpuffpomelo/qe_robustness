# 在qe测试集上
```
PRE_PREFIX=/data1/yanym/data/robust_qe_test_analysis/result
GOLD_PREFIX=/data1/yanym/data/robust_qe_test_analysis/result
echo "====== 基线模型 训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/robust.tag_gold.BPE $PRE_PREFIX/baseline_robust.tag_pred.BPE
echo "====== 基线模型 没训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/not_robust.tag_gold.BPE $PRE_PREFIX/baseline_not_robust.tag_pred.BPE
echo "====== robust_aug_v1模型 训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/robust.tag_gold.BPE $PRE_PREFIX/robust_aug_v1_robust.tag_pred.BPE
echo "====== robust_aug_v1模型 没训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/not_robust.tag_gold.BPE $PRE_PREFIX/robust_aug_v1_not_robust.tag_pred.BPE
```

# 在鲁棒性测试集上
PRE_PREFIX=/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result
GOLD_PREFIX=/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result
echo "====== 基线模型 训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/robust.tag_gold.BPE $PRE_PREFIX/baseline_robust.tag_pred.BPE
echo "====== 基线模型 没训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/not_robust.tag_gold.BPE $PRE_PREFIX/baseline_not_robust.tag_pred.BPE
echo "====== robust_aug_v1模型 训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/robust.tag_gold.BPE $PRE_PREFIX/v1_robust.tag_pred.BPE
echo "====== robust_aug_v1模型 没训练过鲁棒性的token集合 预测结果"
python estimate_word.py $GOLD_PREFIX/not_robust.tag_gold.BPE $PRE_PREFIX/v1_not_robust.tag_pred.BPE

# bash scripts/analysis/robust/test_qe_robust_token.sh