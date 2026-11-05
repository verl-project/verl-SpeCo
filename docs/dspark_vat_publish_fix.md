# DSpark confidence head 发布修复

## 问题与修改

热发布过滤原先读取 `backend.confidence_head_alpha`。普通置信度损失系数实际保存在 `DSparkTrainingModel` 中，backend 没有这个属性，读取会落到默认值 0；此外，VAT 在普通置信度损失关闭时也会训练该头。因此原判断会漏发已经训练的 confidence head。

`DrafterBaseTrainer._is_frozen_publish_param` 改为复用现有模型解包方法，读取 confidence head 参数的 `requires_grad`：头可训练时允许发布，全部冻结时排除。该方法由现有完整 state dict 和选择性 FSDP2 发布路径共用。

本次不修改实验参数。DFlash 的 VAT accept head 是训练专用头，仍排除发布。零权重 KD 的额外计算不在本次修复范围内。

## 验证

回归测试检查实际发布字典中的 confidence head 权重和 bias，覆盖 VAT、普通置信度训练、冻结头，以及带 module 包装和不带包装的模型。测试在 MRv1/MRv2 环境变量下分别运行；这只验证本地发布字典，不代表 GPU 推理端已完成热加载验收。

本次发布、DSpark backend 和 VAT 回归结果为 191 passed、2 skipped、1 deselected。排除的 `test_v1_agent_loop_config_excludes_speco_drafter` 需要当前环境缺失的 `peft` 依赖。命令如下：

```bash
python -m pytest tests/integration/test_selective_publish_state.py tests/integration/test_rollout_publish_contract.py tests/integration/test_dspark_trainer_backend.py tests/unit/test_vat_loss.py --deselect tests/integration/test_rollout_publish_contract.py::test_v1_agent_loop_config_excludes_speco_drafter -q
```

Ruff 检查通过；`test_selective_publish_state.py` 保留提交前已有的 5 项 E402 导入顺序诊断，本次对此文件忽略 E402 后检查通过。
