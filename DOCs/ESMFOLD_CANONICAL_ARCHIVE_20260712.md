# ESMFold 结构归档说明

该目录汇总 PPC 项目已经生成的 ESMFold 单链预测结构。原始结果目录保持不变，
`structures/` 通过同文件系统 hardlink 保存 canonical 视图，不重复占用结构文件空间。

## 当前覆盖

- 7,306 条 exact unique sequence 对应 7,306 个 canonical PDB。
- 扫描了 14,341 个原始 PDB occurrence；重复来源保留在来源清单中。
- 最新 chain-filtered SI30 universe：20,240/20,240 chains 有结构。
- 上述 final universe 对应 6,667/6,667 条 exact unique sequence。
- PDB CA 残基恢复序列与输入序列 SHA-256 不一致：0。
- 同一 exact sequence 出现不同坐标文件 hash 的冲突：0。

## 目录与映射表

- `structures/`：按 `序列长度 + SHA-256 前缀` 稳定命名的 unique PDB。
- `manifests/structure_manifest.tsv`：canonical PDB、完整序列 SHA-256、长度、
  来源、文件 SHA-256、CA B-factor/pLDDT 汇总。
- `manifests/source_inventory.tsv`：每个原始 PDB occurrence 到 canonical structure
  的映射及来源路径。
- `manifests/sequence_dedup_map.tsv`：exact sequence 去重后的结构映射。
- `manifests/all_chain_to_structure.tsv`：特征全集中的 chain 到 structure 映射。
- `manifests/final_chain_to_structure.tsv`：最新 20,240-chain split 的可逆映射；
  包含 split、seq_id、PDB ID、chain ID、component、exact group、SHA1/SHA256。
- `qc/summary.json`：覆盖率和 QC 总结。
- `qc/final_chain_unmatched.tsv`：只有表头，表示 final universe 无缺失。
- `qc/structure_model_conflicts.tsv`：只有表头，表示无同序列坐标冲突。
- `scripts/consolidate_esmfold_structures.py`：生成与复核脚本。

## 严格口径

结构映射以折叠输入的完整实验构想序列为基准，不走
PDBTM -> UniProt -> AlphaFold/TmAlphaFold bridge。脚本从每个 PDB 的有序 CA 原子
恢复实际结构序列，并要求其 SHA-256 与输入 manifest 完全相同。

PDB chain ID 大小写敏感；例如 `A` 与 `a` 是两个不同 chain。映射脚本同时验证
`SHA1(sequence)` 和 `n_residues`，不能通过对整个 seq_id 做 lowercase 来连接数据。

ESMFold 结构来自多轮分片和断点续跑。batch token、chunk size 等参数影响吞吐和
显存，不改变这里以 exact sequence hash 定义的生物学输入身份；逐次运行日志仍保留
在 PPC benchmark 的原始 run 目录中。
