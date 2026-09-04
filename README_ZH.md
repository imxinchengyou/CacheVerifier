# 语义缓存中的同步在线验证门禁

**Chengyou Xin** · LoopDot AI Research · 2026-07-26

[English](README.md) | 简体中文

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21703364.svg)](https://doi.org/10.5281/zenodo.21703364)

语义缓存用向量相似度替代精确匹配来复用大模型的历史回答,但相似度和答案正确性并非同一个量。这个仓库是一项实证研究的完整代码与实验产出,研究的问题很具体:在单层语义缓存架构下,用一个**真实**(非 oracle)、**同步**的验证器对缓存命中做门禁——在约 21 万条真实请求、三个数据集上,和静态阈值、自适应阈值两条基线做正式对比——到底能不能真正改善命中率与错误率之间的权衡?

![LmArena 上命中率-错误率帕累托前沿:静态阈值、自适应阈值、oracle 验证器、开箱即用验证器、领域微调验证器](results/lmarena_pareto_full_with_finetune.png)

**核心结论速览**

- **oracle 验证器**证明了这套机制确实有理论空间:在两个基准数据集上,相同错误率下命中率能提升 **20~28 个百分点**。
- 但换成一个**开箱即用**的现成 cross-encoder 验证器后,在本文最初的网格搜索评估下,这个空间只兑现了很小、且不稳健的一部分——本文的 Go/No-Go 判定是**弱 Go**,不是无条件成立。**【2026-08-15 更新】** 换成一种"诚实"的阈值选取方式重测(按时间顺序切分校准集/测试集,阈值选取时不偷看测试集)后,SearchQueries 的结论反转为在每个测试点都跑赢——原先"净损害"这个结论里,有多少是 SearchQueries 本身确实弱、有多少只是当初阈值网格没选对,现在成了本文自己都还没定论的开放问题(见 §6.1/§5.4)。
- **用数据集自己的灰色地带标签微调同一个验证器**,能在三个独立数据集上都补上大部分差距,包括把 SearchQueries 上一个**净损害**的验证器(AUC 0.60;见 [`PAPER.md`](PAPER.md) / [`PAPER_EN.md`](PAPER_EN.md) 文首勘误——早期版本曾报告 AUC 0.49,系一个已修正的数据缺陷所致)变成一个在 54 个测试点中 53 个跑赢静态阈值前沿、1 个打平、零落后的验证器(AUC 0.71)——在"诚实校准"下,三个数据集同样保持零落后。
- 这套微调方案能容忍现实量级的标签噪声(约 30%)和冷启动,也扛住了真实生产客服流量的检验——但发现了**一个真实存在的反例**,并且追溯到了一个具体、可监控的原因,还做出了一个能提前发现这个风险的监控原型。
- **【2026-08-17 更新】** 发现并修复了自适应阈值基线(Group B)复现里的一个 bug——vCache 官方算法会给每个缓存条目预置两条合成的 bootstrap 观测,本文早期的复刻版本漏掉了这一步。修复后,Group B 的命中率在三个数据集上均提升 **4.4~29.1 倍**,错误率全程仍低于目标上限。

**阅读论文:** [`PAPER.md`](PAPER.md)(中文)· [`PAPER_EN.md`](PAPER_EN.md) · [`PAPER_EN.tex`](PAPER_EN.tex)(LaTeX 源码)

**在线服务:** 本文验证的这套微调 + 漂移监控闭环,作为托管服务运行在 **[cacheverifier.com](https://www.cacheverifier.com)** —— 这个仓库是它背后的研究,不是产品本身。Python 客户端:[`cacheverifier-python`](https://github.com/imxinchengyou/cacheverifier-python)。

## 结果速览

| 数据集 | 开箱即用验证器(Group D) | 领域内微调验证器(Group E) |
|---|---|---|
| LmArena(对话式) | AUC 0.72 · 网格搜索下最佳可复现净收益约 **+1.9 个百分点**命中率 · 诚实校准下 **+5.66 个百分点** | AUC **0.88** · 网格搜索下几乎在所有测试点上跑赢静态阈值前沿 · 诚实校准下 **6/6**,**+5.66 个百分点** |
| SearchQueries(短关键词) | AUC 0.60 · 网格搜索下**净损害**(36个测试点中23个输给静态阈值) · 诚实校准下**反转为 6/6 全赢**(+0.78~+3.67 个百分点) | AUC **0.71** · 网格搜索下 54 个测试点中 53 个跑赢,1个打平,零落后 · 诚实校准下 **6/6**,**+7.74 个百分点** |
| Quora(复述问题对) | —(不在原始 benchmark 内) | 以更小幅度复现同样的模式;无论网格搜索还是诚实校准,都从未比未微调基线更差(两种方式下均零落后) |

"网格搜索"指本文最初手选阈值网格、报告其中最优点的方式。"诚实校准"指把灰色地带样本按时间顺序切成校准集/测试集,阈值只在校准集上用 Youden's J 选取,再在完全没参与选阈值的测试集上测量(见 §5.4)——这是 2026-08-15/16 追加的,专门用来检验上面网格搜索的数字是不是偏乐观;两种方法哪些地方一致、哪些地方不一致的完整说明见 §5.4/§6.1(Quora 是唯一一个诚实校准反而更差的数据集,原因追溯到这个数据集本身的可分性上限,不是校准方法的问题)。

Oracle 理论上限(机制本身的上界,两个基准数据集均适用):相同错误率下命中率 **+20~28 个百分点**。另外,自适应阈值基线(Group B)的一处复现修复让其命中率在三个数据集上提升了 **4.4~29.1 倍**(见 §5.2)——Group B 命中率量级完全不同,不参与上面的 Go/No-Go 对比。完整数字、置信区间,以及更多稳健性/消融实验(噪声、冷启动、漂移监控、τ_high 敏感性、reranker 容量与训练分布对比、Conformal Risk Control、"改写代替拒绝"、Top-K 候选级联、CRC 闭环自选择、成本敏感重分析、LLM 红队测试、对抗训练),见论文 §5.9–§5.19。

## 更多消融实验(§5.9–§5.19)

- **漂移监控(§5.9)**:仅用灰色地带标签跑两种经典的变点检测,能在真实流量反例真正造成损害之前就发出预警,同时在没问题的品牌上不误报。
- **动词分桶预筛选(§5.10)**:在三个数据集上都测试并证伪了。
- **τ_high 敏感性(§5.11)**:效果因数据集而异——放宽阈值能让 LmArena 的净收益翻两倍多,但同样的调整会让 SearchQueries 由正转负。
- **reranker 容量与训练分布对比(§5.12)**:无论是换成同分布、容量更大的 reranker,还是训练分布更广、同样更大的 reranker,都没能明显缩小 SearchQueries 的差距——领域内微调(§5.6)仍是目前唯一验证有效的办法。
- **Conformal Risk Control(§5.13)**:把灰色地带复用阈值从一个点估计升级为有限样本下的风险保证,三个数据集上的效率都接近 oracle(η≈1.0)。
- **改写代替拒绝(§5.14)**:仿照 TweakLLM 思路,拒绝后不直接放弃而是改写后照样服务,测下来相比现有的二元门禁没有可测出的净收益——一个负面消融结果。
- **Top-K 候选级联(§5.15)**:不只检索单一最近邻,在 LmArena 上接近免费收益,在 Quora 上几乎无效果,在 SearchQueries 上存在真实的 hit_rate/error_rate 权衡,微调能缓解但不能根除。
- **CRC 闭环自选择(§5.16)**:一次真正的在线闭环测试(而非静态切分)证实门禁自身的复用/拒绝决策确实会反过来塑造未来缓存状态——伤害幅度和直接命中率呈单调关系,从测不出效应(Quora)到风险翻三倍以上(LmArena);在线重新校准能在三个数据集里两个上完全补偿,第三个上补不完。
- **成本敏感重分析(§5.17)**:把命中率-错误率前沿换成显式的成本比扫描(错误代价 vs. miss 代价)——把 Group D/E 诚实校准的网格补齐到和 Group A 一致后,同步验证机制在错误代价明显高于 miss 代价(通常约 1-9 倍,视数据集而定)时几乎全程经济上胜出;初版结果曾在高代价比区间出现一个反转,补齐网格覆盖后确认是伪影,不是真实反转。
- **LLM 自动化红队测试(§5.18)**:用覆盖五个已知失效轴(否定、动作词替换、方向反转、实体替换、数量替换)的对抗样本测试,把现成验证器的错误通过率推到 84%——远高于自然数据上测出的任何数字;更关键的是,能修补自然数据判别力的领域内微调(§5.6),对这些对抗样本完全没有起到保护作用。
- **对抗训练(§5.19)**:把一小批(仅占 3.8%)、和留出测试集互不重叠的对抗训练数据混进已有的自然微调训练集,能把对抗错误通过率从 84–88% 降到 53.6%(置信区间和另外两个都不重叠),且不牺牲自然数据 AUC——§5.18 发现的鲁棒性缺口是可以补上的,不是微调机制的根本局限,但 53.6% 仍远未解决问题,且命名实体替换这一个类别训练后反而变差。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `cacheverifier/` | 缓存策略(静态/自适应/同步验证)、embedding 模型、验证器、指标、实验运行器 |
| `scripts/` | 论文中提到的数据集转换、微调、漂移监控、绘图脚本 |
| `configs/` | 各数据集的 YAML 配置(LmArena、SearchQueries、Quora、Twitter Amazon/Comcast) |
| `results/` | 论文里报告的全部指标(JSON)和图表(PNG);两个微调验证器 checkpoint 托管在 Hugging Face,没有直接放进这个仓库,见 [`results/PRETRAINED_MODELS.md`](results/PRETRAINED_MODELS.md) |
| `tests/` | `cacheverifier/` 的单元测试 |

## 复现步骤

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 验证器微调 / cross-encoder 实验需要更重的依赖:
.venv/bin/pip install -r requirements-embeddings.txt

.venv/bin/pytest tests/ -q
```

原始数据集没有直接放进这个仓库(见 `.gitignore` 里的 `data/`)——`scripts/convert_*.py` 会从论文 §4.1 引用的公开数据源(HuggingFace `vCache/SemBenchmarkLmArena` / `SemBenchmarkSearchQueries`、Quora Question Pairs、Twitter 客服语料)重新生成。之后每个 `configs/*.yaml` 驱动 `cacheverifier/experiments/run_baselines.py`(A/B 组)和 `run_verified.py`(C/D 组)跑对应数据集。

## 微调模型

Group E 的微调验证器托管在 Hugging Face Hub,不在这个仓库里——见 [`results/PRETRAINED_MODELS.md`](results/PRETRAINED_MODELS.md)。

## 引用

已经在 Zenodo 存档,概念 DOI(始终指向最新版本): [10.5281/zenodo.21703364](https://doi.org/10.5281/zenodo.21703364)
(当前版本为 v1.7.1,DOI: [10.5281/zenodo.22304547](https://doi.org/10.5281/zenodo.22304547))。arXiv 链接即将发布,届时会更新这里的引用信息。

```bibtex
@misc{xin2026synchronous,
  title  = {Synchronous Online Verification Gating in Semantic Caches: An Empirical Study},
  author = {Xin, Chengyou},
  year   = {2026},
  note   = {LoopDot AI Research},
  url    = {https://github.com/imxinchengyou/CacheVerifier},
  doi    = {10.5281/zenodo.21703364}
}
```

## 致谢

Group A/B 直接建立在 **vCache** 项目(L. G. Schroeder、A. Desai、A. Cuadron、K. Chu、S. Liu、M. Zhao、S. Krusche、A. Kemper、I. Stoica、M. Zaharia、J. E. Gonzalez)公开的 benchmark 和参考代码之上——SemCacheLmArena/SemCacheSearchQueries 数据集、静态阈值网格,以及本文逐行复刻的 `VerifiedDecisionPolicy`。5.6 节和 5.8 节还直接建立在 **Quora Question Pairs**(Iyer, Dandekar, & Csernai, 2017)和 Kaggle **"Customer Support on Twitter"**(Axelbrooke, 2017)数据集之上。完整致谢见论文正文。

## 许可

**保留所有权利** —— 见 [`LICENSE`](LICENSE)。这个仓库公开是为了支持论文结果的可复现性,不授予任何复制、修改或再分发的权利。