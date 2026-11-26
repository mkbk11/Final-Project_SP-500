from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
P=Path('數據科學開源工具_周二/期末大作业/SP500').resolve();O=P/'report_drafts';O.mkdir(exist_ok=True)
def newdoc():
 d=Document();s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11);s.top_margin=s.bottom_margin=Inches(.65);s.left_margin=s.right_margin=Inches(.72)
 for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
  st=d.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0);st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体');st.paragraph_format.space_after=Pt(5)
 
 for st in d.styles:
  for b in list(st.element.iter(qn('w:pBdr'))):b.getparent().remove(b)
 d.styles['Normal'].font.size=Pt(11);d.styles['Normal'].paragraph_format.line_spacing=1.08;d.styles['Title'].font.size=Pt(23);d.styles['Heading 1'].font.size=Pt(16);d.styles['Heading 2'].font.size=Pt(12)
 s.header.paragraphs[0].text='S&P 500  |  Data Science Project Report Draft';f=s.footer.paragraphs[0];f.alignment=2;fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)
 return d
def para(d,s,size=None):
 p=d.add_paragraph(s)
 if size:
  for r in p.runs:r.font.size=Pt(size)
 return p
def pair(d,z,e):para(d,z);para(d,e)
def head(d,s):d.add_heading(s,2)
def page(d,s):d.add_page_break();d.add_heading(s,1)
def table(d,headers,rows):
 t=d.add_table(rows=1,cols=len(headers));t.style='Table Grid'
 for c,v in zip(t.rows[0].cells,headers):c.text=str(v)
 for row in rows:
  for c,v in zip(t.add_row().cells,row):c.text=str(v)
 for i,row in enumerate(t.rows):
  row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
  for c in row.cells:
   for p in c.paragraphs:
    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
    for r in p.runs:r.font.size=Pt(10);r.bold=i==0
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EDEDED');c._tc.get_or_add_tcPr().append(sh)
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'));return t
def cap(d,s):para(d,s,9)
# Outline version
x=newdoc();x.add_paragraph('S&P 500 项目报告架构',style='Title');para(x,'S&P 500 Project Report Outline',16)
pair(x,'主题  从收益预测到交易价值的S&P 500数据科学研究','Theme  A Data Science Study of S&P 500 Return Prediction and Trading Value')
pair(x,'主线  数据可信→预测有效→交易有价值→市场状态差异。中心论点是预测分数、经济价值和证据可靠性需要共同评价。','Research chain  Trustworthy data → predictive skill → trading value → state differences. Evaluate model scores, economic value, and evidential reliability together.')
head(x,'0 封面  Cover Page');para(x,'中英文题目；团队名称；实际成员姓名与#NO.；课程和教师；提交日期；草稿状态。')
head(x,'1 引言  Introduction');para(x,'1.1 问题陈述 Problem statement\n1.2 背景与动机 Background and motivation\n1.3 相关研究 Related work\n1.4 研究问题与计划 Questions and plan\n1.5 贡献及创新 Contributions and innovations')
para(x,'先解释为什么研究、研究什么；不要提前展开本地算法或工具。贡献定位为课程研究的协议统一与审计，不声称首创新算法。')
head(x,'2 方法  Methods');para(x,'2.1 数据取得与追踪 Acquisition and provenance\n2.2 日历与异常检查 Calendar and anomaly checks\n2.3 特征探索与可用时间 EDA and feature availability\n2.4 目标与时序隔离 Targets and temporal isolation\n2.5 分类回归及基准 Classification regression and baselines\n2.6 泄漏反例与十种随机划分 Leakage and random-split controls\n2.7 交易成本及每日Double DQN Trading and daily Double DQN\n2.8 状态识别及指标定义 Regimes and evaluation metrics')
page(x,'3 结果与讨论  Results and Discussion')
para(x,'3.1 数据质量与收益分布 Data quality and return distributions\n3.2 分类与回归结果 Predictive performance\n3.3 评估反例 Evaluation controls\n3.4 原测试区间交易 Net trading on original tests\n3.5 成本与阈值敏感性 Cost and threshold sensitivity\n3.6 三种子与投票 Seed variability and voting\n3.7 共同日期与市场状态 Common dates and regimes\n3.8 无效方法与局限 Unsuccessful approaches and limitations\n3.9 运行时间与可复现性 Runtime and reproducibility')
table(x,['图表 Figure or table','证据与作用 Evidence and purpose'],[['划分表 Split table','分类与回归分别列样本数及日期'],['预测表 Prediction tables','分类AUC/AP/平衡准确率/F1；回归MAE/RMSE/R²/方向与基准'],['反例表 Controls','时序干净/泄漏；十个随机种子；解释不可直接比较条件'],['交易表与曲线 Trading','净收益/回撤/夏普/操作次数/换手/加仓/持仓/平均仓位'],['敏感性与种子 Sensitivity','五档成本；各种子、均值、标准差和多数投票'],['状态图表 Regimes','状态画像/时间线/转移/持续时间；原区间与共同日期分开']])
para(x,'讨论线索 Discussion prompts\n误差更小是否等于交易更好；复杂方法何时没有改善；随机划分为何不等于部署；状态小样本如何限制推断；同名文献指标为何不能直接排名。')
page(x,'4 结论及附录  Conclusion and Appendices')
head(x,'4 结论  Conclusion');para(x,'概括动机、方法、发现和限制；提出下一步验证方向，不保证未来收益。可选海报以问题、流程、关键证据和结论边界为四部分；不替代正文。')
head(x,'5 参考文献  References');para(x,'统一IEEE编号格式；区分期刊、会议、预印本与数据/代码；正文按编号引用，核对DOI、表号和页码。')
head(x,'6 附录A 团队贡献  Appendix A Team Contributions');para(x,'按实际人数填写姓名、学号、工作、比例；合计100%；所有成员协商确认。不预填比例或宣称达成共识。')
head(x,'7 附录B 课程评价  Appendix B Course Comments');para(x,'优点、不足、建议；涉及完整工作流、反馈、环境、协作与负面结果评价；须经成员确认。')
table(x,['完整双语稿页码 Full draft pages','内容 Content'],[['1','封面 Cover'],['2–3','引言与相关研究 Introduction and related work'],['4–7','数据、协议、模型、交易、状态 Methods'],['8–12','预测、反例、成本、种子、状态 Results'],['13','讨论与结论 Discussion and conclusion'],['14','参考文献 References'],['15','团队贡献和课程评价 Appendices']])
para(x,'提交前确认 Before submission  团队信息、贡献比例、课程教师与日期；A组阈值边界1个样本差异；学校对双语页数的解释。')
x.save(O/'SP500_Report_Outline_CN_EN.docx')
# Full report
D=newdoc();para(D,'Part 2  Project Report',13);D.add_paragraph('S&P 500 收益预测与交易价值的数据科学研究',style='Title');para(D,'A Data Science Study of S&P 500 Return Prediction and Trading Value',18)
para(D,'预测能力  交易成本  市场状态\nPredictive Skill  Transaction Costs  Market Regimes',13)
para(D,'\n课程 Course  数据科学开源工具\n教师 Instructor  [待填写 / To be completed]\n团队 Team  [待填写 / To be completed]\n成员与#NO. Members and IDs  [按实际成员填写 / Complete the actual roster]\n提交日期 Submission date  [待填写 / To be completed]')
para(D,'\n中英双语讨论稿  Bilingual Discussion Draft',12)
pair(D,'本报告以本地01至09的真实产物为证据，描述从数据审计到预测、交易和状态诊断的完整研究链。主要结论是：较小预测误差不足以单独证明交易价值，评估协议与成本同样影响结论。','This report follows the complete chain from data auditing to prediction, trading, and regime diagnostics using saved outputs from Notebooks 01–09. Lower forecast error alone does not establish trading value; evaluation design and costs also matter.')
pair(D,'成员信息、贡献比例及课程评价仍需团队确认。历史数据已参与开发，本文不是新的前瞻检验，也不构成投资建议。','Team information, effort shares, and course comments require confirmation. Historical data have informed development; this is not a new prospective test or investment recommendation.')
page(D,'1 引言  Introduction')
head(D,'1.1 问题陈述与背景  Problem and Background')
pair(D,'本项目研究三个相连的问题：历史信息对标普500短期收益和上涨事件有多大预测能力；预测能否转化为扣费后的交易价值；表现是否随市场状态变化。我们关注证据是否可靠，而不仅关注哪个模型分数最高。','We examine three connected questions: how much historical information predicts short-horizon S&P 500 returns and upward events; whether forecasts translate into trading value after fees; and whether performance varies across market states. The emphasis is reliable evidence rather than a single highest score.')
pair(D,'短期收益容易受新闻、风险偏好与突发冲击影响。价格长期上涨不意味着每日涨跌容易预测；上涨日较多也会使简单猜涨规则取得看似不错的准确率。因此，研究须回答相对于什么基准有效，并区分统计预测能力与经济价值。','Short-horizon returns respond to news, risk appetite, and unexpected shocks. Long-run growth does not imply predictable daily movements. A high frequency of positive days may make an always-up rule look accurate. We ask which benchmark is exceeded and distinguish statistical prediction from economic value.')
pair(D,'从数据科学角度，挑战不只在算法。交易日期、异常成交量、外部数据公布时间、目标区间和训练测试边界都会改变问题。若未来信息混入特征，高分可能没有可部署意义。若成本和实际持仓区间没有对应，预测成绩也不能直接解释组合收益。','The data-science challenge extends beyond algorithms. Trading dates, abnormal volume, external-data availability, target intervals, and train–test boundaries can change the task. Future information can create scores without deployment value. Forecast scores cannot explain portfolio returns unless costs and holding intervals are aligned.')
head(D,'1.2 动机与计划  Motivation and Plan')
pair(D,'我们先建立可追踪的数据基础，再检查预测能力、交易价值和状态差异，并用评估反例检验结论如何受实验设计影响。报告应让读者追溯每项结果的输入、时间区间与指标定义，也能看到没有明显改善的方法。','We establish traceable data, assess prediction and trading, examine state differences, and use evaluation controls to reveal design effects. Readers should be able to trace inputs, dates, and metric definitions, including approaches that yield little improvement.')
pair(D,'课程目标是展示完整研究过程：从业务问题转化为可评价任务，形成可复查的数据处理与实验记录，再以适当的结论边界解释结果。我们不把一段历史中的收益优势等同于未来可持续盈利。','The course objective is to show a complete process: translating a practical question into an evaluable task, preserving auditable data and experiment records, and interpreting outcomes within appropriate limits. Historical outperformance is not treated as evidence of sustainable future profit.')
page(D,'1 引言续  Related Work and Contributions')
head(D,'1.3 相关研究  Related Work')
pair(D,'Qin [1]的SPY分类会议论文报告XGBoost AUC 0.5393、准确率0.5360，低于多数类基准0.5690，说明应结合类别分布阅读准确率。但其标签是下一日复权收盘上涨，本文标签是延迟入场收益严格超过0.002，不能直接排名。','Qin [1] reports XGBoost AUC .5393 and accuracy .5360 for SPY, below a majority baseline .5690. This conference paper motivates class-aware baselines. Its adjusted-close direction target differs from our delayed-entry return above .002, preventing a direct ranking.')
pair(D,'Chun等 [2]比较岭回归、树模型和MLP，为日收益研究提供背景。其137变量、2012至2020样本外期和重估计安排均不同；R²_oos以20日移动均值为基准，本文原R²以测试均值为分母。原表MSE与R²公式存在口径核对疑点，我们保留原值、不推算其RMSE。','Chun et al. [2] compare ridge, trees, and MLPs. Their 137 predictors, 2012–2020 evaluation, and re-estimation schedule differ. Their R²_oos uses a 20-day moving-mean benchmark, whereas our original R² uses the test mean. Ambiguity between reported MSE and the R² formula prevents a defensible RMSE conversion.')
pair(D,'Zejnullahu等 [3]预印本将Double DQN用于E-mini期货，可做空并含调仓及时间费用。本项目仅空仓或做多，因此不能沿用其收益。Wang等 [4]以HMM轮换因子组合，说明状态条件评价值得研究，但该动态投资策略不等于本文冻结后的状态诊断。','Zejnullahu et al. [3] apply Double DQN to short-enabled E-mini futures with trading and time costs; their returns do not transfer to our long-or-cash setup. Wang et al. [4] motivate state-dependent evaluation through HMM factor rotation, which differs from our frozen-state diagnostics.')
head(D,'1.4 贡献与创新  Contributions')
pair(D,'我们的贡献是将数据质量、时间可用性、预测评价、扣费交易与状态诊断连接为可复查流程；在原研究上融合神经网络对照和多种子策略实验；明确未改善结果与不可比较条件。创新主要在课程项目的评估整合和审计，不是提出新的算法。','Our contribution is an auditable chain connecting data quality, availability, prediction, net trading, and regime diagnostics. We integrate a neural-network control and multi-seed policy evaluation and expose weak outcomes and comparability limits. The contribution is evaluation integration, not a new algorithm.')
page(D,'2 方法  Data Acquisition and Preparation')
head(D,'2.1 数据来源与清洗  Acquisition and Cleaning')
pair(D,'本地Yahoo Finance指数日线缓存 [7]覆盖1996年11月13日至2026年11月12日，共7547个交易日。参考仓库和核对提交见 [8]。融合读取本地缓存，不声称重新抓取全部历史。真实市场极端值保留，不用平滑处理制造更容易预测的收益。','The local Yahoo Finance index cache [7] contains 7,547 sessions from November 13, 1996 to November 12, 2026. The reference repository and audited commit are listed in [8]. Integration uses cached data, not a claimed new historical download. Genuine extremes are preserved rather than smoothed to simplify forecasting.')
pair(D,'时间戳转为纽约交易日期后，核对NYSE日历、重复日、非正价格、OHLC关系和缺失字段。记录数量与日历一致，价格检查通过。2023年5月24日成交量0被标为缺失，保留该交易日。价格按原协议保留3位小数，导出后重读验证。','After converting timestamps to New York dates, we check the NYSE calendar, duplicates, nonpositive prices, OHLC consistency, and missing fields. Counts match the calendar and prices pass checks. Zero volume on May 24, 2023 is marked missing while retaining the session. Prices retain three decimals and exports are read back for verification.')
head(D,'2.2 特征与可用时间  Features and Availability')
table(D,['用途 Use','输入 Input'],[['共同6项 Common six','1/5/20日收益、20日波动、日内收益/振幅\n1/5/20-day returns, 20-day volatility, intraday return/range'],['分类B新增 Class B additions','VIX水平/变化、2年美债变化、DXY变化、WTI变化\nVIX level/change; changes in 2-year yield, DXY and WTI'],['探索 EDA','14项价格成交量特征，不等于B组输入\n14 price/volume features, distinct from Class B inputs']])
pair(D,'跨市场日线至少滞后1个纽约日历日；2016年10月11日前美债收益率滞后7日，陈旧值最多7天。这是保守近似，不等于核实每条发布时间。截断检查、数据指纹与可用日期记录支持泄漏审计；下游读取准备好的CSV，减少重复特征构造。','Cross-market data lag at least one New York calendar day; Treasury yields lag seven days before October 11, 2016, with a seven-day staleness limit. These are conservative approximations, not verified timestamps for every release. Truncation checks, hashes, and availability records support leakage auditing. Downstream models reuse prepared CSV files.')
page(D,'2 方法续  Targets and Model Protocols')
pair(D,'t日收盘发信号，t+1开盘入场，t+2开盘结算。收益目标为open[t+2]/open[t+1]−1；分类标签仅在该收益严格大于0.002时为1。收益阈值不是概率决策阈值。训练/验证标签跨入下一分区时剔除。','Signals follow close on t, enter at the next open, and settle one open later. The target is open[t+2]/open[t+1]−1; classification equals one only above .002. This return cutoff differs from a probability threshold. Labels crossing partition boundaries are removed.')
table(D,['任务 Task','Train','Validation','Test'],[['Classification','3593\n2009-01-02–2023-04-12','448\n2023-04-14–2025-01-27','449\n2025-01-29–2026-11-10'],['Regression','6020\n1996-12-12–2020-11-11','751\n2020-11-13–2023-11-08','752\n2023-11-10–2026-11-10']])
cap(D,'日期是信号日；任务的有效样本池不同。Dates are signal dates; tasks have different valid sample pools.')
table(D,['方法 Method','训练与选择 Training and selection'],[['Classification A/B XGB [5]','深度1、eta .03、min_child_weight 20；验证AUC选轮次、平衡准确率选概率阈值\nDepth 1; validation AUC selects rounds and balanced accuracy selects cutoff'],['Regression XGB','深度3、eta .03；平方误差训练，验证MAE选9轮\nDepth 3; squared-error training, validation MAE selects 9 rounds'],['Ridge','训练期标准化，5个alpha候选，验证RMSE选10000\nTrain-only scaling; validation RMSE selects alpha 10000'],['MLP','4候选、种子42、最多300轮、耐心20\nFour candidates, seed 42, 300-epoch limit, patience 20']])
pair(D,'MLP与原回归共用6特征和分区，输入及目标仅在训练集标准化。验证选中64/32隐藏层、alpha .01、学习率.0005和第8轮。原组员20特征改为6项，内部随机验证改为外部时序验证；是协议适配，不是原实验精确复现。选定模型保存重载后评价测试。','MLP shares the six features and regression partitions; input and target scaling use training data only. Validation selects layers 64/32, alpha .01, learning rate .0005, and epoch 8. We replace the member implementation’s 20 predictors and internal random validation with the local protocol. This is adaptation, not exact replication. Frozen models are saved and reloaded for testing.')
page(D,'2 方法续  Trading and Double DQN')
head(D,'2.4 仓位与成本  Positions and Costs')
pair(D,'分类沿用验证概率阈值，A约.427763、B约.408741。回归预设预测收益>.002满仓，否则现金；另从−.002、−.001、0、.001、.002、.003、.005按验证净夏普选阈值。规则不根据测试修改。买入持有是交易基准，训练均值/中位数是预测基准。','Classification uses validation cutoffs near .427763/.408741. Regression holds for predicted returns above .002, otherwise cash. A separate control selects among −.002, −.001, 0, .001, .002, .003, and .005 by validation net Sharpe. Rules never use test selection. Buy-and-hold is a trading benchmark; train means/medians are forecast benchmarks.')
pair(D,'主成本单边5bps，初始现金，末日强制平仓收费；连续持仓不重复扣调仓费。缺失信号区间持现金，收益0。先扣调仓费用，再结算持仓收益，最后扣终止平仓费。0/3/5/10/20bps敏感性固定仓位路径，不代表不同成本下重新训练。','The main one-way fee is 5 bps. Portfolios start in cash and pay for terminal liquidation; unchanged positions incur no repeated turnover charge. Missing signals imply cash with zero yield. Costs precede market returns and terminal fees follow them. Cost sensitivity fixes paths rather than retraining policies.')
head(D,'2.5 每日Double DQN  Daily Double DQN')
pair(D,'动作只为空仓/满仓。状态包括实际当前仓位、5/20/60日收益、50/200日均线偏离、20日波动及63日回撤，已有指标直接复用。网络8→32→32→2，AdamW学习率.001、weight decay .003，批量128，1800次更新，每100步同步目标网络，梯度裁剪10。Double Q更新分离动作选择与评价 [6]。','Actions are cash/full exposure. State includes actual incoming position, 5/20/60-day returns, 50/200-day trend deviations, 20-day volatility, and 63-day drawdown. Existing features are reused. The 8–32–32–2 network uses AdamW, learning rate .001, weight decay .003, batch 128, 1,800 updates, target synchronization every 100 steps, and clipping at 10. Double Q updates separate selection and evaluation [6].')
pair(D,'沿用材料2的全动作离线更新，在历史路径计算两个动作的奖励，假定无市场冲击。下跌惩罚0/.25/.5仅按验证投票夏普选择，最终为.25。种子42/43/44全部保留，投票时三个模型看到同一实际仓位。测试冻结，不按测试挑种子或配置。','The member-derived offline update evaluates both actions on historical paths, assuming no impact. Penalties 0/.25/.5 are selected by validation voting Sharpe, yielding .25. All seeds 42/43/44 are retained. Voters observe the same actual incoming position. Test policies are frozen without seed or configuration selection on test returns.')
page(D,'2 方法续  Evaluation and Market States')
head(D,'2.6 评估反例  Evaluation Controls')
pair(D,'07保留干净时序、故意泄漏时序、随机种子42–51，每组分别训练分类/回归，共24个模型。泄漏使用未来日内收益，仅为反例。随机与时序测试数量相同但日期不同，随机训练可能含晚于测试日的数据；不作部署性能排名。07树参数与正式分类不同，不替代04。','Notebook 07 retains clean time, intentionally leaked time, and random seeds 42–51, each with classification/regression, totaling 24 models. Future intraday returns are a leakage control only. Random tests have equal counts but different dates and may train on later data, so they are not deployment rankings. Their settings differ from formal classification.')
head(D,'2.7 市场状态  Market States')
pair(D,'9项历史指标为5/20/60日动量、20/60日均线偏离、20日波动、60日回撤、相对成交量与20日均振幅。标准化、K-Means拟合、按60日动量排序命名均只用截至2020年11月11日的数据，早于全部验证期。固定k=3、种子42、n_init=20，后续只判定状态；名称不是未来涨跌真值。','Nine historical indicators include 5/20/60-day momentum, 20/60-day trend deviations, 20-day volatility, 60-day drawdown, relative volume, and 20-day average range. Scaling, fitting, and naming by 60-day momentum use data through November 11, 2020, before all validation periods. We fix k=3, seed 42, n_init=20, then only predict states; names are not future labels.')
pair(D,'先报原测试，再取449个共同信号日作事后诊断，回归/RL/持有剔除303日。分状态交易保持完整时间线，其他状态收益置0、保留原费用，不拼接不连续日期。少于20点或类别不足的相关技能指标为N/A；缺失状态不填补，转移和持续段不跨缺失日。','Original tests precede a post-hoc common 449-date analysis, removing 303 dates from regression/RL/holding. Attribution preserves the full timeline and costs while zeroing other-state returns, without joining disjoint dates. Skill is N/A below 20 observations or with insufficient classes. Transitions and durations do not bridge missing states.')
head(D,'2.8 指标定义  Metric Definitions')
pair(D,'分类用AUC、AP、准确率、平衡准确率和F1；回归用MAE、RMSE、普通R²和收益正负方向准确率，1%=100bps。交易用累计收益、含初始1的最大回撤、按252期与零现金利率的夏普、操作次数、换手、加仓日占比、持仓时间占比和平均仓位。无预测值的DDQN/持有不填预测指标。','Classification uses AUC, AP, accuracy, balanced accuracy, and F1. Regression uses MAE, RMSE, conventional R², and sign accuracy; 1%=100 bps. Trading uses return, drawdown including initial equity 1, Sharpe with 252 periods and zero cash yield, operations, turnover, entry-day fraction, holding fraction, and average exposure. Policies without forecasts receive no forecast metrics.')
page(D,'3 结果与讨论  Predictive Performance')
pair(D,'训练期目标收益超额峰度约7.46，绝对标准化收益超过3σ的比例1.46%，高于正态参考0.27%。少数极端值会影响平方误差。波动持续性图比较收益、绝对/平方收益及滚动波动；窗口重叠会提高自相关，不能由此推导收益的信息上限。','Training target returns have excess kurtosis about 7.46 and a 1.46% frequency beyond 3σ, versus a .27% normal reference. Extremes affect squared error. Persistence diagnostics compare returns, absolute/squared returns, and rolling volatility; overlapping windows inflate autocorrelation and do not establish a predictive-information limit.')
table(D,['Classification n=449','AUC','AP','Accuracy','Bal Acc','F1'],[['A 6 features','.5426','.4631','.5301','.5307','.4866'],['B 11 features','.5576','.4602','.5523','.5682','.5523']])
cap(D,'04正式历史汇总。Formal historical summaries from Notebook 04.')
pair(D,'B组AUC改善约.0150但AP略降。正类约41.65%，始终猜0的准确率约58.35%，高于两模型。04历史A组准确率53.01%，08由CSV概率重算52.78%，相差1/449；阈值边界与浮点精度影响已保留说明，不静默统一数字。','Class B improves AUC by about .0150 but slightly reduces AP. With about 41.65% positives, always-zero accuracy is about 58.35%, above both models. Class A accuracy is 53.01% in 04 and 52.78% when 08 re-thresholds saved probabilities, a one-observation difference. Threshold-boundary precision is disclosed rather than silently reconciled.')
table(D,['Regression n=752','MAE bps','RMSE bps','R²','Direction'],[['XGBoost','62.3551','92.6746','.00479','57.71%'],['Ridge','62.3406','92.4969','.00860','54.52%'],['MLP','62.4066','91.4316','.03131','52.79%'],['Train mean','62.3665','92.9958','−.00212','57.85%'],['Train median','62.0530','92.8976','−.00001','57.85%']])
pair(D,'MLP降低RMSE，却未改善MAE或方向准确率；岭回归RMSE仅比均值常数改善约.50bps。更复杂的模型没有在全部指标胜出。小幅差异不能直接当作显著性或未来优势证据。','MLP lowers RMSE without improving MAE or directional accuracy. Ridge improves RMSE over the mean by only about .50 bps. Greater complexity does not win on every metric, and small differences do not establish significance or future advantage.')
page(D,'3 结果续  Leakage and Split Controls')
table(D,['07 Evaluation control','AUC','MAE bps','RMSE bps','R²'],[['Clean time','.5644','62.3348','91.9011','.0213'],['Leaked time','.9084','35.2320','52.9871','.6747'],['Random mean 10 seeds','.5530','—','116.3598','—'],['Random sample std','.0232','—','6.7706','—']])
cap(D,'分类/回归来自不同模型；随机和时序测试日期不同。Columns describe separate tasks; random and chronological tests use different dates.')
pair(D,'故意泄漏使AUC从.5644到.9084、RMSE从91.90到52.99bps，但关键输入在决策时不存在。这不是成功预测，只展示无效信息制造高分。正式协议必须按可用时点排除该特征，不能按分数决定保留。','Leakage lifts AUC from .5644 to .9084 and lowers RMSE from 91.90 to 52.99 bps, but the input is unavailable at decision time. This is not successful forecasting. The availability audit must exclude it regardless of its score.')
pair(D,'10个随机种子AUC范围约.5077–.5840，回归RMSE范围108.17–131.89bps。随机划分在这里没有稳定改善；日期、波动与未来训练信息不同，不能把分数变化解释为算法提升，也不能声称随机划分总会给出更好成绩。','Across ten random seeds, AUC ranges about .5077–.5840 and regression RMSE 108.17–131.89 bps. Random splitting does not consistently improve scores here. Different dates, volatility, and future training information prevent attributing differences to algorithms or asserting that random splits always look better.')
head(D,'未达到目标的方法  Limited Outcomes')
table(D,['尝试 Attempt','判断与解释 Evidence and interpretation'],[['More classification features','AUC改善但AP下降，不支持全面优势\nHigher AUC but lower AP'],['MLP complexity','RMSE改善，MAE和方向未改善\nRMSE gain does not transfer to all metrics'],['Random splits','非有效部署协议；测试构成改变\nInvalid for chronological deployment comparisons'],['Future feature','高分无效，因为输入尚未发生\nLarge invalid gains from unavailable information'],['Fixed .002 cutoff','极少持仓，低回撤不足以证明有效\nSparse exposure makes drawdown misleading']])
pair(D,'组员材料的普通DQN、MLP+DQN及五日策略不属于本地正式实验，未补训或排名。材料4缺源码，状态模块按可见设计重写，不能声称原算法精确复现。','Ordinary DQN, MLP+DQN, and five-day source strategies are outside the local formal experiments and are not retrained or ranked. Missing source code for material 4 leads to a labeled design reimplementation, not exact replication.')
page(D,'3 结果续  Net Trading and Costs')
table(D,['Original test strategy','Net return','MDD','Sharpe','Trades','Exposure'],[['Class A 449','9.90%','16.87%','.436','112','49.67%'],['Class B 449','21.79%','16.68%','.821','124','58.35%'],['Regression XGB 752','7.70%','.05%','.575','2','.13%'],['Ridge 752','6.12%','6.44%','.339','6','.53%'],['MLP 752','24.79%','.05%','1.289','20','1.33%'],['DDQN vote 752','93.00%','11.07%','1.984','82','40.96%'],['Buy hold 752','71.18%','19.25%','1.295','2','100%']])
cap(D,'5bps单边；分类与其余策略日期不同，不跨区间排名。5-bps one-way fees; classification has a different original test period. Trades count individual nonzero buy/sell operations.')
pair(D,'752日原区间DDQN投票收益与夏普高于持有，回撤较小；只是历史描述。MLP仅持仓10日、回归XGB仅1日，因此.05%回撤不等于稳定低风险。低频交易可能反映预测幅度收缩，而不是可靠择时。','On the 752-day original test, voting has higher return/Sharpe and lower drawdown than holding, a historical observation. MLP holds ten days and regression XGB one day; their .05% drawdown does not imply stable low risk. Sparse trading may reflect shrunk forecasts rather than reliable timing.')
table(D,['Fixed-path return','0 bps','3 bps','5 bps','10 bps','20 bps'],[['Class B 449','29.58%','24.85%','21.79%','14.46%','1.10%'],['MLP 752','26.04%','25.29%','24.79%','23.55%','21.10%'],['DDQN vote 752','101.08%','96.19%','93.00%','85.24%','70.64%'],['Buy hold 752','71.35%','71.25%','71.18%','71.01%','70.67%']])
pair(D,'20bps时投票收益70.64%与持有70.67%接近。验证阈值XGB收益66.03%、持仓98.80%，可能主要来自接近长期持有。MLP验证阈值收益14.02%，反而低于预设24.79%；验证选择不能保证测试优势。','At 20 bps, voting returns 70.64%, near holding at 70.67%. Validation-tuned XGB returns 66.03% with 98.80% exposure, potentially reflecting near-continuous holding. Validation-tuned MLP returns 14.02%, below the preset 24.79%. Validation selection does not guarantee test superiority.')
page(D,'3 结果续  Double DQN Variability')
table(D,['Seed or policy','Net return','MDD','Sharpe','Turnover'],[['42','58.49%','11.07%','1.467','76'],['43','93.24%','5.53%','2.202','74'],['44','114.63%','16.02%','2.126','84'],['Mean','88.79%','10.88%','1.932','78'],['Sample std','28.34 pp','5.25 pp','.404','5.29'],['Majority vote','93.00%','11.07%','1.984','82']])
pair(D,'惩罚0/.25/.5的验证投票夏普为.656/1.053/.790，故选.25。测试种子收益差异大；44收益最高、回撤也最大。投票不是按测试挑最佳种子，种子均值也不是第四个可交易组合。','Validation voting Sharpe is .656/1.053/.790 for penalties 0/.25/.5, selecting .25. Seed outcomes vary materially; 44 has the highest return and drawdown. Voting is not test-based seed selection, and the seed mean is not a fourth tradable portfolio.')
D.add_picture(str(O/'assets/equity.png'),width=Inches(6.65));cap(D,'图1 相同752日原区间净值和回撤，含初始1与末日平仓。Figure 1. Same 752-day original test, including initial equity and liquidation.')
pair(D,'文献 [3]表5最好夏普.915低于本地1.984，但合约、做空权限、成本和时期不同，不能称算法领先。离线全动作奖励假定无市场冲击，真实成交、滑点和新的未见时期仍未检验。','The best Sharpe in [3], Table 5, is .915 versus our 1.984, but contracts, shorting, fees, and dates differ. This is not algorithmic superiority. Offline rewards assume no impact; real execution, slippage, and a new unseen period remain untested.')
page(D,'3 结果续  Market State Diagnostics')
table(D,['State','Train 60d momentum','Train 20d volatility','Common test n'],[['Bull','5.19%','.83%','361'],['Sideways','−3.93%','1.34%','80'],['Bear','−20.41%','3.48%','8']])
pair(D,'共同日期2025年1月29日至2026年11月10日，其余任务剔除303日。全历史牛/震荡/熊自转移概率94.86%/85.58%/84.82%，平均持续19.32/6.93/6.59日。这描述模型状态序列，不是对真实状态的识别准确率。','Common dates run from January 29, 2025 to November 10, 2026, removing 303 observations from other tasks. Full-history self-transition probabilities for bull/sideways/bear are 94.86%/85.58%/84.82%; mean durations are 19.32/6.93/6.59 sessions. These describe assigned states, not accuracy against true regimes.')
D.add_picture(str(O/'assets/regimes.png'),width=Inches(6.65));cap(D,'图2 共同日期样本数及AUC，熊市8点记N/A。Figure 2. Common-date support and AUC; bear-state skill is N/A at n=8.')
table(D,['Common-date measure','Bull','Sideways','Bear'],[['Class B AUC','.5790','.4449','N/A'],['MLP RMSE bps','70.87','140.39','N/A'],['Buy hold attributed return','20.80%','−9.02%','N/A'],['DDQN attributed return','33.78%','.61%','N/A']])
pair(D,'B组震荡AUC低于.5、MLP误差较高，说明总成绩掩盖条件差异。但动量和波动本来参与分组，不能作因果解释。状态归因保留完整时间线与原成本，不是择时回测，且状态收益率不能直接相加。','Lower sideways AUC and higher MLP error reveal heterogeneity. Momentum and volatility define the groups, so this is not causal evidence. Attribution preserves the timeline and original costs; it is not a timing strategy and state return percentages are not additive.')
page(D,'3 讨论及4结论  Discussion and Conclusion')
head(D,'3.8 可复现性与运行时间  Reproducibility and Runtime')
pair(D,'记录数据SHA256、参数、种子、来源、模型和预测；检查重载一致性、日期连接、初始净值及标量费用复算。新结果独立保存。Python 3.11.16配NumPy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1、XGBoost 3.2.0及CPU PyTorch 2.6.0。','Records include SHA256 hashes, parameters, seeds, provenance, models, and predictions. Checks cover reload consistency, joins, initial equity, and independent scalar fees. Outputs are isolated. Python 3.11.16 uses NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.2.0, and CPU PyTorch 2.6.0.')
table(D,['Existing notebook log','Seconds','Scope'],[['05 verification copy','11.31','Models, I/O and display'],['08 trading and DDQN','72.59','Training, evaluation and plots'],['09 regime diagnostics','10.66','Clustering, evaluation and plots']])
cap(D,'单次Notebook执行总时，不是统一硬件的模型速度排名；不含后续文献节。峰值内存与吞吐量未测量。Single-run elapsed times, not controlled speed rankings; later literature sections excluded. Peak memory and throughput were not measured.')
head(D,'3.9 限制  Limitations')
pair(D,'单次历史留出曾参与开发，未做新的前瞻测试或显著性证明。指数不可直接交易，模拟未计滑点、冲击、税费与现金利息；外部发布时间是保守近似。三个RL种子不足以刻画全部不确定性。材料4缺源码，组员适配与文献指标差异均已披露，不声称精确复现。','The historically explored holdout is not prospective validation or significance proof. The index is not directly tradable. Slippage, impact, taxes, and cash interest are omitted, while external timing uses conservative approximations. Three RL seeds do not exhaust uncertainty. Missing code, adaptations, and literature differences prevent exact-replication claims.')
head(D,'4 结论  Conclusion')
pair(D,'我们完成从数据审计、特征可用性、时间隔离到预测、扣费交易和状态诊断的完整研究。MLP降低RMSE但并非全面改善；投票DDQN在本次历史区间较强，但对成本和种子敏感；震荡状态表现弱、熊市样本不足。数据科学的关键是将模型分数、经济价值与证据可靠性共同评价。','We complete the chain from data audits, feature availability, and temporal isolation to forecasts, net trading, and regimes. MLP improves RMSE without broad gains. Voting DDQN is strong historically but sensitive to costs and seeds. Sideways performance is weak and bear support inadequate. Model scores, economic value, and evidential reliability must be evaluated together.')
page(D,'5 参考文献  References')
references=[
'[1] M. Qin, “Empirical research on daily direction prediction of S&P 500 ETF based on machine learning and multivariate technical indicators,” in Proc. AITIA, vol. 16, pp. 412–420, 2026, doi: 10.70267/aitia.2026412420.',
'[2] D. Chun, J. Kang, and J. Kim, “Forecasting returns with machine learning and optimizing global portfolios: Evidence from the Korean and U.S. stock markets,” Financial Innovation, vol. 10, Art. no. 124, 2024, doi: 10.1186/s40854-024-00648-w.',
'[3] F. Zejnullahu, M. Moser, and J. Osterrieder, “Applications of reinforcement learning in finance—Trading with a double deep Q-network,” arXiv:2206.14267, ver. 1, 2022. [Online]. Available: https://arxiv.org/abs/2206.14267.',
'[4] M. Wang, Y.-H. Lin, and I. Mikhelson, “Regime-switching factor investing with hidden Markov models,” J. Risk Financial Manag., vol. 13, no. 12, Art. no. 311, 2020, doi: 10.3390/jrfm13120311.',
'[5] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.',
'[6] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double Q-learning,” arXiv:1509.06461, 2015. [Online]. Available: https://arxiv.org/abs/1509.06461.',
'[7] Yahoo Finance, “S&P 500 historical data,” symbol ^GSPC. [Online]. Available: https://finance.yahoo.com/quote/%5EGSPC/history/. Accessed: Nov. 24, 2026. Local cached observations used; no new download claimed.',
'[8] mkbk11, “Final-Project_SP-500,” GitHub repository, commit 8daef6bac0bc57977cd9e5086b8d42b3d904f940. [Online]. Available: https://github.com/mkbk11/Final-Project_SP-500. Accessed: Nov. 24, 2026.'
]
for r in references:para(D,r,10)
head(D,'组员材料来源  Member Materials')
pair(D,'未公开组员材料非期刊论文：材料1用于交易/MLP，材料2用于每日Double DQN，材料3用于分布与自相关设计，材料4用于状态分析可见设计。具体文件与指纹见member_sources.json，适配详见Notebook。','Unpublished member materials are not journal papers. Material 1 informs trading/MLP, material 2 daily DDQN, material 3 distribution/autocorrelation plots, and material 4 the reimplemented regime design. File identities and hashes appear in member_sources.json; notebooks document adaptations.')
page(D,'6及7 附录  Team Contributions and Course Comments')
head(D,'6 附录A 团队贡献  Appendix A Team Contributions')
pair(D,'按实际人数调整下表，职责只是待讨论的工作范围。比例须全体协商确认、合计100%；本草稿不代表已经达成共识。','Resize the table for the actual team. Work areas are suggestions, not established ownership. All members must agree on shares totaling 100%; this draft does not assert agreement.')
table(D,['姓名及#NO. Name and ID','工作 Work to confirm','Share'],[['[待填]','数据与清洗 Data and cleaning','[待协商]'],['[待填]','特征与预测 Features and prediction','[待协商]'],['[待填]','交易及RL Trading and RL','[待协商]'],['[待填]','状态与复核 Regimes and checks','[待协商]'],['[按人数调整]','整合与报告 Integration and report','[待协商]'],['Total','全体确认日期 Agreement date [待填]','100% after agreement']])
head(D,'7 附录B 课程评价  Appendix B Course Comments')
para(D,'以下供讨论，不代表已收集或一致认可的个人反馈。Proposed wording, not collected or unanimously approved feedback.',10)
pair(D,'优点建议稿  项目连接数据处理、建模和结果表达，有助于理解开源工具协作。强调时间隔离、简单基准和可复现性，能帮助学生形成完整的数据科学工作习惯。','Proposed strengths  Projects connect data preparation, modeling, and communication and show how open-source tools work together. Temporal isolation, simple baselines, and reproducibility encourage a complete data-science workflow.')
pair(D,'不足与建议稿  可进一步明确里程碑、贡献记录和评分标准；增加真实异常、泄漏排查及成本计算案例；提供固定环境示例和早期报告反馈。建议同时评价负面结果解释和实验记录，而不只奖励高准确率。','Proposed improvements  Clarify milestones, contribution records, and assessment criteria; add cases on anomalies, leakage, and costs; provide reproducible environments and early report feedback. Reward negative-result explanations and experiment records as well as accuracy.')
D.save(O/'SP500_Report_Draft_CN_EN.docx')
print('Two bilingual drafts saved')
