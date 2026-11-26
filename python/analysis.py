"""S&P 500 trading and market-state analysis (stages 08 and 09)."""
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score, f1_score, r2_score
from experiment_utils import (ROOT, ORIGINAL as FEATURES, read_fusion_frame as read_frame,
    save_json as write_json, new_fusion_output as new_output,
    latest_completed_run as complete_run, latest_training_folder as training_folder,
    regression_scores, load_fusion_inputs, backtest, trading_scores)

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


def trading():
    OUT = new_output('08_trading')
    MLP_DIR, REG_RUN, CLS_RUN, PRED, THRESHOLDS, META, prices = load_fusion_inputs()

    import torch
    from torch import nn
    from copy import deepcopy
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    SEEDS=(42,43,44); UPDATES=1800; GAMMA=.95**(1/5); FEE=.0005
    DDQN_FEATURES=['return_5','return_20','return_60','trend_50','trend_200','volatility_20','drawdown_63']
    market=pd.DataFrame(index=prices.index)
    existing=read_frame(ROOT/'data/features/sp500_features.csv')
    for n in [5,20]: market[f'return_{n}']=existing[f'return_{n}d'].reindex(market.index)
    market['return_60']=prices.close.pct_change(60)
    for n in [50,200]: market[f'trend_{n}']=prices.close/prices.close.rolling(n).mean()-1
    market['volatility_20']=existing['volatility_20d'].reindex(market.index)
    market['drawdown_63']=prices.close/prices.close.rolling(63).max()-1
    # All features are known at signal-day close. No extra shift is needed here.
    blocks={s:f.join(market).dropna(subset=DDQN_FEATURES) for s,f in PRED['Ridge'].items()}
    scaler=StandardScaler().fit(blocks['train'][DDQN_FEATURES])  # Fit DDQN scaling on train only

    def make_network():
        return nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))

    def train_ddqn(seed,downside):
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        net=make_network();target=deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
        f=blocks['train']; x=np.repeat(scaler.transform(f[DDQN_FEATURES]),2,axis=0)
        positions=np.tile([0,1],len(f)); actions=np.array([0,1])
        states=torch.tensor(np.column_stack([x,positions]),dtype=torch.float32)
        gross=1+np.repeat(f.target_return.to_numpy(),2)[:,None]*actions
        costs=1-FEE*abs(positions[:,None]-actions)
        # Last transition includes liquidation, matching evaluation.
        gross[-2:]*=(1-FEE*actions)  # Include the final liquidation cost in reward
        logs=np.log(gross*costs)
        reward=torch.tensor(100*(logs+downside*np.minimum(logs,0)),dtype=torch.float32)
        next_rows=np.minimum(np.arange(len(f))+1,len(f)-1)
        next_ids=torch.tensor(2*np.repeat(next_rows,2)[:,None]+actions)
        mask=torch.ones((2*len(f),1));mask[-2:]=0
        for update in range(UPDATES):
            batch=rng.integers(len(states),size=128)
            with torch.no_grad():
                ns=states[next_ids[batch]]
                act=net(ns).argmax(-1,keepdim=True)
                future=target(ns).gather(-1,act).squeeze(-1)
                desired=reward[batch]+GAMMA*mask[batch]*future  # Compute a bootstrapped Double DQN target
            loss=nn.functional.smooth_l1_loss(net(states[batch]),desired)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
            if (update+1)%100==0:target.load_state_dict(net.state_dict())
        return net.eval()

    def policy_path(models,frame):
        x=np.repeat(scaler.transform(frame[DDQN_FEATURES]),2,axis=0)
        states=torch.tensor(np.column_stack([x,np.tile([0,1],len(frame))]),dtype=torch.float32)
        with torch.no_grad(): choices=np.stack([m(states).reshape(-1,2,2).argmax(-1).numpy() for m in models])
        prior=0; path=[]
        for i in range(len(frame)):
            # Every voter sees the same incoming position of this actual path.
            pos=int(choices[:,i,prior].sum()>len(models)//2)
            path.append(pos);prior=pos
        return np.asarray(path)

    search=[]; fitted={}
    for downside in [0.,.25,.5]:
        models=[train_ddqn(seed,downside) for seed in SEEDS]
        h=backtest(blocks['val'],policy_path(models,blocks['val']))
        search.append({'downside':downside,**trading_scores(h)})
        fitted[downside]=models
        print('DDQN validation',downside,trading_scores(h)['sharpe'],flush=True)
    selection=max(search,key=lambda r: r['sharpe'] if np.isfinite(r['sharpe']) else -np.inf)['downside']  # Select downside penalty by validation Sharpe
    models=fitted[selection]
    pd.DataFrame(search).to_csv(OUT/'ddqn_validation_search.csv',index=False)
    joblib.dump(scaler,OUT/'ddqn_scaler.joblib')
    for seed,m in zip(SEEDS,models):
        torch.save(m.state_dict(),OUT/f'ddqn_{seed}.pt')
        restored=make_network();restored.load_state_dict(torch.load(OUT/f'ddqn_{seed}.pt',weights_only=True));restored.eval()
        np.testing.assert_array_equal(policy_path([m],blocks['test']),policy_path([restored],blocks['test']))
    write_json(OUT/'ddqn_config.json',{'source':'material2/SP500.ipynb cell19-20','seeds':SEEDS,'updates':UPDATES,'batch':128,'network':[8,32,32,2],'optimizer':'AdamW','lr':.001,'weight_decay':.003,'target_update':100,'gradient_clip':10,'loss':'smooth_l1','gamma':GAMMA,'fee':FEE,'downside_candidates':[0,.25,.5],'selected_downside':selection,'selection':'validation vote Sharpe','features':DDQN_FEATURES,'reload_actions_exact':True})


    HIST={}; seed_rows=[]; threshold_rows=[]
    for name,parts in PRED.items():
        f=parts['test']; task=META[name]['task']; th=THRESHOLDS[name]
        pos=(f.prediction>=th).astype(int) if task=='classification' else (f.prediction>th).astype(int)
        HIST[name]=backtest(f,pos)
        if task=='regression':
            val=parts['val']; candidates=[-.002,-.001,0,.001,.002,.003,.005]
            scores=[trading_scores(backtest(val,(val.prediction>t).astype(int)))['sharpe'] for t in candidates]
            chosen=candidates[max(range(len(scores)),key=lambda i:scores[i] if np.isfinite(scores[i]) else -np.inf)]
            HIST[name+' val threshold']=backtest(f,(f.prediction>chosen).astype(int))
            threshold_rows.extend([{'model':name,'threshold':t,'val_sharpe':v,'selected':t==chosen} for t,v in zip(candidates,scores)])
    HIST['Buy hold']=backtest(PRED['Ridge']['test'],np.ones(len(PRED['Ridge']['test'])))
    for seed,m in zip(SEEDS,models):
        h=backtest(blocks['test'],policy_path([m],blocks['test']))
        HIST[f'DDQN seed {seed}']=h;seed_rows.append({'seed':seed,**trading_scores(h)})
    HIST['DDQN vote']=backtest(blocks['test'],policy_path(models,blocks['test']))  # Vote on positions from three saved seeds
    seed_table=pd.DataFrame(seed_rows).set_index('seed');display(seed_table)
    seed_summary=seed_table.agg(['mean','std']);display(seed_summary)
    seed_table.to_csv(OUT/'ddqn_seed_metrics.csv');seed_summary.to_csv(OUT/'ddqn_seed_mean_std.csv')
    pd.DataFrame(threshold_rows).to_csv(OUT/'regression_threshold_search.csv',index=False)
    rows=[]; sensitivity=[]
    for name,h in HIST.items():
        h.to_csv(OUT/(name.replace(' ','_')+'.csv'))
        rows.append({'strategy':name,'first_signal':h.index.min(),'last_signal':h.index.max(),**trading_scores(h)})
        for bps in [0,3,5,10,20]:
            sensitivity.append({'strategy':name,'cost_bps':bps,**trading_scores(backtest(h,h.position,bps/10000))})
    table=pd.DataFrame(rows);table.to_csv(OUT/'trading_metrics.csv',index=False);display(table)
    sens=pd.DataFrame(sensitivity);sens.to_csv(OUT/'cost_sensitivity.csv',index=False)
    display(sens.pivot(index='strategy',columns='cost_bps',values='total_return'))
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for name,h in HIST.items():
        if 'seed' in name or 'val threshold' in name:continue
        axes[0].plot([h.trade_date.iloc[0],*h.settle_date],[1,*h.equity],label=name)
        axes[1].plot([h.trade_date.iloc[0],*h.settle_date],[0,*h.drawdown],label=name)
    axes[0].legend(ncol=3);axes[0].set_ylabel('Equity');axes[1].set_ylabel('Drawdown')
    fig.tight_layout();fig.savefig(OUT/'equity_drawdown.png');display(fig);plt.close(fig)


    prediction_rows=[]
    for name,parts in PRED.items():
        f=parts['test'];y=f.target_return;p=f.prediction
        if META[name]['task']=='classification':
            label=f.target_class
            scores={'n':len(f),'roc_auc':roc_auc_score(label,p),'ap':average_precision_score(label,p),
                    'accuracy':accuracy_score(label,p>=THRESHOLDS[name]),'balanced_accuracy':balanced_accuracy_score(label,p>=THRESHOLDS[name]),'f1':f1_score(label,p>=THRESHOLDS[name],zero_division=0)}
        else:scores=regression_scores(y,p)
        prediction_rows.append({'model':name,'task':META[name]['task'],**scores})
    pd.DataFrame(prediction_rows).to_csv(OUT/'prediction_metrics.csv',index=False);display(pd.DataFrame(prediction_rows))
    write_json(OUT/'sources.json',{'classification':str(CLS_RUN.relative_to(ROOT)),'regression':str(REG_RUN.relative_to(ROOT)),'mlp':str(MLP_DIR.relative_to(ROOT))})
    # Cost and initial-equity checks.
    example=PRED['Ridge']['test'].iloc[:2]
    h=backtest(example,[1,1],.0005)
    np.testing.assert_allclose(h.equity.iloc[-1],(.9995**2)*np.prod(1+example.target_return))
    assert h.turnover.sum()==2
    loss_example=example.copy();loss_example['target_return']=[-.1,0]
    assert backtest(loss_example,[1,1],0).drawdown.min()<=-.099999
    assert all(h.index.is_unique for h in HIST.values())
    write_json(OUT/'verification.json',{'return_alignment':True,'initial_and_final_cost':True,'initial_equity_in_drawdown':True,'MLP_reload':True,'DDQN_reload':True,'seeds':SEEDS,'test_selection':False})
    write_json(ROOT/'output/fusion/trading_latest.json',{'path':str(OUT.relative_to(ROOT))})
    print('Saved:',OUT.relative_to(ROOT))


    if os.environ.get("SP500_INCLUDE_LITERATURE") == "1":
        SECTION='08_trading'
        assert (ROOT/'01_data_preprocessing.ipynb').is_file()
        LIT_OUT = ROOT/'output/literature'/SECTION/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        LIT_OUT.mkdir(parents=True)
        trade_dir=ROOT/json.loads((ROOT/'output/fusion/trading_latest.json').read_text())['path'];path=trade_dir/'trading_metrics.csv';INPUTS=[path]
        f=pd.read_csv(path);rows=[]
        for _,r in f.iterrows():
            rows.append({'origin':'local','strategy':r.strategy,'n':r.n,'first_signal':r.first_signal,'last_signal':r.last_signal,'Sharpe':r.sharpe,'cumulative_return':r.total_return,'max_drawdown':r.max_drawdown,'annual_mean_return_paper':None,'cost':'5bps one-way; cash yield 0; final liquidation'})
        for name,sh,ann in [('Market',.883,.2232),('Model 0',.785,.1977),('Model 1',.915,.2287),('Model 2',-.219,-.0547),('Model 3',.676,.1591)]:
            rows.append({'origin':'paper reported','strategy':'Zejnullahu '+name,'n':None,'first_signal':'NR (figure: autumn 2019)','last_signal':'NR (figure: autumn 2021)','Sharpe':sh,'cumulative_return':None,'max_drawdown':None,'annual_mean_return_paper':ann,'cost':'1bps turnover + 0.1bps time cost; table5/table10'})
        PAPER={'authors':'Frensi Zejnullahu; Maurice Moser; Joerg Osterrieder','year':2022,'url':'https://arxiv.org/pdf/2206.14267','version':'v1','location':'Table5 PDF p12; Table10 PDF p18; Appendix7.2 PDF p17','publication':'arXiv preprint','cost_caption_ambiguity':True,'not_reproduced':True}
        literature_table=pd.DataFrame(rows)

        literature_table.to_csv(LIT_OUT/'literature_comparison.csv',index=False,encoding='utf-8-sig')
        (LIT_OUT/'extraction.json').write_text(json.dumps({'checked_on':'2026-11-24','paper':PAPER,'table':literature_table.replace({np.nan:None}).to_dict('records'),'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}},ensure_ascii=False,indent=2),encoding='utf-8')
        display(literature_table.fillna('N/A'))
        print('Saved:',LIT_OUT.relative_to(ROOT))


def regime():
    OUT = new_output('09_regime')
    MLP_DIR, REG_RUN, CLS_RUN, PRED, THRESHOLDS, META, prices = load_fusion_inputs()

    from sklearn.cluster import KMeans
    TRADING_DIR=ROOT/json.loads((ROOT/'output/fusion/trading_latest.json').read_text())['path']
    source=json.loads((TRADING_DIR/'sources.json').read_text())
    assert source=={'classification':str(CLS_RUN.relative_to(ROOT)),'regression':str(REG_RUN.relative_to(ROOT)),'mlp':str(MLP_DIR.relative_to(ROOT))}
    cutoff=min(p['train'].index.max() for p in PRED.values())  # Use the earliest model training cutoff
    assert cutoff < min(p['val'].index.min() for p in PRED.values())
    c=prices.close;sf=pd.DataFrame(index=prices.index)
    for n in [5,20,60]:sf[f'momentum_{n}']=c.pct_change(n)
    for n in [20,60]:sf[f'trend_{n}']=c/c.rolling(n).mean()-1
    sf['volatility_20']=c.pct_change().rolling(20).std()
    sf['drawdown_60']=c/c.rolling(60).max()-1
    sf['volume_ratio']=prices.volume/prices.volume.rolling(20).mean()-1
    sf['range_20']=((prices.high-prices.low)/c).rolling(20).mean()
    sf=sf.replace([np.inf,-np.inf],np.nan).dropna();fit=sf.loc[:cutoff]
    scaler=StandardScaler().fit(fit);km=KMeans(n_clusters=3,n_init=20,random_state=42).fit(scaler.transform(fit))  # Fit regime scaling before validation dates
    train_label=pd.Series(km.labels_,index=fit.index)
    order=fit.groupby(train_label).momentum_60.mean().sort_values().index  # Name clusters by train-only momentum
    names=dict(zip(map(int,order),['Bear','Sideways','Bull']))
    states=pd.Series([names[i] for i in km.predict(scaler.transform(sf))],index=sf.index,name='state')  # Freeze clustering before assigning later states
    profile=fit.assign(state=train_label.map(names)).groupby('state').mean();profile['n']=train_label.map(names).value_counts()
    profile.to_csv(OUT/'training_state_profile.csv');display(profile)
    states.to_csv(OUT/'states.csv')
    joblib.dump({'scaler':scaler,'kmeans':km,'names':names},OUT/'regime.joblib')
    loaded=joblib.load(OUT/'regime.joblib');np.testing.assert_array_equal(km.predict(scaler.transform(sf)),loaded['kmeans'].predict(loaded['scaler'].transform(sf)))
    write_json(OUT/'config.json',{'source':'material4 reference design reimplementation','cutoff':cutoff,'train_n':len(fit),'features':list(sf.columns),'seed':42,'k':3,'n_init':20,'naming':'training momentum_60 ascending','sources':source})
    print('Unified training cutoff:',cutoff)
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for name,color in [('Bear','crimson'),('Sideways','goldenrod'),('Bull','teal')]:
     idx=states.index[states==name];axes[0].scatter(idx,c.reindex(idx),s=3,color=color,label=name)
    axes[0].axvline(cutoff,color='black',ls='--');axes[0].legend();axes[0].set_title('Historical state timeline')
    prof=fit.assign(state=train_label.map(names)).groupby('state').mean()
    axes[1].imshow(scaler.transform(prof),aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
    axes[1].set_yticks(range(3),prof.index);axes[1].set_xticks(range(len(sf.columns)),sf.columns,rotation=25,ha='right')
    axes[1].set_title('Training profiles (standardized)');fig.tight_layout();fig.savefig(OUT/'timeline_profiles.png');display(fig);plt.close(fig)
    previous_state=states.reindex(prices.index).shift().reindex(states.index)
    trans=pd.crosstab(previous_state,states).reindex(index=['Bear','Sideways','Bull'],columns=['Bear','Sideways','Bull'],fill_value=0)
    trans=trans.div(trans.sum(axis=1),axis=0);trans.to_csv(OUT/'transition_matrix.csv');display(trans)
    breaks=states.ne(previous_state)
    segments=states.groupby(breaks.cumsum()).agg(['first','size']);segments.columns=['state','sessions'];segments.index.name=None
    duration=segments.groupby('state').sessions.agg(['count','mean','median','max']);duration.to_csv(OUT/'state_duration.csv');display(duration)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].imshow(trans,vmin=0,vmax=1,cmap='Blues')
    axes[0].set_xticks(range(3),trans.columns);axes[0].set_yticks(range(3),trans.index)
    for i in range(3):
     for j in range(3):axes[0].text(j,i,f'{trans.iloc[i,j]:.2f}',ha='center')
    duration['mean'].plot.bar(ax=axes[1],title='Mean duration (sessions)');fig.tight_layout();fig.savefig(OUT/'transition_duration.png');display(fig);plt.close(fig)


    HIST={}
    for path in TRADING_DIR.glob('*.csv'):
     if path.name in ['trading_metrics.csv','cost_sensitivity.csv','prediction_metrics.csv','ddqn_seed_metrics.csv','ddqn_seed_mean_std.csv','ddqn_validation_search.csv','regression_threshold_search.csv']:continue
     f=read_frame(path)
     if 'net_return' in f:HIST[path.stem.replace('_',' ')]=f
    common=set.intersection(*[set(f.index) for f in HIST.values()],*[set(p['test'].index) for p in PRED.values()],set(states.index))  # Compare models on identical signal dates
    common=pd.DatetimeIndex(sorted(common),name='signal_date');assert len(common)>0
    exclusions=pd.DataFrame([{'model_or_strategy':n,'original_n':len(f),'common_n':len(common),'excluded':len(f)-len(common),'first_common':common.min(),'last_common':common.max()} for n,f in [(n,p['test']) for n,p in PRED.items()]+list(HIST.items())])
    exclusions.to_csv(OUT/'common_date_exclusions.csv',index=False);display(exclusions)
    rows=[]
    for period in ['original_test','common_dates_posthoc']:
     for name,parts in PRED.items():
      f=parts['test'].copy()
      if period=='common_dates_posthoc':f=f.loc[common]
      f=f.join(states)
      for state in ['Bear','Sideways','Bull']:
       g=f.loc[f.state==state];y=g.target_return;p=g.prediction;label=g.target_class
       support={'n':len(g),'first_signal':g.index.min(),'last_signal':g.index.max(),'class_0':int((label==0).sum()),'class_1':int((label==1).sum()),'positive_return_fraction':(y>0).mean()}
       task=META[name]['task'];reason='N/A: fewer than 20 samples' if len(g)<20 else ''  # Avoid unstable small-sample metrics
       if task=='classification':
        valid=len(g)>=20 and label.nunique()==2
        if len(g)>=20 and label.nunique()<2:reason='N/A: only one label class'
        pred=p>=THRESHOLDS[name];scores={k:np.nan for k in ['roc_auc','ap','accuracy','balanced_accuracy','f1']}
        if valid:scores.update(roc_auc=roc_auc_score(label,p),ap=average_precision_score(label,p),accuracy=accuracy_score(label,pred),balanced_accuracy=balanced_accuracy_score(label,pred),f1=f1_score(label,pred,zero_division=0))
        majority=int(parts['train'].target_class.mean()>=.5);scores['train_majority_accuracy']=(label==majority).mean() if len(g) else np.nan
       else:
        scores={k:np.nan for k in ['mae_bp','rmse_bp','r2','direction_accuracy']}
        if len(g)>=20:scores.update(regression_scores(y,p));scores.pop('n',None)
        for baseline in ['train_mean','train_median']:
         if len(g)>=20:
          b=regression_scores(y,np.repeat(META[name][baseline],len(g)));scores.update({baseline+'_'+k:v for k,v in b.items() if k!='n'})
        scores['always_up_direction']=(y>0).mean() if len(g) else np.nan
       rows.append({'period':period,'model':name,'task':task,'state':state,**support,**scores,'note':reason})
    pred_table=pd.DataFrame(rows);pred_table.to_csv(OUT/'prediction_by_state.csv',index=False);display(pred_table.fillna('N/A'))


    rows=[]
    for period in ['original_test','common_dates_posthoc']:
     for name,h in HIST.items():
      if period=='common_dates_posthoc':
       timeline=prices.index[(prices.index>=common.min()) & (prices.index<=common.max())]
       h=h.reindex(timeline)
       for col in ['net_return','position','turnover','entry_turnover','exit_turnover','add_day']:h[col]=h[col].fillna(0)
      for state in ['Bear','Sideways','Bull']:
       mask=states.reindex(h.index).eq(state)
       if period=='common_dates_posthoc':mask &= h.index.isin(common)
       sub=h.loc[mask];g=h.copy();g['net_return']=h.net_return.where(mask,0.)  # Attribute realized returns on the full calendar
       g['equity']=(1+g.net_return).cumprod();g['drawdown']=g.equity/np.maximum.accumulate(np.r_[1,g.equity])[1:]-1  # Compound masked returns without joining distant dates
       scores=trading_scores(g);scores.update(n=len(sub),turnover=sub.turnover.sum(),trades=int((sub.entry_turnover>0).sum()+(sub.exit_turnover>0).sum()),add_day_fraction=sub.add_day.mean(),holding_time_fraction=(sub.position>0).mean(),average_position=sub.position.mean())
       if len(sub)<20:
        for k in ['total_return','max_drawdown','sharpe']:scores[k]=np.nan
       rows.append({'period':period,'strategy':name,'state':state,'first_signal':sub.index.min(),'last_signal':sub.index.max(),'timeline_n':len(h),**scores,'note':'N/A: fewer than 20 samples' if len(sub)<20 else 'full timeline attribution'})
    trade_table=pd.DataFrame(rows);trade_table.to_csv(OUT/'trading_by_state.csv',index=False);display(trade_table.fillna('N/A'))
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    a=pred_table.query("period=='common_dates_posthoc' and task=='classification'");b=pred_table.query("period=='common_dates_posthoc' and task=='regression'");c=trade_table.query("period=='common_dates_posthoc'")
    a.pivot(index='model',columns='state',values='roc_auc').plot.bar(ax=axes[0],title='Classification AUC');axes[0].axhline(.5,color='black',ls='--')
    b.pivot(index='model',columns='state',values='rmse_bp').plot.bar(ax=axes[1],title='Regression RMSE (bps)')
    c.pivot(index='strategy',columns='state',values='total_return').plot.bar(ax=axes[2],title='State return attribution')
    fig.tight_layout();fig.savefig(OUT/'performance_by_state.png');display(fig);plt.close(fig)
    write_json(OUT/'verification.json',{'cutoff':cutoff,'fit_before_all_validation':True,'state_reload_exact':True,'common_n':len(common),'original_and_common_separate':True,'noncontinuous_dates_not_compacted':True})
    write_json(ROOT/'output/fusion/regime_latest.json',{'path':str(OUT.relative_to(ROOT))})
    print('Saved:',OUT.relative_to(ROOT))


    if os.environ.get("SP500_INCLUDE_LITERATURE") == "1":
        SECTION='09_regime'
        assert (ROOT/'01_data_preprocessing.ipynb').is_file()
        LIT_OUT = ROOT/'output/literature'/SECTION/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        LIT_OUT.mkdir(parents=True)
        trade_dir=ROOT/json.loads((ROOT/'output/fusion/trading_latest.json').read_text())['path'];path=trade_dir/'trading_metrics.csv';INPUTS=[path];f=pd.read_csv(path);rows=[]
        for _,r in f[f.strategy.isin(['Buy hold','DDQN vote'])].iterrows():
            rows.append({'origin':'local full-test context','strategy':r.strategy,'Sharpe':r.sharpe,'max_drawdown':r.max_drawdown,'period':str(r.first_signal)+' to '+str(r.last_signal),'risk_free':'0','cost':'5bps one-way','state_method':'KMeans diagnostic; no regime timing strategy'})
        for name,sh,mdd in [('S&P500',-.174,.341),('HMM factor rotation',2.017,.1283)]:
            rows.append({'origin':'paper reported Table4','strategy':name,'Sharpe':sh,'max_drawdown':mdd,'period':'2017-09 to 2020-04','risk_free':'10-year US Treasury yield','cost':'NR','state_method':'HMM daily refit; factor rotation'})
        PAPER={'authors':'Matthew Wang; Yi-Hong Lin; Ilya Mikhelson','year':2020,'doi':'10.3390/jrfm13120311','url':'https://mdpi-res.com/d_attachment/jrfm/jrfm-13-00311/article_deploy/jrfm-13-00311.pdf','location':'Table4 PDF p11; Table7 PDF p12; section4.1','not_reproduced':True,'transaction_cost':'not established from inspected text'}
        literature_table=pd.DataFrame(rows)

        literature_table.to_csv(LIT_OUT/'literature_comparison.csv',index=False,encoding='utf-8-sig')
        (LIT_OUT/'extraction.json').write_text(json.dumps({'checked_on':'2026-11-24','paper':PAPER,'table':literature_table.replace({np.nan:None}).to_dict('records'),'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}},ensure_ascii=False,indent=2),encoding='utf-8')
        display(literature_table.fillna('N/A'))
        print('Saved:',LIT_OUT.relative_to(ROOT))

        regime_literature=pd.DataFrame({'paper_state':['Regime 0','Regime 1','Regime 2'],'SP500_unannualized_conditional_Sharpe':[.03488,.06872,-.02504],'local_state_mapping':['not established']*3})
        regime_literature.to_csv(LIT_OUT/'paper_state_sharpe.csv',index=False)
        display(regime_literature)


STAGES = {"08": trading, "09": regime}

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["08", "09", "all"])
    args = parser.parse_args()
    for number in (STAGES if args.stage == "all" else [args.stage]):
        STAGES[number]()


if __name__ == "__main__":
    main()
