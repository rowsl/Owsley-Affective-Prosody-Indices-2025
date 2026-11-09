#!/usr/bin/env python
# coding: utf-8

# # Corrected final PHQ-8 endpoint × time sensitivity run
# 
# One executable cell. Uses the **complete** 7,220-row episode acoustic source, joins PHQ-8 by participant from the narrow episode lookup, and verifies endpoint counts **4,914 / 5,688 / 6,613** before accepting the episode analysis. Other units remain distinct, with unsupported mappings logged. Writes four timestamped CSV files under `data/final_phq8_sensitivity`. Does not repeat spline bootstrap.
# 
# Run on your Windows Jupyter kernel with the DAIC_WOZ_FRONTIERS data folder present. The notebook has not been executed here because your local dataset is not available in this environment.
# 

# In[1]:


# ONE-CELL FINAL SENSITIVITY RUN — run from the DAIC_WOZ_FRONTIERS project.
# Reuses verified PHQ-8 energy specifications; never relabels speech proportion as energy SD.
from pathlib import Path
import re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices
from datetime import datetime

ROOT=Path(r'C:\Users\rache\Documents\DAIC_WOZ_FRONTIERS')
DATA=ROOT/'data'
OUT=DATA/'final_phq8_sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
STAMP=datetime.now().strftime('%Y%m%d_%H%M%S')

def locate(*paths):
    for rel in paths:
        p=ROOT/rel
        if p.is_file(): return p
    return None

def load(p):
    return pd.read_pickle(p) if p.suffix.lower() in ('.pkl','.pickle') else pd.read_csv(p)

def pid(s):
    return pd.to_numeric(s.astype(str).str.extract(r'(\d{3})',expand=False),errors='coerce').astype('Int64')

def choose(df,*names):
    return next((n for n in names if n in df.columns),None)

specs=[
 ('episode',('data/episode_acoustic_temporal_clinical_integrated.csv',),
  ('participant_id',),('resp_start',),('resp_stop',),('PHQ8_Score',),
  ('energy_sd_db','f0_sd_hz'),('clean_speech_duration',)),
 ('artstein_turn',('data/artstein_model_df.pkl',),
  ('PID_int',),('start_time',),('stop_time',),('PHQ8_Score',),
  ('speech_prop',),('duration_sec',)),
 ('original_turn_timing',('data/temporal_original/turn_level_timing_modeling_ready.csv',),
  ('PID_int','participant'),('start_time',),('stop_time',),('PHQ8_Score','PHQ8_Score_raw'),
  ('speech_prop',),('duration_sec',)),
 ('reconstructed_episode_timing',('data/temporal_revised/episode_temporal_transcript_reconstructed.csv',),
  ('participant_id','PID_int','participant'),('resp_start','start_time'),('resp_stop','stop_time'),
  ('PHQ8_Score','PHQ8_Score_raw'),('speech_prop',),('duration_sec','clean_speech_duration')),
]
loaded={}; audit=[]; results=[]; contrasts=[]; failures=[]
for name,paths,pids,starts,ends,phqs,outcomes,durations in specs:
    path=locate(*paths)
    if path is None:
        failures.append({'dataset':name,'issue':'Source file not found; not substituted'})
        continue
    try:
        raw=load(path)
        if name == 'episode':
            # PHQ is available in the narrow episode file, but the acoustic rows
            # must come from the COMPLETE integrated episode source.
            phq_path=locate('data/episode_primary_clinical_PHQ8_Emotion_VADER.csv')
            if phq_path is None: raise FileNotFoundError('Missing narrow episode PHQ lookup')
            phq_source=load(phq_path)
            if not {'participant_id','PHQ8_Score'}.issubset(phq_source.columns):
                raise KeyError('Narrow episode PHQ lookup lacks participant_id/PHQ8_Score')
            lookup=phq_source[['participant_id','PHQ8_Score']].copy()
            lookup['pid_merge']=pid(lookup['participant_id'])
            lookup['PHQ8_Score']=pd.to_numeric(lookup['PHQ8_Score'],errors='coerce')
            if lookup.groupby('pid_merge')['PHQ8_Score'].nunique(dropna=True).gt(1).any():
                raise ValueError('Conflicting PHQ scores in episode lookup')
            lookup=lookup.dropna(subset=['pid_merge','PHQ8_Score']).drop_duplicates('pid_merge')[['pid_merge','PHQ8_Score']]
            raw=raw.copy()
            raw['pid_merge']=pid(raw['participant_id'])
            raw=raw.drop(columns=['PHQ8_Score'],errors='ignore').merge(lookup,on='pid_merge',how='left',validate='many_to_one')
            raw=raw.drop(columns='pid_merge')
        cols=dict(pid=choose(raw,*pids),start=choose(raw,*starts),end=choose(raw,*ends),
                  phq=choose(raw,*phqs),duration=choose(raw,*durations))
        available=[v for v in outcomes if v in raw.columns]
        if not all(cols.values()) or not available:
            failures.append({'dataset':name,'issue':f'Unsupported source schema: {cols}; outcomes {available}; columns {list(raw.columns)}'})
            continue
        d=raw.rename(columns={v:k for k,v in cols.items()})
        d=d.copy();d['pid']=pid(d.pid)
        for c in ('start','end','phq','duration'):
            d[c]=pd.to_numeric(d[c],errors='coerce')
        d=d.loc[d.pid.notna() & d.start.notna() & d.end.notna() & (d.end>=d.start)].copy()
        if d.empty: raise ValueError('No valid timestamps')
        if d.groupby('pid').phq.nunique(dropna=True).gt(1).any():
            raise ValueError('Conflicting PHQ scores for participant')
        loaded[name]=(d,available,str(path))
        print(f'Loaded {name}: {len(d)} valid rows, {d.pid.nunique()} participants, {d.phq.notna().sum()} rows with PHQ')
    except Exception as exc:
        failures.append({'dataset':name,'issue':str(exc)})

boundary_path=locate('data/multiple_interview_boundaries.csv')
if boundary_path is None: raise FileNotFoundError('Missing data/multiple_interview_boundaries.csv')
b=load(boundary_path)
bpid=choose(b,'participant_id','PID_int','PID')
if not bpid: raise KeyError('Boundary participant ID column not found')
b=b.copy();b['pid']=pid(b[bpid])
if b.pid.duplicated().any(): raise ValueError('Duplicate boundary IDs')
windows={'narrow':'narrow_end_time','broad':'broad_end_time','full':'full_end_time'}
if 'narrow_end_time' not in b and 'primary_end_time' in b: windows['narrow']='primary_end_time'
for w,c in windows.items():
    if c not in b: raise KeyError(f'Missing {w} endpoint column: {c}')
    b[c]=pd.to_numeric(b[c],errors='coerce')

# Common origin: earliest observed start for each participant across loaded datasets.
# This is an operational anchor, NOT a verified recording start; recorded explicitly.
# IMPORTANT: episode rows come from the COMPLETE file, not the narrow subset.
if not loaded: raise RuntimeError('No usable dataset; inspect failure report below')
starts=pd.concat([d[['pid','start']] for d,_,_ in loaded.values()],ignore_index=True)
anchors=starts.groupby('pid',as_index=False).start.min().rename(columns={'start':'anchor'})

for unit,(raw,outcomes,path) in loaded.items():
    base=raw.merge(anchors,on='pid',validate='many_to_one').merge(b[['pid']+list(windows.values())],on='pid',validate='many_to_one')
    for w,endcol in windows.items():
        end=base[endcol]
        inside=(base.start>=base.anchor)&(base.end<=end)&(end>base.anchor)
        a=base.loc[inside].sort_values(['pid','start','end']).copy()
        # Fail loudly if a purported broad/full episode window contains no new rows.
        if unit == 'episode':
            expected={'narrow':4914,'broad':5688,'full':6613}[w]
            if len(a) != expected:
                raise ValueError(f'Episode {w}: got {len(a)} rows, expected {expected}. Check timestamps, boundaries and source; no misleading CSV will be accepted.')
        a['t_elapsed']=(a.start-a.anchor)/(a[endcol]-a.anchor)
        a['ord']=a.groupby('pid').cumcount()
        a['n_ord']=a.groupby('pid').pid.transform('size')
        a['t_ordinal']=np.where(a.n_ord>1,a.ord/(a.n_ord-1),.5)
        for clock in ('elapsed','ordinal'):
            for outcome in outcomes:
                z=a[['pid','phq','duration','t_'+clock,outcome]].copy().rename(columns={'t_'+clock:'time',outcome:'y'})
                z=z.replace([np.inf,-np.inf],np.nan).dropna()
                z=z.loc[(z.duration>0)&z.phq.between(0,24)&z.time.between(0,1)].copy()
                rec={'dataset':unit,'outcome':outcome,'endpoint':w,'time_method':clock,
                     'source':path,'raw_rows':len(raw),'inside_rows':len(a),'usable_rows':len(z),
                     'participants':z.pid.nunique(),'crossing_rows':int(((base.start<end)&(base.end>end)).sum()),
                     'elapsed_anchor':'earliest available observed start across loaded datasets'}
                audit.append(rec)
                if len(z)<100 or z.pid.nunique()<20:
                    failures.append({'dataset':unit,'endpoint':w,'time_method':clock,'outcome':outcome,'issue':'Insufficient usable data'})
                    continue
                z['time_mid']=z.time-.5
                z['log_duration']=np.log(z.duration)
                z['PHQc']=z.phq-11.
                z['PHQc2']=z.PHQc**2
                z['phq_sq']=z.phq**2
                z['group']=pd.Categorical(pd.cut(z.phq,[-np.inf,4,9,14,np.inf],labels=['Lowest','Mild','Moderate','Higher']),categories=['Lowest','Mild','Moderate','Higher'],ordered=True)
                forms={
                  'linear_severity_time':'y ~ PHQc*time_mid + log_duration',
                  'quadratic_severity':'y ~ PHQc + PHQc2 + time_mid + log_duration',
                  'quadratic_severity_time':'y ~ (PHQc + PHQc2)*time_mid + log_duration',
                  'moderate_groups':'y ~ C(group, Treatment(reference="Lowest"))*time_mid + log_duration',
                }
                for label,formula in forms.items():
                    try:
                        fit=smf.ols(formula,z).fit(cov_type='cluster',cov_kwds={'groups':z.pid},use_t=True)
                        if unit == 'episode' and w == 'narrow' and clock == 'elapsed' and outcome == 'energy_sd_db' and label == 'quadratic_severity':
                            figure_fit = fit
                            figure_data = z.copy()
                        ci=fit.conf_int()
                        for term in fit.params.index:
                            results.append({**rec,'model':label,'term':term,'estimate':float(fit.params[term]),
                                            'se':float(fit.bse[term]),'p':float(fit.pvalues[term]),
                                            'ci_low':float(ci.loc[term].iloc[0]),'ci_high':float(ci.loc[term].iloc[1])})
                        if label=='moderate_groups':
                            def xrow(g):
                                return pd.DataFrame({'group':pd.Categorical([g],categories=z.group.cat.categories,ordered=True),
                                                     'time_mid':[0.],'log_duration':[z.log_duration.median()]})
                            di=fit.model.data.design_info
                            xmod=np.asarray(build_design_matrices([di],xrow('Moderate'))[0])[0]
                            sub=[]
                            for other in ('Lowest','Mild','Higher'):
                                xx=np.asarray(build_design_matrices([di],xrow(other))[0])[0]
                                test=fit.t_test(xmod-xx)
                                ci2=np.asarray(test.conf_int()).reshape(-1,2)[0]
                                sub.append({**rec,'contrast':'Moderate minus '+other,
                                            'estimate':float(np.asarray(test.effect).item()),
                                            'p':float(np.asarray(test.pvalue).item()),
                                            'ci_low':float(ci2[0]),'ci_high':float(ci2[1])})
                            adj=multipletests([x['p'] for x in sub],method='holm')[1]
                            for item,ap in zip(sub,adj): item['holm_p_within_configuration']=float(ap)
                            contrasts.extend(sub)
                    except Exception as exc:
                        failures.append({'dataset':unit,'endpoint':w,'time_method':clock,'outcome':outcome,'model':label,'issue':str(exc)})

# Always save audits, including missing/unsupported configurations. Never silently claim 4 x 3 x 2 coverage.
for filename,rows in [('model_coefficients',results),('sample_audit',audit),('moderate_contrasts',contrasts),('failures_and_gaps',failures)]:
    frame=pd.DataFrame(rows)
    path=OUT/f'{filename}_{STAMP}.csv'
    frame.to_csv(path,index=False)
    print(f'{filename}: {len(frame)} rows -> {path}')
print('\nDATASETS LOADED:',list(loaded))
print('Models run:',len(results),'coefficient rows; inspect failures_and_gaps for incomplete coverage.')
print('NOTE: Original flexible-spline/1200-bootstrap findings are preserved in their source notebooks;')
print('this batch does not claim to rerun those expensive procedures or prove four-way acoustic comparability.')


# In[2]:


# Publication figure: adjusted quadratic prediction with participant-clustered 95% confidence band
# Run the corrected analysis cell above first. This figure uses the SAME fitted model.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import build_design_matrices

if 'figure_fit' not in globals() or 'figure_data' not in globals():
    raise RuntimeError('Run the preceding corrected sensitivity-analysis cell first; figure_fit was not saved.')
fit = figure_fit
z = figure_data
# Limit the display to PHQ-8 scores observed in this model's analytic sample.
scores = np.linspace(z.phq.min(), z.phq.max(), 250)
ref_log_duration = float(z.log_duration.median())
pred = pd.DataFrame({'PHQc':scores-11., 'PHQc2':(scores-11.)**2,
                     'time_mid':np.zeros(len(scores)),
                     'log_duration':np.full(len(scores),ref_log_duration)})
X = np.asarray(build_design_matrices([fit.model.data.design_info], pred)[0])
mean = X @ fit.params.to_numpy()
V = fit.cov_params().to_numpy()
se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,V,X),0))
# statsmodels used use_t=True: use the fitted model's cluster-robust t critical value.
from scipy.stats import t
crit = t.ppf(.975,df=float(fit.df_resid_inference)) if hasattr(fit,'df_resid_inference') else 1.96
low, high = mean-crit*se,mean+crit*se
fig,ax=plt.subplots(figsize=(8.4,5.4),dpi=180)
ax.fill_between(scores,low,high,alpha=.18,label='95% confidence band')
ax.plot(scores,mean,linewidth=2.4,label='Adjusted quadratic prediction')
for boundary in [5,10,15]:
    if scores.min()<boundary<scores.max(): ax.axvline(boundary,color='gray',linewidth=.8,linestyle=':',alpha=.6)
ax.set(xlabel='PHQ-8 symptom severity',ylabel='Predicted within-episode energy variability (dB)',
       title='Depressive severity and acoustic energy variability')
# Participant-level rug: one mark per participant, not one per episode.
# Scores are discrete, so identical values overlap (the rug is not a frequency plot).
rug = z[['pid', 'phq']].drop_duplicates(subset='pid').dropna(subset=['phq'])
ax.plot(rug['phq'].to_numpy(),
        np.full(len(rug), -0.025), '|',
        color='black', markersize=7, alpha=.35,
        transform=ax.get_xaxis_transform(), clip_on=False,
        label='_nolegend_')
ax.set_xlim(scores.min(),scores.max())
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False)
fig.text(.12,.015,'Narrow clinical endpoint; elapsed time at midpoint; log duration at analytic-sample median.',fontsize=8)
fig.tight_layout(rect=(0,.045,1,1))
out=Path(OUT)
out.mkdir(parents=True,exist_ok=True)
for extension in ('png','svg','pdf'):
    fig.savefig(out/f'figure_quadratic_energy_narrow_elapsed.{extension}',dpi=300,bbox_inches='tight')
plt.show()
print('Saved figure files to:',out)
print('Rug participants:',len(rug))
print('Analytic PHQ-8 range:',z.phq.min(), 'to',z.phq.max())
print('Quadratic coefficient:',fit.params['PHQc2'])
print('Estimated quadratic vertex PHQ-8:',11 - fit.params['PHQc']/(2*fit.params['PHQc2']))


# In[ ]:




