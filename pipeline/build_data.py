# build_data.py
# Run from the folder that has model_data.csv and race_results.csv (your "files" folder).
#   python build_data.py
# It retrains with the fixed team-form features, walks forward 2018-2024,
# and writes data.json: per-driver 2024 probabilities, season accuracy, feature importance.
# That JSON is what the web page embeds as `const D=...`.
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DIR = Path(__file__).resolve().parent
d=pd.read_csv(DIR / 'model_data.csv').sort_values(['season','round']).reset_index(drop=True)
rr=pd.read_csv(DIR / 'race_results.csv')[['season','round','raceName']].drop_duplicates()
d['race_id']=d.season*100+d['round']
pt={1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
d['pts']=d.positionOrder.map(pt).fillna(0)
t=d.groupby(['constructorId','race_id']).agg(f=('positionOrder','mean'),p=('pts','mean')).reset_index().sort_values('race_id')
t['tf']=t.groupby('constructorId').f.transform(lambda s:s.shift(1).rolling(3,min_periods=1).mean())
t['tp']=t.groupby('constructorId').p.transform(lambda s:s.shift(1).rolling(3,min_periods=1).mean())
d=d.merge(t[['constructorId','race_id','tf','tp']],on=['constructorId','race_id'])
d['tf']=d.tf.fillna(10.5);d['tp']=d.tp.fillna(5)
F=["grid","qualifying_position","quali_gap_to_pole","driver_avg_finish_last5","driver_avg_points_last5","tf","tp","driver_circuit_avg_finish"]
mk=lambda:RandomForestClassifier(300,max_depth=6,min_samples_leaf=5,random_state=42)
wf=[]
for s in range(2018,2025):
    tr=d[d.season<s];te=d[d.season==s];m=mk().fit(tr[F],tr.points_finish);q=m.predict_proba(te[F])[:,1]
    g=te.grid.replace(0,21)
    wf.append(dict(s=s,m=round(accuracy_score(te.points_finish,q>.5)*100,1),b=round(accuracy_score(te.points_finish,g<=10)*100,1),auc=round(roc_auc_score(te.points_finish,q),3)))
    if s==2024: te=te.assign(p=q);mm=m
te=te.merge(rr,on=['season','round'])
nm=lambda x:' '.join(w.capitalize() for w in x.split('_'))
races=[]
for r,g in te.groupby('round'):
    g=g.assign(o=g.grid.replace(0,99)).sort_values('o')
    races.append(dict(r=int(r),n=g.raceName.iloc[0].replace(' Grand Prix',' GP'),rows=[dict(d=nm(x.driverId),t=x.constructorId,g=int(x.grid),p=round(float(x.p),3),f=int(x.positionOrder)) for x in g.itertuples()]))
imp=sorted(zip(["Grid","Quali position","Quali gap to pole","Driver form (5)","Driver points (5)","Team form (3 races)","Team points (3 races)","Circuit history"],mm.feature_importances_.round(3).tolist()),key=lambda z:-z[1])
y=wf[-1]
out=dict(races=races,wf=wf,imp=imp,auc=y['auc'],n=int(len(te)))
json.dump(out,open(DIR / 'data.json','w'),separators=(',',':'))
print(len(json.dumps(out)),wf,imp[:3])
