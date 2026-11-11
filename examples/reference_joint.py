"""Actual NumPy dense matrix reference, not a call to sparse model forward."""
import json,itertools
from pathlib import Path
import numpy as np
from graphlab.core import load
from graphlab.joint import joint_sensitivity
root=Path(__file__).resolve().parents[1];cases=[]
for f in sorted((root/'tests').glob('joint-reference-*.json')):cases+=json.loads(f.read_text())
count=0;largest=0.
for case in cases:
 v=case['input'];m=v['model'];x=np.asarray(v['features'],dtype=np.float64);n=len(x);d,h,c=m['inputs'],m['hidden'],m['classes'];p=np.asarray(m['parameters']);w0=p[:d*h].reshape(d,h);w1=p[d*h:].reshape(h,c)
 for row in case['report']['rows']:
  a=np.eye(n)
  if m['mode']=='gcn':
   for i,(u,w)in enumerate(v['edges']):
    if i not in row['deleted_indices']:a[u,w]=a[w,u]=1.
  degree=a.sum(axis=1);s=a/np.sqrt(degree[:,None]*degree[None,:])
  z=(s@np.maximum((s@x)@w0,0.)@w1)[v['node']]
  ref=np.asarray(row['target_logits']);error=float(np.max(np.abs(ref-z)));largest=max(largest,error);assert np.allclose(ref,z,rtol=2e-11,atol=2e-11)
  # Discrete class checks only away from a numerical tie, or exact all-zero ties.
  gap=np.sort(z)[-1]-np.sort(z)[-2]
  if gap>1e-10 or np.all(z==0):assert int(np.argmax(z))==row['predicted_class']
  count+=1
print(json.dumps({'reference':'NumPy '+np.__version__,'complete_cases':len(cases),'dense_forwards':count,'maximum_absolute_logit_difference':largest,'scope':'finite numerical reference, no exact tie or real-arithmetic proof'}))
