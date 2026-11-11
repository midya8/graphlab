"""Deterministic browser controls and complete native reference reports."""
import json,random,sys
from pathlib import Path
from graphlab.core import Network,artifact
from graphlab.joint import joint_demo,joint_sensitivity
from graphlab.attribution import attribution_demo


def cases():
    z=joint_demo();a=attribution_demo()
    def wrap(title,description,model,x,e,k=2,node=0,pos=1,neg=0):
        return {'title':title,'description':description,'input':{'model':model,'features':x,'edges':e,'max_deletions':k,'node':node,'positive':pos,'negative':neg}}
    net=Network(2,2);net.params=a['parameters']
    mlp=dict(z['model'],mode='mlp');tie=dict(z['model'],parameters=[0.]*6)
    multi=Network(1,2,3);multi.params=[1.,-1.,0.,1.,2.,0.,-1.,2.]
    return [wrap('Every single deletion preserves the class','Hand-set four-node counterexample, found by enumerating integer feature controls. Not a trained result.',z['model'],z['features'],z['edges']),
            wrap('Normalization reaches past the feature gradient','Original hand-set six-node input-gradient path. Edge deletion is a different operation.',artifact(net),a['features'],a['edges'],3),
            wrap('Feature-only MLP ignores edges','Same hand-set weights and inputs as the first control, identity propagation.',mlp,z['features'],z['edges']),
            wrap('All-class prediction is not a chosen margin','Third class can win while the selected two-class margin changes.',artifact(multi),z['features'],z['edges']),
            wrap('Lowest-index ties','All-zero weights produce tied logits; class 0 wins.',tie,z['features'],z['edges']),
            wrap('No original edges','One empty perturbation set, self loops retained.',z['model'],z['features'],[],8)]


def references():
    out=cases();rnd=random.Random(1319)
    for i in range(40):
        nodes=2+i%4;net=Network(1+i%3,1+i%4,2+i%3,mode='mlp'if i%9==0 else'gcn',seed=119+i)
        x=[[rnd.uniform(-2,2)for _ in range(net.inputs)]for _ in range(nodes)]
        e=[[j,k]for j in range(nodes)for k in range(j+1,nodes)if rnd.random()<.55][:7]
        out.append({'title':'seeded-'+str(i),'description':'Finite parity control','input':{'model':artifact(net),'features':x,'edges':e,'node':i%nodes,'positive':1,'negative':0,'max_deletions':i%4}})
    for c in out:
        v=c['input'];c['report']=joint_sensitivity(v['model']and __import__('graphlab.core',fromlist=['load']).load(v['model']),v['features'],v['edges'],v['node'],v['positive'],v['negative'],v['max_deletions'],4096)
    return out


def export(check=False):
    root=Path(__file__).resolve().parents[1];refs=references()
    outputs={'docs/data.js':'globalThis.GraphCases = '+json.dumps(cases(),separators=(',',':'),allow_nan=False)+';\n'}
    for i in range(0,len(refs),8):outputs['tests/joint-reference-'+str(i//8)+'.json']=json.dumps(refs[i:i+8],separators=(',',':'),allow_nan=False)+'\n'
    for f,text in outputs.items():
        p=root/f
        if check:assert p.read_text()==text,'stale '+f
        else:p.write_text(text)
    print(json.dumps({'cases':len(refs),'rows':sum(len(c['report']['rows'])for c in refs),'developed':len(cases())}))
if __name__=='__main__':export('--check'in sys.argv)
