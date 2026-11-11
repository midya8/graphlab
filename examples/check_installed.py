"""Build wheel, import outside checkout, replay all nine CLI commands."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args,**kw):
 p=subprocess.run(args,text=True,capture_output=True,**kw)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p.stdout
with tempfile.TemporaryDirectory()as t:
 p=Path(t);run([sys.executable,'-m','venv',str(p/'venv')]);py=str(p/'venv/bin/python');exe=str(p/'venv/bin/graphlab')
 run([sys.executable,'-m','pip','wheel','--no-deps','-w',str(p/'dist'),str(ROOT)])
 run([py,'-m','pip','install','--no-deps',str(next((p/'dist').glob('*.whl')))],cwd=t)
 env=dict(os.environ);env.pop('PYTHONPATH',None)
 def cli(args,module=True):return json.loads(run(([py,'-m','graphlab']if module else[exe])+args,cwd=t,env=env))
 origins=json.loads(run([py,'-c',"import json,graphlab,graphlab.core,graphlab.demo,graphlab.attribution,graphlab.edge_audit,graphlab.joint,graphlab.cli; print(json.dumps([m.__file__ for m in [graphlab,graphlab.core,graphlab.demo,graphlab.attribution,graphlab.edge_audit,graphlab.joint,graphlab.cli]]))"],cwd=t,env=env))
 assert len(origins)==7 and all('site-packages'in s and str(ROOT)not in s for s in origins)
 count=0
 for command in ['demo','attribution-demo','edge-demo','joint-demo']:
  a=cli([command]);b=cli([command],False);assert a==b;count+=2
 z=cli(['joint-demo']);m=p/'model.json';i=p/'predict.json';m.write_text(json.dumps(z['model']));i.write_text(json.dumps({'features':z['features'],'edges':z['edges']}))
 args=[str(m),str(i),'--node','0','--positive','1','--negative','0']
 for command in ['predict','explain','edges','joint']:
  aa=[str(m),str(i)]if command=='predict'else args
  if command=='joint':aa=args+['--max-forwards','16']
  a=cli([command]+aa);b=cli([command]+aa,False);assert a==b
  if command=='joint':assert a==z['audit']
  count+=2
 train=p/'train.json';train.write_text(json.dumps({'features':[[1.],[2.]],'edges':[[0,1]],'labels':[0,1],'train_indices':[0,1],'classes':2}))
 for module in [True,False]:
  r=cli(['train',str(train),str(p/'trained.json'),'--updates','2'],module);assert r['history']['final_update']==2;count+=1
 print(json.dumps({'module_origins':7,'cli_commands':9,'two_entrypoint_invocations':count,'outside_checkout':True,'wheel':next((p/'dist').glob('*.whl')).name}))
