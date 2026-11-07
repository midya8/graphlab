import argparse
import json
from .core import fit, load, artifact
from .demo import demo


def read(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description='Offline two-layer GCN audit')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo')
    sub.add_parser('attribution-demo')
    t=sub.add_parser('train');t.add_argument('input');t.add_argument('output')
    t.add_argument('--updates',type=int,default=300);t.add_argument('--mode',choices=['gcn','mlp'],default='gcn')
    q=sub.add_parser('predict');q.add_argument('model');q.add_argument('input')
    explain=sub.add_parser('explain')
    explain.add_argument('model');explain.add_argument('input')
    explain.add_argument('--node',type=int,required=True)
    explain.add_argument('--positive',type=int,required=True)
    explain.add_argument('--negative',type=int,required=True)
    a=p.parse_args()
    try:
        if a.command=='demo':result=demo()
        elif a.command=='attribution-demo':
            from .attribution import attribution_demo
            result=attribution_demo()
        elif a.command=='train':
            data=read(a.input)
            if not isinstance(data,dict) or not {'features','edges','labels','train_indices','classes'} <= set(data) <= {'features','edges','labels','train_indices','classes','validation_indices'}:
                raise ValueError('invalid training request fields')
            net,h=fit(data['features'],data['edges'],data['labels'],data['train_indices'],classes=data['classes'],updates=a.updates,mode=a.mode,validation_indices=data.get('validation_indices'))
            with open(a.output,'w',encoding='utf-8') as f:json.dump(artifact(net),f,allow_nan=False)
            result={'output':a.output,'history':h}
        else:
            data=read(a.input)
            if not isinstance(data,dict) or set(data)!={'features','edges'}:raise ValueError('invalid prediction request fields')
            net=load(read(a.model))
            if a.command=='explain':
                from .attribution import margin_attribution
                result=margin_attribution(net,data['features'],data['edges'],a.node,a.positive,a.negative)
            else:result=net.predict(data['features'],data['edges'])
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError) as e:p.exit(2,f'graphlab: {e}\n')


if __name__=='__main__':main()
