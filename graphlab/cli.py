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
    t=sub.add_parser('train');t.add_argument('input');t.add_argument('output')
    t.add_argument('--updates',type=int,default=300);t.add_argument('--mode',choices=['gcn','mlp'],default='gcn')
    q=sub.add_parser('predict');q.add_argument('model');q.add_argument('input')
    a=p.parse_args()
    try:
        if a.command=='demo':result=demo()
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
            result=load(read(a.model)).predict(data['features'],data['edges'])
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError) as e:p.exit(2,f'graphlab: {e}\n')


if __name__=='__main__':main()
