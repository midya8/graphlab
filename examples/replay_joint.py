"""Independently replay a browser census with the native sparse evaluator."""
import json,math,sys
from graphlab.core import load
from graphlab.joint import joint_sensitivity

def compare(a,b):
    if type(b)is int:
        assert type(a)is int and a==b;return
    if type(a)in(int,float)and type(b)in(int,float):
        assert math.isfinite(a)and math.isfinite(b)and abs(a-b)<=2e-11*max(1,abs(a),abs(b));return
    assert type(a)is type(b),(type(a),type(b))
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for v,w in zip(a,b):compare(v,w)
    else:assert a==b
if __name__=='__main__':
    z=json.load(open(sys.argv[1]));v=z['input'];r=joint_sensitivity(load(v['model']),v['features'],v['edges'],v['node'],v['positive'],v['negative'],v['max_deletions'],4096);compare(z['report'],r);print('native sparse replay accepted')
