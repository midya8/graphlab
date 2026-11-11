import copy
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal,localcontext
from graphlab import Network,joint_sensitivity
from graphlab.core import artifact
from graphlab.joint import joint_demo


def dense(net,x,e,node,decimal=False):
    # Full adjacency, matrix products and bitmask enumeration are separate from
    # sparse production propagation and cardinality combination generation.
    D=(lambda v:Decimal(str(v))) if decimal else float
    n=len(x);a=[[D(int(i==j)) for j in range(n)] for i in range(n)]
    if net.mode=='gcn':
        for i,j in e:a[i][j]=a[j][i]=D(1)
    degree=[sum(r) for r in a]
    s=[[a[i][j]/((degree[i]*degree[j]).sqrt() if decimal else math.sqrt(degree[i]*degree[j])) for j in range(n)]for i in range(n)]
    def mul(a,b):return [[sum(a[i][k]*b[k][j]for k in range(len(b)))for j in range(len(b[0]))]for i in range(len(a))]
    d,h,c=net.inputs,net.hidden,net.classes;p=list(map(D,net.params))
    w0=[p[i*h:(i+1)*h]for i in range(d)];w1=[p[d*h+i*c:d*h+(i+1)*c]for i in range(h)]
    pre=mul(mul(s,[[D(v)for v in r]for r in x]),w0)
    return mul(mul(s,[[max(D(0),v)for v in r]for r in pre]),w1)[node]


class JointTests(unittest.TestCase):
    def test_all_subsets_dense_independent_reference(self):
        rnd=random.Random(2911);total=0
        for case in range(180):
            n=2+case%4;d=1+case%3;h=1+case%4;c=2+case%3
            net=Network(d,h,c,mode='mlp'if case%7==0 else'gcn',seed=case)
            x=[[rnd.uniform(-3,3)for _ in range(d)]for _ in range(n)]
            e=[[i,j]for i in range(n)for j in range(i+1,n)if rnd.random()<.6][:8]
            k=case%4;target=case%n
            r=joint_sensitivity(net,x,e,target,1,0,k,1024)
            expected={tuple(i for i in range(len(e))if mask>>i&1)for mask in range(1<<len(e))if mask.bit_count()<=k}
            self.assertEqual({tuple(z['deleted_indices'])for z in r['rows']},expected)
            self.assertEqual(len(r['rows']),len(expected));total+=len(expected)
            self.assertEqual([z['deleted_indices']for z in r['rows']],sorted((list(z)for z in expected),key=lambda z:(len(z),z)))
            for z in r['rows']:
                kept=[p for i,p in enumerate(e)if i not in z['deleted_indices']]
                ref=dense(net,x,kept,target)
                for v,w in zip(z['target_logits'],ref):self.assertAlmostEqual(v,w,places=11)
                self.assertEqual(z['predicted_class'],max(range(c),key=lambda j:ref[j]))
                self.assertAlmostEqual(z['margin'],ref[1]-ref[0],places=11)
            for b in r['budgets']:
                ids=[i for i,z in enumerate(r['rows'])if len(z['deleted_indices'])<=b['budget']]
                self.assertEqual(b['minimum_margin_row'],min(ids,key=lambda i:r['rows'][i]['margin']))
                self.assertEqual(b['maximum_margin_row'],max(ids,key=lambda i:r['rows'][i]['margin']))
                self.assertEqual(b['maximum_interaction_row'],max(ids,key=lambda i:abs(r['rows'][i]['interaction_residual'])))
                self.assertEqual(b['changed_predictions'],sum(r['rows'][i]['prediction_changed']for i in ids))
        self.assertGreater(total,1000)

    def test_singletons_miss_pair_and_decimal_check(self):
        z=joint_demo();a=z['audit'];net=Network(1,2);net.params=z['model']['parameters']
        self.assertEqual(a['forwards'],16);self.assertEqual(a['budgets'][1]['changed_predictions'],0)
        pair=next(r for r in a['rows']if r['deleted_indices']==[1,3])
        self.assertLess(pair['margin'],0);self.assertTrue(pair['prediction_changed'])
        self.assertGreater(pair['singleton_sum']+a['baseline_margin'],0)
        self.assertGreater(abs(pair['interaction_residual']),.5)
        with localcontext() as ctx:
            ctx.prec=70
            for r in a['rows']:
                kept=[p for i,p in enumerate(z['edges'])if i not in r['deleted_indices']]
                ref=dense(net,z['features'],kept,0,True)
                self.assertLess(abs(Decimal(str(r['margin']))-(ref[1]-ref[0])),Decimal('3e-15'))

    def test_preflight_no_calls_and_actual_count(self):
        class Count(Network):
            calls=0
            def forward(self,*args):self.calls+=1;return super().forward(*args)
        n=Count(1,2);e=[[i,j]for i in range(8)for j in range(i+1,8)];x=[[1.]for _ in range(8)]
        with self.assertRaisesRegex(ValueError,'needs 4791323 forwards'):joint_sensitivity(n,x,e,0,1,0,8,65536)
        self.assertEqual(n.calls,0)
        r=joint_sensitivity(n,x,[[0,1],[1,2]],0,1,0,2,4);self.assertEqual(n.calls,4);self.assertEqual(r['forwards'],4)

    def test_zero_empty_and_all_edges(self):
        n=Network(1);x=[[1.],[2.]]
        for e,k,count in [([],8,1),([[0,1]],0,1),([[0,1]],8,2)]:
            r=joint_sensitivity(n,x,e,0,1,0,k,2);self.assertEqual(r['forwards'],count);self.assertEqual(r['rows'][0]['deleted_indices'],[])

    def test_mlp_and_multiclass_margin_not_prediction(self):
        n=Network(1,1,3,mode='mlp');n.params=[1.,2.,1.,3.]
        r=joint_sensitivity(n,[[1.],[2.]],[[0,1]],0,1,0)
        self.assertEqual(r['baseline_class'],2);self.assertEqual(r['baseline_margin'],-1.)
        self.assertTrue(all(z['margin_change']==0 and not z['prediction_changed']for z in r['rows']))

    def test_tie_lowest_index_and_no_mutation(self):
        n=Network(1,2);n.params=[0.]*6;x=[[1.],[2.]];e=[[0,1]];before=copy.deepcopy((x,e,n.params))
        r=joint_sensitivity(n,x,e,0,1,0)
        self.assertTrue(all(z['predicted_class']==0 for z in r['rows']))
        self.assertEqual((x,e,n.params),before)

    def test_orientation_permutation_and_margin_reversal(self):
        z=joint_demo();n=Network(1,2);n.params=z['model']['parameters']
        a=z['audit'];b=joint_sensitivity(n,z['features'],[p[::-1]for p in z['edges']],0,0,1)
        for r,s in zip(a['rows'],b['rows']):
            self.assertEqual(r['deleted_edges'],s['deleted_edges']);self.assertAlmostEqual(r['margin'],-s['margin']);self.assertEqual(r['predicted_class'],s['predicted_class'])
        perm=[2,0,3,1];xx=[None]*4
        for i,p in enumerate(perm):xx[p]=z['features'][i]
        c=joint_sensitivity(n,xx,[[perm[i],perm[j]]for i,j in z['edges']],perm[0],1,0)
        for r,s in zip(a['rows'],c['rows']):self.assertAlmostEqual(r['margin'],s['margin'],places=12)

    def test_strict_inputs(self):
        n=Network(1);x=[[1.],[2.]];e=[[0,1]]
        for bad in [True,-1,9,2.5,'2']:
            with self.assertRaises(ValueError):joint_sensitivity(n,x,e,0,1,0,bad)
        for bad in [True,0,65537,1.5]:
            with self.assertRaises(ValueError):joint_sensitivity(n,x,e,0,1,0,2,bad)
        for xx,ee in [([[True],[2]],e),(x,[[0,0]]),(x,[[1,0],[0,1]]),(x,[[0,2]]),([[1,2],[2,3]],e)]:
            with self.assertRaises(ValueError):joint_sensitivity(n,xx,ee,0,1,0)

    def test_cli_replays_and_budget_error(self):
        z=joint_demo()
        with tempfile.TemporaryDirectory() as t:
            from pathlib import Path
            m=Path(t)/'m.json';i=Path(t)/'i.json';m.write_text(json.dumps(z['model']));i.write_text(json.dumps({'features':z['features'],'edges':z['edges']}))
            args=[sys.executable,'-m','graphlab','joint',str(m),str(i),'--node','0','--positive','1','--negative','0','--max-forwards','16']
            p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),z['audit'])
            p=subprocess.run(args[:-1]+['15'],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
