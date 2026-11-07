import copy
import json
import math
import unittest
from graphlab.core import Network, operator, propagate, fit, artifact, load
from graphlab.demo import expressivity, centroid_control

X=[[.2,1.],[-.7,.4],[.9,-.3],[1.1,.8]]
E=[[0,1],[1,2]]
Y=[0,1,1,0]


class GraphTests(unittest.TestCase):
    def test_independent_normalization(self):
        op=operator(3,[[0,1],[1,2]])
        dense=[[0.]*3 for _ in range(3)]
        for i,row in enumerate(op):
            for j,w in row:dense[i][j]=w
        expected=[[.5,1/math.sqrt(6),0.],[1/math.sqrt(6),1/3,1/math.sqrt(6)],[0.,1/math.sqrt(6),.5]]
        for a,b in zip(dense,expected):
            for u,v in zip(a,b):self.assertAlmostEqual(u,v)
        self.assertNotAlmostEqual(sum(dense[0]),1.)

    def test_isolated_self_loops(self):
        self.assertEqual(propagate(operator(2,[]),[[2.,3.],[4.,5.]]),[[2.,3.],[4.,5.]])

    def test_undirected_order(self):
        self.assertEqual(operator(3,[[0,1],[1,2]]),operator(3,[[2,1],[1,0]]))

    def test_invalid_edges(self):
        for e in [[[0,0]],[[0,1],[1,0]],[[0,3]],[[True,1]],[[0]],'wrong']:
            with self.assertRaises(ValueError):operator(3,e)

    def test_features_boundaries(self):
        for x in [[],[[]],[[1.],[1.,2.]],[[float('nan')]],[[True]],[[101.]]]:
            with self.assertRaises(ValueError):Network(2).predict(x,[])

    def test_every_gradient_both_modes(self):
        for mode in ['gcn','mlp']:
            n=Network(2,3,2,mode,5)
            _,cache=n.forward(X,E)
            self.assertGreater(min(abs(v) for row in cache[2] for v in row),1e-5)
            _,g=n.loss_grad(X,E,Y,[0,2])
            for j in range(len(n.params)):
                old=n.params[j]
                n.params[j]=old+1e-6;plus=n.loss_grad(X,E,Y,[0,2])[0]
                n.params[j]=old-1e-6;minus=n.loss_grad(X,E,Y,[0,2])[0]
                n.params[j]=old
                self.assertAlmostEqual(g[j],(plus-minus)/2e-6,places=8)

    def test_sparse_dense_forward_reference(self):
        n=Network(2,3,2,seed=5)
        p,cache=n.forward(X,E)
        op=operator(4,E);s=[[0.]*4 for _ in range(4)]
        for i,row in enumerate(op):
            for j,w in row:s[i][j]=w
        w0=n.params[:6];w1=n.params[6:]
        h=[[max(0.,sum(s[i][k]*X[k][d]*w0[d*3+j] for k in range(4) for d in range(2))) for j in range(3)] for i in range(4)]
        logits=[[sum(s[i][k]*h[k][d]*w1[d*2+j] for k in range(4) for d in range(3)) for j in range(2)] for i in range(4)]
        for a,b in zip(cache[-1],logits):
            for u,v in zip(a,b):self.assertAlmostEqual(u,v,places=12)

    def test_permutation_equivariance(self):
        n=Network(2,3);perm=[2,0,3,1]
        xp=[None]*4
        for old,new in enumerate(perm):xp[new]=X[old]
        ep=[[perm[a],perm[b]] for a,b in E]
        a=n.predict(X,E)['probabilities'];b=n.predict(xp,ep)['probabilities']
        for old,new in enumerate(perm):
            for u,v in zip(a[old],b[new]):self.assertAlmostEqual(u,v,places=12)

    def test_mlp_ignores_valid_edges(self):
        n=Network(2,3,mode='mlp')
        self.assertEqual(n.predict(X,E),n.predict(X,[]))

    def test_constant_regular_graph_counterexample(self):
        self.assertEqual(expressivity()['max_probability_difference'],0.)

    def test_holdout_labels_do_not_affect_gradients(self):
        n=Network(2,3)
        self.assertEqual(n.loss_grad(X,E,[0,0,1,0],[0,2]),n.loss_grad(X,E,[0,None,1,None],[0,2]))

    def test_holdout_labels_do_not_affect_fit(self):
        a,_=fit(X,E,[0,0,1,0],[0,2],updates=4)
        b,_=fit(X,E,[0,1,1,None],[0,2],updates=4)
        self.assertEqual(a.params,b.params)

    def test_permuted_training(self):
        perm=[2,0,3,1];xp=[None]*4;yp=[None]*4
        for old,new in enumerate(perm):xp[new],yp[new]=X[old],Y[old]
        ep=[[perm[a],perm[b]] for a,b in E]
        a,_=fit(X,E,Y,[0,2],updates=3)
        b,_=fit(xp,ep,yp,[perm[0],perm[2]],updates=3)
        for u,v in zip(a.params,b.params):self.assertAlmostEqual(u,v,places=10)

    def test_round_trip_and_no_mutation(self):
        x,e,y=copy.deepcopy(X),copy.deepcopy(E),list(Y)
        n,report=fit(x,e,y,[0,2],updates=4)
        self.assertEqual((x,e,y),(X,E,Y))
        a=artifact(n);b=load(json.loads(json.dumps(a)))
        self.assertEqual(n.predict(X,E),b.predict(X,E))
        a['parameters'][0]=999
        self.assertNotEqual(n.params[0],999)

    def test_validation_selection_matches_prefix_refits(self):
        net,report=fit(X,E,Y,[0,2],updates=7,validation_indices=[1,3])
        candidates=[Network(2)]
        for k in range(1,8):candidates.append(fit(X,E,Y,[0,2],updates=k)[0])
        losses=[n.loss_grad(X,E,Y,[1,3])[0] for n in candidates]
        best=min(range(8),key=lambda i:losses[i])
        self.assertEqual(report['selected_update'],best)
        self.assertEqual(net.params,candidates[best].params)
        self.assertEqual(load(report['final_artifact']).params,candidates[-1].params)

    def test_validation_masks_reject_overlap_and_missing_labels(self):
        for val,labels in [([0,1],Y),([1,3],[0,None,1,None]),([],Y)]:
            with self.assertRaises(ValueError):fit(X,E,labels,[0,2],updates=1,validation_indices=val)

    def test_mask_and_label_validation(self):
        for labels,indices in [(Y,[]),(Y,[True]),(Y,[0,0]),(Y,[4]),([None]*4,[0]),([0,1],[0])]:
            with self.assertRaises(ValueError):Network(2).loss_grad(X,E,labels,indices)

    def test_strict_artifact(self):
        base=artifact(Network(2));variants=[]
        for key in base:
            d=copy.deepcopy(base);del d[key];variants.append(d)
        d=copy.deepcopy(base);d['extra']=0;variants.append(d)
        d=copy.deepcopy(base);d['parameters'][0]=float('inf');variants.append(d)
        d=copy.deepcopy(base);d['parameters'].pop();variants.append(d)
        d=copy.deepcopy(base);d['mode']='other';variants.append(d)
        for d in variants:
            with self.assertRaises(ValueError):load(d)

    def test_training_parameter_validation(self):
        for kw in [{'updates':0},{'updates':True},{'hidden':33},{'learning_rate':0},{'classes':1},{'seed':-1}]:
            with self.assertRaises(ValueError):fit(X,E,Y,[0],**kw)

    def test_centroid_uses_only_training_labels(self):
        a=centroid_control(X,Y,[0,2],[1,3])
        b=centroid_control(X,[0,0,1,1],[0,2],[1,3])
        self.assertEqual(a['centers'],b['centers'])

    def test_probabilities_sum_and_stable_cross_entropy(self):
        n=Network(2,1,mode='mlp');n.params=[10.,10.,100.,-100.]
        p=n.predict([[1.,1.]],[])['probabilities'][0]
        self.assertEqual(p,[1.,0.])
        loss,_=n.loss_grad([[1.,1.]],[],[1],[0])
        self.assertEqual(loss,4000.)
