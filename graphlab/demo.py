import random
import math
from .core import Network, fit, load


def fixture():
    r=random.Random(41)
    labels=[i//48 for i in range(96)]
    features=[[r.gauss((2*y-1)*.25,1.) for _ in range(4)] for y in labels]
    edges=[[i,j] for i in range(96) for j in range(i+1,96)
           if r.random() < (.2 if labels[i]==labels[j] else .015)]
    train,val,test=[],[],[]
    for cls in range(2):
        nodes=list(range(cls*48,(cls+1)*48));r.shuffle(nodes)
        train+=nodes[:8];val+=nodes[8:16];test+=nodes[16:]
    perm=list(range(96));random.Random(47).shuffle(perm)
    scrambled=[[perm[i],perm[j]] for i,j in edges]
    return features,edges,scrambled,labels,train,val,test


def evaluate(net,x,e,y,indices):
    p=net.predict(x,e)
    loss,_=net.loss_grad(x,e,y,indices)
    return {'accuracy':sum(p['classes'][i]==y[i] for i in indices)/len(indices),'cross_entropy':loss}


def expressivity():
    x=[[1.,.5]]*6
    cycle=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
    triangles=[[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]]
    net=Network(2,4)
    a,b=net.predict(x,cycle)['probabilities'],net.predict(x,triangles)['probabilities']
    return {'cycle_probabilities':a,'two_triangles_probabilities':b,
            'max_probability_difference':max(abs(a[i][j]-b[i][j]) for i in range(6) for j in range(2)),
            'reason':'constant identical features and degree-2 regular graphs stay identical at every normalized message-passing layer'}


def centroid_control(x,y,train,test):
    centers=[[sum(x[i][j] for i in train if y[i]==c)/sum(y[i]==c for i in train)
              for j in range(len(x[0]))] for c in range(2)]
    probs=[]
    for row in x:
        logits=[-sum((v-m)**2 for v,m in zip(row,mean)) for mean in centers]
        e=[math.exp(z-max(logits)) for z in logits]
        probs.append([z/sum(e) for z in e])
    return {'test_accuracy':sum(max(range(2),key=lambda c:probs[i][c])==y[i] for i in test)/len(test),
            'centers':centers,'fitted_numbers':8,'protocol':'class centroids from 16 training nodes only; no validation selection'}


def demo():
    x,e,scr,y,tr,va,te=fixture()
    arms={}
    for name,edges,mode in [('gcn',e,'gcn'),('feature_mlp',e,'mlp'),('scrambled_gcn',scr,'gcn')]:
        # Test labels are removed before fitting; validation labels select checkpoints only.
        visible=[label if i in tr+va else None for i,label in enumerate(y)]
        net,report=fit(x,edges,visible,tr,mode=mode,validation_indices=va)
        arms[name]={'train':evaluate(net,x,edges,y,tr),'validation':evaluate(net,x,edges,y,va),
                    'test':evaluate(net,x,edges,y,te),'training':report,
                    'final_test':evaluate(load(report['final_artifact']),x,edges,y,te),
                    'parameters':len(net.params),'prediction':net.predict(x,edges)}
    r={'fixture':{'seed':41,'scramble_seed':47,'nodes':96,'features':4,'edges':len(e),
                  'train':tr,'validation':va,'test':te,'updates':300,'learning_rate':.03,'network_seed':13,
                  'within_edge_probability':.2,'between_edge_probability':.015,'feature_class_mean':.25},
       'arms':arms,'centroid_control':centroid_control(x,y,tr,te),'expressivity':expressivity(),'graph':{'features':x,'edges':e,'labels':y}}
    assert arms['gcn']['test']['accuracy'] > arms['feature_mlp']['test']['accuracy']
    assert r['expressivity']['max_probability_difference']==0.
    return r
