"""Two-layer Kipf-Welling GCN, sparse symmetric propagation, exact gradients."""
import math
import random


def integer(x, lo, hi, name):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError(f'{name} must be integer in [{lo},{hi}]')
    return x


def number(x, name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError(name+' must be finite')
    return float(x)


def matrix(x):
    if not isinstance(x,list) or not 1 <= len(x) <= 128:
        raise ValueError('features require 1..128 nodes')
    if not isinstance(x[0],list) or not 1 <= len(x[0]) <= 16:
        raise ValueError('features require width 1..16')
    width=len(x[0])
    if any(not isinstance(row,list) or len(row)!=width for row in x):
        raise ValueError('ragged features')
    rows=[[number(v,'feature') for v in row] for row in x]
    if any(abs(v)>100 for row in rows for v in row):
        raise ValueError('features outside [-100,100]')
    return rows


def operator(nodes, edges, mode='gcn'):
    integer(nodes,1,128,'nodes')
    if mode not in ('gcn','mlp'):raise ValueError('mode must be gcn or mlp')
    if not isinstance(edges,list) or len(edges)>nodes*(nodes-1)//2:
        raise ValueError('invalid edge list')
    neighbors=[{i} for i in range(nodes)]
    seen=set()
    for pair in edges:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('edges need two indices')
        a,b=[integer(v,0,nodes-1,'edge index') for v in pair]
        key=tuple(sorted([a,b]))
        if a==b or key in seen:raise ValueError('self loops and duplicate edges are not input edges')
        seen.add(key);neighbors[a].add(b);neighbors[b].add(a)
    if mode=='mlp':return [[(i,1.)] for i in range(nodes)]
    degree=[len(row) for row in neighbors]
    return [[(j,1/math.sqrt(degree[i]*degree[j])) for j in sorted(row)] for i,row in enumerate(neighbors)]


def propagate(op, rows):
    return [[sum(w*rows[j][k] for j,w in row) for k in range(len(rows[0]))] for row in op]


def multiply(rows, weights, inputs, outputs):
    return [[sum(x[k]*weights[k*outputs+j] for k in range(inputs)) for j in range(outputs)] for x in rows]


def mask(indices,n):
    if not isinstance(indices,list) or not indices:raise ValueError('need nonempty training indices')
    result=[integer(i,0,n-1,'training index') for i in indices]
    if len(set(result))!=len(result):raise ValueError('duplicate training indices')
    return result


class Network:
    def __init__(self, inputs, hidden=12, classes=2, mode='gcn', seed=13):
        integer(inputs,1,16,'inputs');integer(hidden,1,32,'hidden');integer(classes,2,8,'classes')
        integer(seed,0,2**32-1,'seed')
        if mode not in ('gcn','mlp'):raise ValueError('mode must be gcn or mlp')
        self.inputs,self.hidden,self.classes,self.mode=inputs,hidden,classes,mode
        r=random.Random(seed)
        self.params=[r.gauss(0,math.sqrt(2/inputs)) for _ in range(inputs*hidden)]
        self.params += [r.gauss(0,1/math.sqrt(hidden)) for _ in range(hidden*classes)]

    def forward(self, features, edges):
        x=matrix(features)
        if len(x[0])!=self.inputs:raise ValueError('feature width differs from model')
        op=operator(len(x),edges,self.mode)
        sx=propagate(op,x)
        split=self.inputs*self.hidden
        pre=multiply(sx,self.params[:split],self.inputs,self.hidden)
        h=[[max(0.,v) for v in row] for row in pre]
        sh=propagate(op,h)
        logits=multiply(sh,self.params[split:],self.hidden,self.classes)
        if not all(math.isfinite(v) for row in logits for v in row):raise ValueError('nonfinite logits')
        probs=[]
        for row in logits:
            e=[math.exp(v-max(row)) for v in row];den=sum(e)
            probs.append([v/den for v in e])
        return probs,(op,sx,pre,sh,logits)

    def loss_grad(self, features, edges, labels, indices):
        probs,(op,sx,pre,sh,logits)=self.forward(features,edges)
        n,c,h=len(probs),self.classes,self.hidden
        indices=mask(indices,n)
        if not isinstance(labels,list) or len(labels)!=n:raise ValueError('labels need one entry per node')
        for y in labels:
            if y is not None:integer(y,0,c-1,'label')
        if any(labels[i] is None for i in indices):raise ValueError('training node has no label')
        d=[[0.]*c for _ in range(n)];loss=0.
        for i in indices:
            row=logits[i];m=max(row);y=labels[i]
            loss += (m+math.log(sum(math.exp(v-m) for v in row))-row[y])/len(indices)
            d[i]=[(probs[i][k]-(k==y))/len(indices) for k in range(c)]
        split=self.inputs*h
        g1=[sum(sh[i][j]*d[i][k] for i in range(n)) for j in range(h) for k in range(c)]
        dh=[[sum(d[i][k]*self.params[split+j*c+k] for k in range(c)) for j in range(h)] for i in range(n)]
        # The propagation operator is symmetric, so its transpose is itself.
        dp=propagate(op,dh)
        dp=[[v if pre[i][j]>0 else 0. for j,v in enumerate(row)] for i,row in enumerate(dp)]
        g0=[sum(sx[i][k]*dp[i][j] for i in range(n)) for k in range(self.inputs) for j in range(h)]
        return loss,g0+g1

    def predict(self, features, edges):
        p,_=self.forward(features,edges)
        return {'probabilities':p,'classes':[max(range(self.classes),key=lambda j:row[j]) for row in p],
                'mode':self.mode,'context':'all supplied features and graph edges; no labels'}


def fit(features, edges, labels, train_indices, hidden=12, classes=2, mode='gcn',
        updates=300, learning_rate=.03, seed=13, validation_indices=None):
    x=matrix(features)
    integer(updates,1,5000,'updates')
    lr=number(learning_rate,'learning_rate')
    if not 1e-6<=lr<=.1:raise ValueError('learning rate outside [1e-6,.1]')
    net=Network(len(x[0]),hidden,classes,mode,seed)
    # Validate labels/mask/edges before any optimizer update.
    net.loss_grad(x,edges,labels,train_indices)
    validation = None if validation_indices is None else mask(validation_indices,len(x))
    if validation is not None:
        if set(validation)&set(train_indices):raise ValueError('train/validation masks overlap')
        best_loss=net.loss_grad(x,edges,labels,validation)[0]
    else:best_loss=None
    best_params,best_step=list(net.params),0
    m,v=[0.]*len(net.params),[0.]*len(net.params)
    history=[]
    for step in range(1,updates+1):
        loss,g=net.loss_grad(x,edges,labels,train_indices)
        for j,z in enumerate(g):
            m[j]=.9*m[j]+.1*z;v[j]=.999*v[j]+.001*z*z
            net.params[j]-=lr*(m[j]/(1-.9**step))/(math.sqrt(v[j]/(1-.999**step))+1e-8)
        if not all(math.isfinite(z) for z in net.params):raise ValueError('training diverged')
        if validation is not None:
            val_loss=net.loss_grad(x,edges,labels,validation)[0]
            if val_loss < best_loss:
                best_loss,best_step,best_params=val_loss,step,list(net.params)
        if step==1 or step%50==0 or step==updates:
            history.append({'update':step,'train_loss_after_update':net.loss_grad(x,edges,labels,train_indices)[0]})
    final=artifact(net)
    if validation is not None:net.params=best_params
    else:best_step=updates
    return net,{'history':history,'selected_update':best_step,'final_update':updates,
                'selected_validation_loss':best_loss,'final_artifact':final}


def artifact(net):
    return {'format':'graphlab_v1','inputs':net.inputs,'hidden':net.hidden,
            'classes':net.classes,'mode':net.mode,'parameters':list(net.params)}


def load(value):
    fields={'format','inputs','hidden','classes','mode','parameters'}
    if not isinstance(value,dict) or set(value)!=fields or value['format']!='graphlab_v1':
        raise ValueError('invalid model fields or version')
    n=Network(value['inputs'],value['hidden'],value['classes'],value['mode'])
    p=value['parameters']
    if not isinstance(p,list) or len(p)!=len(n.params):raise ValueError('invalid parameter count')
    n.params=[number(v,'parameter') for v in p]
    return n
