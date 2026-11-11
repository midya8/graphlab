(function(root){'use strict';
const integer=(v,a,b,n)=>{if(!Number.isInteger(v)||v<a||v>b)throw Error(n+' must be integer in ['+a+','+b+']');return v;};
const num=(v,n)=>{if(typeof v!=='number'||!Number.isFinite(v))throw Error(n+' must be finite');return v;};
function validate(m,x,e){
 if(!m||Object.keys(m).sort().join(',')!=='classes,format,hidden,inputs,mode,parameters'||m.format!=='graphlab_v1')throw Error('invalid model');
 integer(m.inputs,1,16,'inputs');integer(m.hidden,1,32,'hidden');integer(m.classes,2,8,'classes');
 if(!['gcn','mlp'].includes(m.mode))throw Error('invalid mode');
 if(!Array.isArray(m.parameters)||m.parameters.length!==m.inputs*m.hidden+m.hidden*m.classes)throw Error('parameter count');m.parameters.forEach(v=>num(v,'parameter'));
 if(!Array.isArray(x)||x.length<1||x.length>128||x.some(r=>!Array.isArray(r)||r.length!==m.inputs))throw Error('invalid features');x.flat().forEach(v=>{if(Math.abs(num(v,'feature'))>100)throw Error('feature bound');});
 if(!Array.isArray(e)||e.length>x.length*(x.length-1)/2)throw Error('invalid edges');const seen=new Set();e.forEach(p=>{if(!Array.isArray(p)||p.length!==2)throw Error('edge pair');p.forEach(v=>integer(v,0,x.length-1,'edge index'));const t=p.slice().sort((a,b)=>a-b).join(',');if(p[0]===p[1]||seen.has(t))throw Error('loop or duplicate');seen.add(t);});
}
function logits(m,x,e,node){
 const n=x.length,adj=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));if(m.mode==='gcn')for(const[a,b]of e)adj[a][b]=adj[b][a]=1;
 const deg=adj.map(r=>r.reduce((a,b)=>a+b,0)),s=adj.map((r,i)=>r.map((v,j)=>v/Math.sqrt(deg[i]*deg[j])));
 const mm=(a,b)=>a.map(r=>b[0].map((_,j)=>r.reduce((z,v,k)=>z+v*b[k][j],0)));
 const w0=Array.from({length:m.inputs},(_,i)=>m.parameters.slice(i*m.hidden,(i+1)*m.hidden));
 const w1=Array.from({length:m.hidden},(_,i)=>m.parameters.slice(m.inputs*m.hidden+i*m.classes,m.inputs*m.hidden+(i+1)*m.classes));
 const h=mm(mm(s,x),w0).map(r=>r.map(v=>Math.max(0,v))),out=mm(mm(s,h),w1)[node];out.forEach(v=>num(v,'logit'));return out;
}
function combinations(n,k){const out=[];function walk(a,start){if(a.length===k){out.push(a.slice());return;}for(let i=start;i<=n-(k-a.length);i++){a.push(i);walk(a,i+1);a.pop();}}walk([],0);return out;}
function run(m,x,e,node,positive,negative,max_deletions=2,max_forwards=4096){
 validate(m,x,e);integer(node,0,x.length-1,'node');integer(positive,0,m.classes-1,'positive');integer(negative,0,m.classes-1,'negative');if(positive===negative)throw Error('margin classes must differ');integer(max_deletions,0,8,'max_deletions');integer(max_forwards,1,65536,'max_forwards');
 const k=Math.min(max_deletions,e.length);let required=1,c=1;for(let j=1;j<=k;j++){c=c*(e.length-j+1)/j;required+=c;if(required>max_forwards)throw Error('joint audit exceeds forward budget');}
 const rows=[],single={};let base,baseClass;
 for(let size=0;size<=k;size++)for(const ids of combinations(e.length,size)){
 const gone=new Set(ids),z=logits(m,x,e.filter((_,i)=>!gone.has(i)),node),margin=num(z[positive]-z[negative],'margin');let pred=0;for(let i=1;i<z.length;i++)if(z[i]>z[pred])pred=i;
 if(base===undefined){base=margin;baseClass=pred;}const delta=num(margin-base,'delta');if(size===1)single[ids[0]]=delta;const additive=num(ids.reduce((v,i)=>v+single[i],0),'sum');
 rows.push({deleted_indices:ids,deleted_edges:ids.map(i=>e[i].slice().sort((a,b)=>a-b)),target_logits:z,margin,margin_change:delta,singleton_sum:additive,interaction_residual:num(delta-additive,'residual'),predicted_class:pred,prediction_changed:pred!==baseClass});}
 const budgets=[];for(let b=0;b<=k;b++){const ids=rows.map((_,i)=>i).filter(i=>rows[i].deleted_indices.length<=b),changes=ids.filter(i=>rows[i].prediction_changed);const arg=fn=>ids.reduce((best,i)=>fn(rows[i])<fn(rows[best])?i:best,ids[0]);budgets.push({budget:b,sets:ids.length,minimum_margin_row:arg(r=>r.margin),maximum_margin_row:arg(r=>-r.margin),maximum_interaction_row:arg(r=>-Math.abs(r.interaction_residual)),changed_predictions:changes.length,first_changed_row:changes.length?changes[0]:null});}
 return {schema:'graphlab-joint/1',node,positive_class:positive,negative_class:negative,requested_max_deletions:max_deletions,effective_max_deletions:k,edge_count:e.length,baseline_margin:base,baseline_class:baseClass,forwards:rows.length,required_forwards:required,max_forwards,rows,budgets,scope:'all original-edge subsets up to budget; weights/features fixed; degrees recomputed; binary64 census, not causal or formal robustness proof'};
}
const api={run,validate,logits};if(typeof module!=='undefined')module.exports=api;root.GraphJoint=api;
})(typeof globalThis!=='undefined'?globalThis:this);
