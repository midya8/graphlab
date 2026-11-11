'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
test('actual Chrome joint-deletion census, graphs, budgets, invalidation and trusted downloads',{timeout:180000},async()=>{
 const root=path.join(__dirname,'../docs'),server=http.createServer((req,res)=>{try{const u=new URL(req.url,'http://localhost');const p=path.join(root,u.pathname==='/'?'index.html':u.pathname);res.setHeader('Content-Type',({'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json'})[path.extname(p)]||'text/plain');res.end(fs.readFileSync(p));}catch{res.statusCode=404;res.end('missing');}});await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'graphjoint-chrome-'));let browser,ws,stderr='';
 try{
 browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe'],detached:true});browser.stderr.on('data',d=>stderr=(stderr+d).slice(-5000));
 let startup;browser.on('error',e=>startup=e);
 for(let i=0;i<1200&&!startup&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
 assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),String(startup||stderr));const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];let tab,last;
 for(let i=0;i<1200&&browser.exitCode===null&&!tab;i++){try{const tabs=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json();tab=tabs.find(t=>t.type==='page');}catch(e){last=e;}if(!tab)await delay(25);}assert.ok(tab,String(last||stderr));
 ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};

 const url=process.env.GRAPH_URL||'http://127.0.0.1:'+server.address().port+'/';await call('Page.enable');await call('Page.navigate',{url});
 for(let i=0;i<500;i++){if(await js('Boolean(globalThis.GraphCases&&document.querySelectorAll("#rows tr").length)'))break;await delay(20);}assert.ok(await js('Boolean(globalThis.GraphCases)'));
 const refs=fs.readdirSync(__dirname).filter(f=>/^joint-reference-\d+\.json$/.test(f)).sort().flatMap(f=>JSON.parse(fs.readFileSync(path.join(__dirname,f)))).slice(0,6),screens=process.env.GRAPH_SCREEN_DIR;
 if(screens)fs.mkdirSync(screens,{recursive:true});
 const shot=async name=>{if(!screens)return;await js('document.activeElement.blur();window.scrollTo(0,0)');const m=await call('Page.getLayoutMetrics');const r=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:m.cssContentSize.width,height:m.cssContentSize.height,scale:1}});fs.writeFileSync(path.join(screens,name+'.png'),Buffer.from(r.data,'base64'));};
 const choose=(id,value)=>js(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event('change',{bubbles:true}))`);
 const click=async id=>{const p=await js(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);await call('Input.dispatchMouseEvent',{type:'mouseMoved',...p});await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...p});await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...p});};
 const downloads=fs.mkdtempSync(path.join(os.tmpdir(),'graphjoint-downloads-'));await call('Page.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});let views=0,sets=0,rows=0,files=0;
 for(const width of [1280,768,390,320]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});
 for(let i=0;i<refs.length;i++){
  await choose('control',i);const a=refs[i].report;
  assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'));
  assert.equal(await js('document.getElementById("status").textContent'),'Complete census computed. Binary64, not a formal proof.');
  for(const b of a.budgets){await choose('budget',b.budget);const ids=a.rows.map((_,j)=>j).filter(j=>a.rows[j].deleted_indices.length<=b.budget);
    assert.equal(await js('document.querySelectorAll("#rows tr").length'),b.sets);rows+=b.sets;
    assert.ok((await js('document.getElementById("summary").textContent')).includes(`${b.changed_predictions} change the baseline prediction`));
    for(const j of ids){await choose('set',j);const r=a.rows[j],detail=await js('document.getElementById("detail").textContent');assert.ok(detail.includes(Number(r.margin).toPrecision(9)));assert.ok(detail.includes(Number(r.interaction_residual).toPrecision(9)));assert.ok(detail.includes('All-class prediction: '+r.predicted_class));assert.equal(await js('document.querySelectorAll("#perturbed line[stroke-dasharray]").length'),r.deleted_indices.length);const z=await js('JSON.parse(document.getElementById("logits").textContent)');z.forEach((v,k)=>assert.ok(Math.abs(v-r.target_logits[k])<2e-11));sets++;}
  }
  await choose('budget',a.effective_max_deletions);if(i===0){const j=a.rows.findIndex(r=>r.deleted_indices.join(',')==='1,3');await choose('set',j);}await shot('control-'+i+'-'+width);views++;
  if(width===1280){await click('download');for(let j=0;j<500&&!fs.existsSync(path.join(downloads,'graphlab-joint.json'));j++)await delay(20);const p=path.join(downloads,'graphlab-joint.json');assert.ok(fs.existsSync(p));const data=JSON.parse(fs.readFileSync(p));assert.deepEqual(data.input,refs[i].input);
  const replay=spawn(process.env.PYTHON||'python',['examples/replay_joint.py',p],{cwd:path.join(__dirname,'..'),stdio:['ignore','pipe','pipe']});let err='';replay.stderr.on('data',d=>err+=d);const code=await new Promise(r=>replay.once('exit',r));assert.equal(code,0,err);fs.unlinkSync(p);files++;}
 }}
 // Trusted keyboard change invalidates every rendered result and download.
 await choose('control',0);await js('document.getElementById("request").focus();document.getElementById("request").select()');await call('Input.insertText',{text:'{"bad":true}'});assert.equal(await js('document.getElementById("download").disabled'),true);assert.equal(await js('document.getElementById("original").children.length'),0);await click('compute');assert.ok((await js('document.getElementById("status").textContent')).startsWith('Rejected:'));assert.equal(await js('document.querySelectorAll("#rows tr").length'),0);
 await choose('control',0);await choose('budget',0);assert.equal(await js('document.querySelectorAll("#rows tr").length'),1);await click('compute');assert.equal(await js('document.getElementById("download").disabled'),false);
 console.log(JSON.stringify({views,subsetSelections:sets,tableRows:rows,downloads:files,widths:[1280,768,390,320],trustedInput:true,trustedCompute:true,invalidation:true}));await fs.promises.rm(downloads,{recursive:true,force:true});
 }finally{server.close();if(ws)ws.close();if(browser){try{process.kill(-browser.pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')throw e;}if(browser.exitCode===null)await new Promise(r=>{browser.once('exit',r);setTimeout(r,3000);});}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:20,retryDelay:100});}
});
