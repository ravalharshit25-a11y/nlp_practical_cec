const $=id=>document.getElementById(id);let DATA=null;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("sample").onclick=async()=>{$("text").value=(await (await fetch("/api/sample")).json()).text};
$("go").onclick=async()=>{
 const fd=new FormData();fd.append("text",$("text").value);fd.append("query",$("query").value);
 if($("file").files[0])fd.append("file",$("file").files[0]);
 $("status").textContent="Analyzing…";$("go").disabled=true;
 try{const r=await fetch("/api/analyze",{method:"POST",body:fd});const j=await r.json();
  if(!r.ok)throw new Error(j.error);DATA=j;render();$("status").textContent="Done.";}
 catch(e){$("status").textContent="Error: "+e.message}finally{$("go").disabled=false}};
const col=l=>l==="High"?"var(--hi)":l==="Medium"?"var(--me)":"var(--lo)";
function render(){const d=DATA;$("out").hidden=false;
 $("score").textContent=d.overall_score;$("level").textContent=d.overall_level+" risk";$("level").style.color=col(d.overall_level);
 $("ring").style.setProperty("--p",d.overall_score*10);$("ring").style.setProperty("--c",col(d.overall_level));
 $("summary").innerHTML=d.summary.map(s=>`<li>${esc(s.length>230?s.slice(0,230)+"…":s)}</li>`).join("");
 const s=d.stats;$("stats").innerHTML=[`${s.words} words`,`${s.clauses} clauses`,`~${s.read_minutes} min read`,`${s.high} high`,`${s.medium} medium`,`${s.low} low`].map(x=>`<span class=pill>${x}</span>`).join("");
 $("toprisks").innerHTML=d.top_risks.length?d.top_risks.map(r=>`<div class=hit>${esc(r[0])} <small>× ${r[1]}</small></div>`).join(""):"No major risks detected.";
 $("missing").innerHTML=d.missing_clauses.length?d.missing_clauses.map(m=>`<span class=pill>${esc(m)}</span>`).join(""):"<small>None of the standard clauses are missing.</small>";
 const mx=Math.max(...Object.values(d.categories));$("cats").innerHTML=Object.entries(d.categories).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<div class=barrow><span>${esc(k)}</span><div class=b><div style="width:${v/mx*100}%"></div></div><span>${v}</span></div>`).join("");
 $("searchbox").hidden=!d.search||!d.search.length;
 if(d.search)$("search").innerHTML=d.search.map(h=>{const c=d.clauses[h.id-1];return `<div class=hit><b>Clause ${h.id} · ${esc(c.category)}</b> <small>match ${h.score}</small><br>${esc(c.simple.plain)}</div>`}).join("");
 $("ents").innerHTML=d.entities.map(e=>`<span class=pill>${esc(e.text)}<i>${e.label}</i></span>`).join("")||"None found.";
 $("obl").innerHTML=d.obligations.map(o=>`<li><b>${esc(o.who)}</b>: ${esc(o.text)}</li>`).join("")||"<li>None detected.</li>";
 const cats=[...new Set(d.clauses.map(c=>c.category))];$("fcat").innerHTML='<option value="">All categories</option>'+cats.map(c=>`<option>${esc(c)}</option>`).join("");
 $("engine").textContent="Engines — NER: "+d.engine.ner+" · Similarity: "+d.engine.similarity+" · Classifier: "+d.engine.classifier;
 list();$("out").scrollIntoView({behavior:"smooth"})}
function list(){const fc=$("fcat").value,fr=$("frisk").value,q=$("fsearch").value.toLowerCase();
 $("clauses").innerHTML=DATA.clauses.filter(c=>(!fc||c.category===fc)&&(!fr||c.risk_level===fr)&&(!q||c.text.toLowerCase().includes(q))).map(c=>`
 <details class="${c.risk_level}" ${c.risk_level==="High"?"open":""}><summary><span>Clause ${c.id}</span><span class=tag>${esc(c.category)} · ${Math.round(c.confidence*100)}%</span><span class="tag rl">${c.risk_level} risk ${c.risk_score}/10</span></summary>
 <div class=plain><b>In plain English:</b> ${esc(c.simple.plain)}<br><small>${esc(c.simple.meaning)}</small></div>
 ${c.risks.map(r=>`<div class=risk><b class=rl>⚠ ${esc(r.title)}</b> <small>(${r.matched_by}, similarity ${r.similarity})</small><br>${esc(r.why)}<br><i>Suggestion:</i> ${esc(r.suggestion)}</div>`).join("")}
 <div class=orig>${esc(c.text)}</div>
 <div class=pills>${c.entities.map(e=>`<span class=pill>${esc(e.text)}<i>${e.label}</i></span>`).join("")}</div></details>`).join("")||"<p class=muted>No clauses match the filters.</p>"}
["fcat","frisk"].forEach(i=>$(i).onchange=list);$("fsearch").oninput=list;
$("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(DATA,null,2)],{type:"application/json"}));a.download="legal_analysis.json";a.click()};
