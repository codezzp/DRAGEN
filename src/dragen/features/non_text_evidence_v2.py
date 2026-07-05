"""Build non-text Evidence-v2 features from observable cascade/window tables."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict,deque
from pathlib import Path
from typing import Any,Dict,Iterable,List,Mapping,Optional,Tuple
from dragen.features.evidence_schema import DEFAULT_EVIDENCE_SCHEMA, FORBIDDEN_INPUT_COLUMNS, write_schema
from dragen.features.evidence_normalizer import clamp01, entropy_from_counts, gini, log1p_pos, row_value, safe_div

IDN=["cascade_idx","window_idx","user_idx"]
IDW=["cascade_idx","window_idx"]
BIN=60

def build_non_text_evidence_v2(*,feature_dir:Path,window_dir:Path,global_candidate_edges:Optional[Path],out_dir:Path)->Dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    win=read_csv(feature_dir/"window_features.csv"); node=read_csv(feature_dir/"node_window_features.csv")
    edge=read_csv(window_dir/"edge_window_table.csv"); glob=read_csv(global_candidate_edges) if global_candidate_edges and global_candidate_edges.exists() else []
    forbid(node,"node_window_features.csv"); forbid(win,"window_features.csv")
    b=Builder(win,node,edge,glob); nr,wr=b.build()
    write_csv(out_dir/"node_evidence_features.csv",IDN+DEFAULT_EVIDENCE_SCHEMA.node_columns,nr)
    write_csv(out_dir/"window_evidence_features.csv",IDW+DEFAULT_EVIDENCE_SCHEMA.window_columns,wr)
    write_schema(out_dir/"feature_schema.json")
    diag=diagnostics(nr,wr); diag.update({"feature_dir":str(feature_dir),"window_dir":str(window_dir),"global_candidate_edges":str(global_candidate_edges or ""),"feature_groups":DEFAULT_EVIDENCE_SCHEMA.to_json()})
    write_json(out_dir/"evidence_diagnostics.json",diag); return diag

class Builder:
    def __init__(self,win,node,edge,glob):
        self.win=sorted(win,key=lambda r:(int(r["cascade_idx"]),int(r["window_idx"])))
        self.node=sorted(node,key=lambda r:(int(r["cascade_idx"]),int(r["window_idx"]),int(r["user_idx"])))
        self.edge=edge; self.glob=glob
        self.wc=grp(self.win,"cascade_idx"); self.nc=grp(self.node,"cascade_idx"); self.ec=grp(edge,"cascade_idx"); self.gc=grp(glob,"cascade_idx")
    def build(self):
        nr=[]; wr=[]
        for c in sorted(self.nc,key=lambda x:int(x)):
            ctx=Ctx(c,self.wc.get(c,[]),self.nc[c],self.ec.get(c,[]),self.gc.get(c,[]))
            nr+=ctx.nodes(); wr+=ctx.windows()
        return nr,wr

class Ctx:
    def __init__(self,c,win,node,edge,glob):
        self.c=int(c); self.win=sorted(win,key=lambda r:int(r["window_idx"])); self.node=sorted(node,key=lambda r:(int(r["window_idx"]),int(r["user_idx"])))
        self.nw=grp(self.node,"window_idx"); self.ids=[int(r["window_idx"]) for r in self.win]; self.maxw=max(self.ids or [0])
        self.bywu={(int(r["window_idx"]),int(r["user_idx"])):r for r in self.node}
        self.cur,self.ctx=self.edge_maps(edge); self.g=self.global_stats(glob); self.coord={w:self.coord_graph(w) for w in self.ids}
        self.prev=defaultdict(float); self.prev_delta=defaultdict(float)
    def nodes(self):
        out=[]
        for r in self.node:
            w=int(r["window_idx"]); u=int(r["user_idx"]); row={"cascade_idx":self.c,"window_idx":w,"user_idx":u}
            row.update(self.beh(r,w,u)); row.update(self.tmp(r,w,u)); row.update(self.struc(r,w,u)); row.update(self.coor(r,w,u)); row.update(self.globev(r,w,u))
            out.append(roundrow(row)); p=row_value(r,"num_posts_cur"); self.prev_delta[u]=p-self.prev[u]; self.prev[u]=p
        return out
    def windows(self):
        out=[]; seen=set(); prev_heat=0.0; prev_delta=0.0
        for r in self.win:
            w=int(r["window_idx"]); ns=self.nw.get(str(w),[]); active={int(x["user_idx"]) for x in ns if row_value(x,"num_posts_cur")>0}
            new=active-seen; rep=active&seen; seen|=active; heat=row_value(r,"heat_cur",row_value(r,"num_retweets_cur")); delta=heat-prev_heat; acc=delta-prev_delta; prev_heat=heat; prev_delta=delta
            bins=self.bin_counts(ns); deg=[row_value(x,"in_degree_cur")+row_value(x,"out_degree_cur") for x in ns]; depths=[row_value(x,"depth") for x in ns if row_value(x,"has_tree_feature")>0]
            cur=self.cur.get(w,[]); ctx=self.ctx.get(w,[]); co=self.coord.get(w,Coord.empty()); n=max(len(ns),1); glob=self.g["edges"]
            row={"cascade_idx":self.c,"window_idx":w,
            "win_heat_cur_log":log1p_pos(heat),"win_heat_cum_log":log1p_pos(row_value(r,"heat_cum",row_value(r,"num_retweets_cum"))),"win_delta_heat":delta,"win_growth_rate":safe_div(delta,max(prev_heat,1.0)),"win_acceleration":acc,"win_active_users_cur_log":log1p_pos(row_value(r,"num_active_users_cur",len(active))),"win_active_users_cum_log":log1p_pos(row_value(r,"num_active_users_cum",len(seen))),"win_new_user_ratio":safe_div(len(new),len(active)),"win_repeat_user_ratio":safe_div(len(rep),len(active)),
            "win_temporal_entropy":entropy_from_counts(bins.values()),"win_max_bin_share":safe_div(max(bins.values()) if bins else 0,len(active)),"win_gini_time_bin":gini(bins.values()),"win_burstiness":safe_div(max(bins.values()) if bins else 0,sum(bins.values())),"win_active_bin_ratio":safe_div(len([v for v in bins.values() if v>0]),max(len(bins),1)),
            "win_edge_count_cur_log":log1p_pos(len(cur)),"win_edge_count_ctx_log":log1p_pos(len(ctx)),"win_density_cur":safe_div(len(cur),n*max(n-1,1)),"win_density_ctx":safe_div(len(ctx),n*max(n-1,1)),"win_degree_gini":gini(deg),"win_largest_component_ratio":largest_component_ratio(cur,active or {int(x["user_idx"]) for x in ns}),"win_depth_mean":safe_div(sum(depths),len(depths)),"win_depth_max":max(depths) if depths else 0.0,"win_branch_entropy":entropy_from_counts([int(row_value(x,"out_degree_cur")) for x in ns]),
            "win_coord_edge_count_log":log1p_pos(len(co.edges)),"win_coord_density":safe_div(len(co.edges),n*max(n-1,1)/2),"win_coord_largest_component_ratio":co.largest(n),"win_coord_avg_component_size":co.avg_comp(),"win_coord_follow_supported_ratio":safe_div(co.follow_edges,len(co.edges)),
            "win_global_candidate_edge_count_log":log1p_pos(len(glob)),"win_global_candidate_density":safe_div(len(glob),n*max(n-1,1)),"win_global_follow_overlap_cur":safe_div(overlap(glob,cur),len(cur)),"win_global_follow_overlap_ctx":safe_div(overlap(glob,ctx),len(ctx))}
            out.append(roundrow(row))
        return out

    def beh(self,r,w,u):
        pc=row_value(r,"num_posts_cur"); pm=sum(row_value(x,"num_posts_cur") for x in self.nw.get(str(w),[])); pcum=row_value(r,"num_posts_cum"); cm=sum(row_value(x,"num_posts_cum") for x in self.nw.get(str(w),[])); aw=row_value(r,"active_window_count")
        denom=max(row_value(r,"first_seen_time")+row_value(r,"time_since_first_seen"),1.0)
        return {"beh_active_cur":float(pc>0),"beh_posts_cur_log":log1p_pos(pc),"beh_posts_cum_log":log1p_pos(pcum),"beh_visible_texts_cur_log":log1p_pos(row_value(r,"num_texts_cur")),"beh_visible_texts_cum_log":log1p_pos(row_value(r,"num_texts_visible")),"beh_contribution_share_cur":safe_div(pc,pm),"beh_contribution_share_cum":safe_div(pcum,cm),"beh_active_window_count_log":log1p_pos(aw),"beh_active_window_ratio":safe_div(aw,w+1),"beh_first_seen_norm":clamp01(safe_div(row_value(r,"first_seen_time"),denom)),"beh_time_since_first_seen_norm":clamp01(safe_div(row_value(r,"time_since_first_seen"),denom)),"beh_window_reactivation":float(pc>0 and self.prev.get(u,0.0)<=0 and w>0)}
    def tmp(self,r,w,u):
        pc=row_value(r,"num_posts_cur"); delta=pc-self.prev.get(u,0.0); acc=delta-self.prev_delta.get(u,0.0); sb=self.same_bin(r,w); active=max(sum(1 for x in self.nw.get(str(w),[]) if row_value(x,"num_posts_cur")>0),1); pos=safe_div(w,max(self.maxw,1)); denom=max(row_value(r,"first_seen_time")+row_value(r,"time_since_first_seen"),1.0)
        return {"tmp_window_pos":pos,"tmp_first_seen_window_norm":clamp01(safe_div(row_value(r,"first_seen_time"),denom)),"tmp_time_since_first_seen_norm":clamp01(safe_div(row_value(r,"time_since_first_seen"),denom)),"tmp_is_early_participant":float(pos<=0.33),"tmp_is_late_participant":float(pos>=0.67),"tmp_activity_delta":delta,"tmp_activity_acceleration":acc,"tmp_same_bin_user_count_log":log1p_pos(sb),"tmp_same_bin_user_share":safe_div(sb,active),"tmp_nearest_event_gap_norm":clamp01(safe_div(row_value(r,"parent_time_gap"),max(row_value(r,"time_since_first_seen"),1.0))),"tmp_burst_bin_rank":safe_div(sb,active),"tmp_temporal_sync_score":safe_div(sb,active)}
    def struc(self,r,w,u):
        ic=row_value(r,"in_degree_cur"); oc=row_value(r,"out_degree_cur"); ix=row_value(r,"in_degree_ctx"); ox=row_value(r,"out_degree_ctx"); dc=ic+oc; dx=ix+ox; dcm=row_value(r,"in_degree_cum")+row_value(r,"out_degree_cum"); ns=self.nw.get(str(w),[]); tdc=sum(row_value(x,"in_degree_cur")+row_value(x,"out_degree_cur") for x in ns); tdm=sum(row_value(x,"in_degree_cum")+row_value(x,"out_degree_cum") for x in ns); md=max([row_value(x,"depth") for x in ns] or [1.0])
        return {"str_in_degree_cur_log":log1p_pos(ic),"str_out_degree_cur_log":log1p_pos(oc),"str_total_degree_cur_log":log1p_pos(dc),"str_in_degree_ctx_log":log1p_pos(ix),"str_out_degree_ctx_log":log1p_pos(ox),"str_total_degree_ctx_log":log1p_pos(dx),"str_degree_cum_log":log1p_pos(dcm),"str_degree_delta":dc-dx,"str_degree_share_cur":safe_div(dc,tdc),"str_degree_share_cum":safe_div(dcm,tdm),"str_proxy_depth_norm":clamp01(safe_div(row_value(r,"depth"),max(md,1.0))),"str_parent_score":row_value(r,"parent_score"),"str_child_count_log":log1p_pos(oc),"str_local_clustering":local_clustering(u,self.cur.get(w,[]))}
    def coor(self,r,w,u):
        co=self.coord.get(w,Coord.empty()); deg=co.deg.get(u,0); n=max(len(self.nw.get(str(w),[])),1); csum=sum(self.coord.get(t,Coord.empty()).deg.get(u,0) for t in self.ids if t<=w)
        return {"coord_sync_degree_cur_log":log1p_pos(deg),"coord_sync_weight_sum_cur":co.wsum.get(u,0.0),"coord_sync_degree_cum_log":log1p_pos(csum),"coord_component_size_ratio":safe_div(co.comp_size(u),n),"coord_component_rank_norm":safe_div(co.comp_size(u),max(co.max_comp,1)),"coord_follow_supported_sync_ratio":safe_div(co.follow_by_node.get(u,0),deg),"coord_same_parent_sync_count_log":log1p_pos(co.same_parent.get(u,0)),"coord_same_time_bin_sync_count_log":log1p_pos(co.same_bin.get(u,0)),"coord_coordination_clustering":co.local_density(u),"coord_coordination_density_local":safe_div(deg,max(n-1,1))}
    def globev(self,r,w,u):
        g=self.g; inc=g["in"].get(u,0); out=g["out"].get(u,0); total=inc+out; ns=self.nw.get(str(w),[]); n=max(len(ns),1); cur=self.cur.get(w,[]); ctx=self.ctx.get(w,[]); neigh=g["neigh"].get(u,set()); active=[v for v in neigh if (w,v) in self.bywu and row_value(self.bywu[(w,v)],"num_posts_cur")>0]; ndegs=[row_value(self.bywu[(w,v)],"in_degree_cur")+row_value(self.bywu[(w,v)],"out_degree_cur") for v in neigh if (w,v) in self.bywu]
        return {"glob_follow_in_cand_log":log1p_pos(inc),"glob_follow_out_cand_log":log1p_pos(out),"glob_follow_total_cand_log":log1p_pos(total),"glob_follow_candidate_share":safe_div(total,max(n-1,1)),"glob_follow_current_overlap":node_overlap(u,g["edges"],cur),"glob_follow_context_overlap":node_overlap(u,g["edges"],ctx),"glob_follow_sync_overlap":safe_div(len(active),max(len(neigh),1)),"glob_follow_neighbor_active_mean":safe_div(len(active),max(len(neigh),1)),"glob_follow_neighbor_degree_mean":safe_div(sum(ndegs),len(ndegs)),"glob_follow_reciprocal_count_log":log1p_pos(g["recip"].get(u,0)),"glob_follow_supported_edge_ratio":safe_div(overlap(g["edges"],cur),len(cur))}
    def edge_maps(self,edges):
        cur=defaultdict(list); ctx=defaultdict(list)
        for r in edges:
            w=int(r.get("window_idx",0)); e=(int(r.get("src_user_idx",r.get("src_local_idx",0))),int(r.get("dst_user_idx",r.get("dst_local_idx",0))))
            (ctx if r.get("window_scope")=="context" else cur)[w].append(e)
        return cur,ctx
    def global_stats(self,rows):
        inc=defaultdict(int); out=defaultdict(int); recip=defaultdict(int); neigh=defaultdict(set); edges=set()
        for r in rows:
            s=edge_user(r,"src"); d=edge_user(r,"dst")
            if s is None or d is None or s==d: continue
            edges.add((s,d)); out[s]+=1; inc[d]+=1; neigh[s].add(d); neigh[d].add(s)
        for s,d in list(edges):
            if (d,s) in edges: recip[s]+=1
        return {"in":inc,"out":out,"recip":recip,"neigh":neigh,"edges":edges}
    def coord_graph(self,w):
        ns=self.nw.get(str(w),[]); active=[int(r["user_idx"]) for r in ns if row_value(r,"num_posts_cur")>0]
        bins={int(r["user_idx"]):self.bin(r) for r in ns}; depth={int(r["user_idx"]):int(row_value(r,"depth")) for r in ns}; struct=set(self.cur.get(w,[]))|set(self.ctx.get(w,[])); glob=self.g["edges"]; edges={}; flags={}
        for i,u in enumerate(active):
            for v in active[i+1:]:
                sb=bins.get(u)==bins.get(v); sp=depth.get(u)==depth.get(v) and depth.get(u,0)>0; st=(u,v) in struct or (v,u) in struct; fl=(u,v) in glob or (v,u) in glob
                if not(sb or sp or st or fl): continue
                k=(min(u,v),max(u,v)); edges[k]=0.35*sb+0.25*sp+0.20*st+0.20*fl; flags[k]={"same_bin":sb,"same_parent":sp,"follow":fl}
        return Coord(edges,flags)
    def bin_counts(self,ns):
        out=defaultdict(int)
        for r in ns:
            if row_value(r,"num_posts_cur")>0: out[self.bin(r)]+=1
        return out
    def bin(self,r): return int((row_value(r,"first_seen_time")+row_value(r,"time_since_first_seen"))//BIN)
    def same_bin(self,r,w):
        b=self.bin(r); return sum(1 for x in self.nw.get(str(w),[]) if row_value(x,"num_posts_cur")>0 and self.bin(x)==b)

class Coord:
    def __init__(self,edges,flags):
        self.edges=dict(edges); self.flags=dict(flags); self.deg=defaultdict(int); self.wsum=defaultdict(float); self.follow_by_node=defaultdict(int); self.same_parent=defaultdict(int); self.same_bin=defaultdict(int); self.adj=defaultdict(set); self.follow_edges=0
        for (u,v),w in self.edges.items():
            self.deg[u]+=1; self.deg[v]+=1; self.wsum[u]+=w; self.wsum[v]+=w; self.adj[u].add(v); self.adj[v].add(u); fl=self.flags.get((u,v),{})
            if fl.get("follow"): self.follow_edges+=1; self.follow_by_node[u]+=1; self.follow_by_node[v]+=1
            if fl.get("same_parent"): self.same_parent[u]+=1; self.same_parent[v]+=1
            if fl.get("same_bin"): self.same_bin[u]+=1; self.same_bin[v]+=1
        self.comp=self.components(); self.max_comp=max(self.comp.values()) if self.comp else 0
    @classmethod
    def empty(cls): return cls({}, {})
    def comp_size(self,u): return self.comp.get(u,0)
    def largest(self,n): return safe_div(self.max_comp,n)
    def avg_comp(self): return safe_div(sum(self.comp.values()),len(self.comp))
    def local_density(self,u):
        ns=list(self.adj.get(u,set())); n=len(ns)
        if n<2: return 0.0
        links=sum(1 for i,a in enumerate(ns) for b in ns[i+1:] if b in self.adj.get(a,set()))
        return safe_div(links,n*(n-1)/2)
    def components(self):
        comp={}; seen=set()
        for s in self.adj:
            if s in seen: continue
            q=deque([s]); seen.add(s); cur=[]
            while q:
                u=q.popleft(); cur.append(u)
                for v in self.adj.get(u,set()):
                    if v not in seen: seen.add(v); q.append(v)
            for u in cur: comp[u]=len(cur)
        return comp


def read_csv(path:Path)->List[Dict[str,str]]:
    if path is None or not path.exists(): return []
    with path.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def write_csv(path:Path,fields:List[str],rows:List[Mapping[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,0) for k in fields})
def write_json(path:Path,payload:Mapping[str,Any]): path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def grp(rows:Iterable[Mapping[str,str]],key:str):
    g=defaultdict(list)
    for r in rows: g[str(r[key])].append(r)
    return g
def edge_user(r,side):
    for k in [f"{side}_user_idx",f"{side}_local_idx"]:
        if r.get(k) not in (None,""): return int(r[k])
    return None
def overlap(a,b):
    a=set(a); b=set(b); return len(a&b)+len({(v,u) for u,v in a}&b)
def node_overlap(u,glob,edges):
    inc={(a,b) for a,b in glob if a==u or b==u}
    if not inc: return 0.0
    e=set(edges)|{(b,a) for a,b in edges}; return safe_div(len(inc&e),len(inc))
def largest_component_ratio(edges,users):
    users=set(users)
    if not users: return 0.0
    adj=defaultdict(set)
    for u,v in edges: adj[u].add(v); adj[v].add(u)
    seen=set(); best=0
    for s in users:
        if s in seen: continue
        q=deque([s]); seen.add(s); n=0
        while q:
            u=q.popleft(); n+=1
            for v in adj.get(u,set()):
                if v not in seen: seen.add(v); q.append(v)
        best=max(best,n)
    return safe_div(best,len(users))
def local_clustering(u,edges):
    adj=defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    ns=list(adj.get(u,set())); n=len(ns)
    if n<2: return 0.0
    links=sum(1 for i,a in enumerate(ns) for b in ns[i+1:] if b in adj.get(a,set()))
    return safe_div(links,n*(n-1)/2)
def roundrow(r): return {k:(round(v,8) if isinstance(v,float) else v) for k,v in r.items()}
def forbid(rows,source):
    if rows:
        bad=sorted(set(rows[0])&FORBIDDEN_INPUT_COLUMNS)
        if bad: raise ValueError(f"Forbidden label-construction columns in {source}: {bad}")
def diagnostics(node_rows,window_rows):
    warn=[]; d={"num_node_rows":len(node_rows),"num_window_rows":len(window_rows),"missing_rate_by_feature":{},"nan_count_by_feature":{},"inf_count_by_feature":{},"feature_mean_std":{},"feature_min_max":{},"warnings":warn}
    for label,rows,cols in [("node",node_rows,DEFAULT_EVIDENCE_SCHEMA.node_columns),("window",window_rows,DEFAULT_EVIDENCE_SCHEMA.window_columns)]:
        for c in cols:
            vals=[float(r.get(c,0) or 0) for r in rows]; n=len(vals); mean=safe_div(sum(vals),n); var=safe_div(sum((v-mean)**2 for v in vals),n)
            d["missing_rate_by_feature"][f"{label}.{c}"]=safe_div(sum(1 for r in rows if r.get(c) in (None,"")),n)
            d["nan_count_by_feature"][f"{label}.{c}"]=sum(1 for v in vals if math.isnan(v))
            d["inf_count_by_feature"][f"{label}.{c}"]=sum(1 for v in vals if math.isinf(v))
            d["feature_mean_std"][f"{label}.{c}"]={"mean":round(mean,8),"std":round(var**0.5,8)}
            d["feature_min_max"][f"{label}.{c}"]={"min":round(min(vals),8) if vals else 0,"max":round(max(vals),8) if vals else 0}
            if vals and max(abs(v) for v in vals)<=1e-12: warn.append(f"{label}.{c} is all zero")
    return d
