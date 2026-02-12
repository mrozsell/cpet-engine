# E21 v1.0 — Kinetic Phenotype Engine (standalone module)

class Engine_E21_KineticPhenotype:
    """E21 v1.0 — Kinetic Phenotype Classification & Training Priorities.
    Cross-validates VO₂ kinetics (E14) with thresholds (E02) to produce
    phenotype classification + training priorities.
    References: Poole&Jones 2012, Jones 2011, Barstow 1996, Iannetta 2020."""

    TAU_MOD_ELITE=15; TAU_MOD_TRAINED=25; TAU_MOD_ACTIVE=40
    TAU_HVY_ELITE=20; TAU_HVY_TRAINED=35; TAU_HVY_ACTIVE=50
    SC_MINIMAL=3.0; SC_NORMAL_HI=8.0; SC_HIGH_HI=15.0

    @classmethod
    def run(cls, results: dict, cfg=None) -> dict:
        import numpy as np
        out = {'status':'OK','version':'1.0','engine':'E21_KineticPhenotype',
               'domain_validation':{},'domain_flags':[],
               'tau_class_moderate':None,'tau_class_heavy':None,'tau_class_severe':None,
               'tau_ratio_heavy_mod':None,'sc_class_heavy':None,'sc_class_severe':None,
               'recovery_class':None,'limitation_type':None,'limitation_evidence':[],
               'fiber_type_estimate':{},'phenotype':None,'phenotype_confidence':0.0,
               'phenotype_description':'','phenotype_archetype':'',
               'training_priorities':[],'summary':{},'flags':[]}

        def _f(d,k,fb=None):
            v=d.get(k)
            if v is None: return fb
            try: return float(v)
            except: return fb

        e01=results.get('E01',{}); e02=results.get('E02',{})
        e14=results.get('E14',{}); e18=results.get('E18',{})

        if e14.get('status')!='OK' or e14.get('mode')!='CWR_KINETICS':
            out['status']='NO_KINETICS_DATA'; return out
        stages=e14.get('stages',[])
        if len(stages)<2: out['status']='INSUFFICIENT_STAGES'; return out

        vo2max=_f(e01,'vo2peak_abs_mlmin') or _f(e01,'vo2max_abs_mlmin')
        vo2max_kg=_f(e01,'vo2peak_rel_mlkgmin')
        vt1_spd=_f(e02,'vt1_speed_kmh'); vt2_spd=_f(e02,'vt2_speed_kmh')
        vt1_vo2=_f(e02,'vt1_vo2_abs'); vt2_vo2=_f(e02,'vt2_vo2_abs')
        bm=None
        if cfg:
            ac=cfg.get('_acfg',cfg) if isinstance(cfg,dict) else cfg
            bm=getattr(ac,'body_mass_kg',None)
        if bm is None and vo2max and vo2max_kg and vo2max_kg>0: bm=vo2max/vo2max_kg
        has_thr=vt1_spd is not None and vt2_spd is not None
        if not vo2max or vo2max<=0: out['status']='NO_VO2MAX_REFERENCE'; return out

        # ═══ STEP 1: DOMAIN VALIDATION ═══
        dv={}; dfl=[]
        exp=['MODERATE','HEAVY','SEVERE','VERY_SEVERE']
        for i,stg in enumerate(stages):
            sn=stg.get('stage_num',i+1); spd=stg.get('speed_kmh')
            sv=stg.get('vo2_mean_mlmin') or stg.get('vo2_mean') or 0
            e={'stage':sn,'speed_kmh':spd,
               'vo2_pct_max':round(sv/vo2max*100,1) if sv and vo2max else None,
               'domain_e14':stg.get('domain','?'),'domain_validated':None,
               'validation_method':[]}
            if has_thr and spd:
                p1=spd/vt1_spd*100; p2=spd/vt2_spd*100
                e['pct_vt1_speed']=round(p1,1); e['pct_vt2_speed']=round(p2,1)
                if p1<92: e['domain_validated']='MODERATE'
                elif p2<98: e['domain_validated']='HEAVY'
                elif p2>110: e['domain_validated']='VERY_SEVERE'
                elif p2>=95: e['domain_validated']='SEVERE'
                else: e['domain_validated']='HEAVY_SEVERE_BORDER'
                e['validation_method'].append('speed_vs_thresholds')
            rer=stg.get('rer_mean',0)
            if rer>0:
                rd='MODERATE' if rer<0.95 else ('HEAVY' if rer<1.05 else 'SEVERE')
                if not e['domain_validated']: e['domain_validated']=rd
                e['rer_domain']=rd
            if i<len(exp):
                e['expected_domain']=exp[i]
                if (e['domain_validated'] and e['domain_validated']!=exp[i] and
                    not(e['domain_validated'] in('SEVERE','VERY_SEVERE') and exp[i] in('SEVERE','VERY_SEVERE'))):
                    dfl.append(f"S{sn}_MISMATCH:{exp[i]}→{e['domain_validated']}")
            dv[f'S{sn}']=e
        out['domain_validation']=dv; out['domain_flags']=dfl

        # ═══ STEP 2: τ CLASSIFICATION ═══
        tau_m=tau_h=tau_s=r2_m=r2_h=None
        for stg in stages:
            t=_f(stg,'tau_on_s'); r=_f(stg,'fit_r2'); sn=stg.get('stage_num',0)
            if sn==1: tau_m=t; r2_m=r
            elif sn==2: tau_h=t; r2_h=r
            elif sn==3: tau_s=t
        def _ct(t,e,tr,a):
            if t is None: return None
            return 'ELITE' if t<=e else('TRAINED' if t<=tr else('ACTIVE' if t<=a else 'SLOW'))
        out['tau_class_moderate']=_ct(tau_m,cls.TAU_MOD_ELITE,cls.TAU_MOD_TRAINED,cls.TAU_MOD_ACTIVE)
        out['tau_class_heavy']=_ct(tau_h,cls.TAU_HVY_ELITE,cls.TAU_HVY_TRAINED,cls.TAU_HVY_ACTIVE)
        out['tau_class_severe']='EXPECTED_SLOW' if tau_s and tau_s>50 else('FAST_FOR_SEVERE' if tau_s else None)
        if tau_m and tau_h and tau_m>0: out['tau_ratio_heavy_mod']=round(tau_h/tau_m,2)
        out['summary'].update({'tau_moderate_s':round(tau_m,1) if tau_m else None,
            'tau_heavy_s':round(tau_h,1) if tau_h else None,
            'tau_severe_s':round(tau_s,1) if tau_s else None,
            'r2_moderate':round(r2_m,3) if r2_m else None,
            'r2_heavy':round(r2_h,3) if r2_h else None})

        # ═══ STEP 3: SLOW COMPONENT ═══
        sc_h=sc_s=None
        for stg in stages:
            sc=_f(stg,'sc_pct'); sn=stg.get('stage_num',0)
            if sn==2: sc_h=sc
            elif sn==3: sc_s=sc
        def _csc(s):
            if s is None: return None
            s=abs(s)
            return 'MINIMAL' if s<cls.SC_MINIMAL else('NORMAL' if s<cls.SC_NORMAL_HI else('HIGH' if s<cls.SC_HIGH_HI else 'VERY_HIGH'))
        out['sc_class_heavy']=_csc(sc_h); out['sc_class_severe']=_csc(sc_s)
        out['summary']['sc_heavy_pct']=round(sc_h,1) if sc_h is not None else None
        out['summary']['sc_severe_pct']=round(sc_s,1) if sc_s is not None else None

        # ═══ STEP 4: RECOVERY ═══
        recs=[_f(o,'t_half_s') for o in e14.get('off_kinetics',[]) if _f(o,'t_half_s') is not None and _f(o,'t_half_s')>1]
        if recs:
            mth=float(np.median(recs))
            out['recovery_class']='EXCELLENT' if mth<=30 else('GOOD' if mth<=60 else('MODERATE' if mth<=90 else 'SLOW'))
            out['summary']['recovery_t_half_median_s']=round(mth,1)

        # ═══ STEP 5: LIMITATION ═══
        tc_m=out['tau_class_moderate']; tc_h=out['tau_class_heavy']; sc_hc=out['sc_class_heavy']
        lim='UNDETERMINED'; lev=[]
        if tc_m and tc_h:
            mr={'ELITE':0,'TRAINED':1,'ACTIVE':2,'SLOW':3}.get(tc_m,2)
            hr={'ELITE':0,'TRAINED':1,'ACTIVE':2,'SLOW':3}.get(tc_h,2)
            if mr<=1 and hr<=1: lim='WELL_INTEGRATED'; lev.append(f'τ mod={tc_m}, heavy={tc_h} → oba szybkie')
            elif mr<=1 and hr>=2: lim='DELIVERY_LIMITED'; lev.append(f'τ mod={tc_m}(OK) ale heavy={tc_h}(wolne) → limitacja O₂ delivery')
            elif mr>=2 and hr>=2: lim='PERIPHERAL_LIMITED'; lev.append(f'τ mod={tc_m}, heavy={tc_h} → oba wolne → limitacja oksydacyjna')
            else: lim='MIXED_CHECK_DATA'; lev.append('Nietypowy wzorzec')
        if sc_hc in('HIGH','VERY_HIGH'):
            lev.append(f'SC heavy={sc_hc} → wysoka rekrutacja Type II')
            if lim=='WELL_INTEGRATED': lim='EFFICIENCY_LIMITED'
        elif sc_hc=='MINIMAL': lev.append(f'SC heavy={sc_hc} → znakomita ekonomia mięśniowa')
        out['limitation_type']=lim; out['limitation_evidence']=lev

        # Threshold %VO₂max
        vt1p=vt1_vo2/vo2max*100 if vt1_vo2 and vo2max else None
        vt2p=vt2_vo2/vo2max*100 if vt2_vo2 and vo2max else None
        hzw=vt2p-vt1p if vt1p and vt2p else None
        if vt1p: out['summary']['vt1_pct_vo2max']=round(vt1p,1)
        if vt2p: out['summary']['vt2_pct_vo2max']=round(vt2p,1)
        if hzw: out['summary']['heavy_zone_width_pct']=round(hzw,1)

        # ═══ STEP 6: FIBER TYPE PROXY ═══
        ft={'method':'SC_proxy_Barstow1996','confidence':'LOW'}
        if sc_h is not None:
            a=abs(sc_h)
            if a<3: ft['estimated_type_I_range']='65-80%'; ft['predominance']='TYPE_I_DOMINANT'
            elif a<8: ft['estimated_type_I_range']='50-65%'; ft['predominance']='MIXED'
            elif a<15: ft['estimated_type_I_range']='35-50%'; ft['predominance']='TYPE_II_DOMINANT'
            else: ft['estimated_type_I_range']='<35%'; ft['predominance']='STRONGLY_TYPE_II'
            ft['note']='Indirect: Barstow 1996 r=0.72 SC↔%TypeII'
        out['fiber_type_estimate']=ft

        # ═══ STEP 7: PHENOTYPE ═══
        phen,conf,desc,arch=cls._phenotype(tc_m,tc_h,sc_hc,lim,vt1p,hzw,out['recovery_class'])
        out['phenotype']=phen; out['phenotype_confidence']=conf
        out['phenotype_description']=desc; out['phenotype_archetype']=arch

        # ═══ STEP 8: TRAINING ═══
        out['training_priorities']=cls._training(tc_m,tc_h,sc_hc,lim,
            out['recovery_class'],tau_m,tau_h,sc_h,vt1_spd,vt2_spd)

        # ═══ STEP 9: FLAGS ═══
        if dfl: out['flags'].extend(dfl)
        for stg in stages:
            if stg.get('stage_num')==4:
                d=_f(stg,'duration_s')
                if d and d<180: out['flags'].append(f'S4_SHORT:{d:.0f}s'); out['summary']['s4_tlim_s']=round(d)
        if tau_h and tau_m and tau_h<tau_m*0.8: out['flags'].append('POSSIBLE_PRIMING')
        if r2_m and r2_m<0.7: out['flags'].append(f'LOW_R2_MOD:{r2_m:.3f}')
        if r2_h and r2_h<0.5: out['flags'].append(f'LOW_R2_HVY:{r2_h:.3f}')
        if not has_thr: out['flags'].append('NO_THRESHOLD_DATA')
        return out

    @staticmethod
    def _phenotype(tc_m,tc_h,sc_h,lim,vt1p,hzw,rec):
        sc={'ELITE_AEROBIC':0,'DIESEL':0,'TEMPO_RUNNER':0,'BURST_RECOVER':0,
            'POWER_ENDURANCE':0,'DELIVERY_LIMITED':0,'PERIPHERAL_LIMITED':0}
        if tc_m=='ELITE': sc['ELITE_AEROBIC']+=3;sc['DIESEL']+=2;sc['TEMPO_RUNNER']+=1
        elif tc_m=='TRAINED': sc['DIESEL']+=2;sc['TEMPO_RUNNER']+=2
        elif tc_m=='ACTIVE': sc['BURST_RECOVER']+=1;sc['POWER_ENDURANCE']+=1;sc['PERIPHERAL_LIMITED']+=1
        elif tc_m=='SLOW': sc['PERIPHERAL_LIMITED']+=3;sc['POWER_ENDURANCE']+=1

        if tc_h=='ELITE': sc['ELITE_AEROBIC']+=3;sc['DIESEL']+=2
        elif tc_h=='TRAINED': sc['DIESEL']+=1;sc['TEMPO_RUNNER']+=2
        elif tc_h=='ACTIVE': sc['DELIVERY_LIMITED']+=2;sc['BURST_RECOVER']+=1
        elif tc_h=='SLOW': sc['DELIVERY_LIMITED']+=3;sc['PERIPHERAL_LIMITED']+=1

        if sc_h=='MINIMAL': sc['DIESEL']+=2;sc['ELITE_AEROBIC']+=2;sc['TEMPO_RUNNER']+=1
        elif sc_h=='NORMAL': sc['TEMPO_RUNNER']+=1
        elif sc_h in('HIGH','VERY_HIGH'): sc['POWER_ENDURANCE']+=3;sc['PERIPHERAL_LIMITED']+=1

        if vt1p:
            if vt1p>=75: sc['ELITE_AEROBIC']+=2;sc['DIESEL']+=2
            elif vt1p>=65: sc['DIESEL']+=1;sc['TEMPO_RUNNER']+=2
            elif vt1p>=55: sc['TEMPO_RUNNER']+=1;sc['BURST_RECOVER']+=1
            else: sc['PERIPHERAL_LIMITED']+=1;sc['POWER_ENDURANCE']+=1
        if hzw:
            if hzw>=20: sc['DIESEL']+=1;sc['ELITE_AEROBIC']+=1
            elif hzw<12: sc['POWER_ENDURANCE']+=1
        if rec in('EXCELLENT','GOOD'): sc['BURST_RECOVER']+=2;sc['ELITE_AEROBIC']+=1
        elif rec=='SLOW': sc['PERIPHERAL_LIMITED']+=1
        if lim=='DELIVERY_LIMITED': sc['DELIVERY_LIMITED']+=2
        elif lim=='PERIPHERAL_LIMITED': sc['PERIPHERAL_LIMITED']+=2

        best=max(sc,key=sc.get); bs=sc[best]; tot=sum(sc.values())
        conf=round(bs/max(tot,1),2)
        info={
            'ELITE_AEROBIC':('Zoptymalizowany system aerobowy — szybka kinetyka, minimalna SC, wysokie progi.','Top endurance athlete'),
            'DIESEL':('Silnik wytrzymałościowy — szybkie τ, niski SC, szeroka strefa heavy. Zbudowany do długiego submaks. wysiłku.','Ironman, ultra, HYROX PRO'),
            'TEMPO_RUNNER':('Profil tempowca — dobra kinetyka, umiarkowany SC, sprawny w heavy z potencjałem optymalizacji.','10K–HM, HYROX Open/Pro'),
            'BURST_RECOVER':('Profil powtarzalnych wysiłków — wolniejsze τ ale szybka recovery, dobrze radzi sobie ze zmianami tempa.','Team sport, MMA, CrossFit'),
            'POWER_ENDURANCE':('Profil siłowo-wytrzymałościowy — dominacja Type II, wysoki SC, lepszy w krótkich intensywnych wysiłkach.','CrossFit, sprinter→endurance'),
            'DELIVERY_LIMITED':('Limitacja O₂ delivery — mięśnie OK (szybkie τ mod) ale wolne τ heavy → ograniczenie sercowo-naczyniowe.','Potencjał via trening progowy'),
            'PERIPHERAL_LIMITED':('Limitacja obwodowa — wolna kinetyka we wszystkich domenach → niska zdolność oksydacyjna mięśni.','Potencjał via base + HIIT'),
        }
        d,a=info.get(best,('—','—'))
        return best,conf,d,a

    @staticmethod
    def _training(tc_m,tc_h,sc_h,lim,rec,tau_m,tau_h,sc_hp,vt1s,vt2s):
        pr=[]; p=0
        if tc_m in('ACTIVE','SLOW'):
            p+=1; tgt=f'{tau_m:.0f}→{max(15,tau_m*0.7):.0f}s' if tau_m else '→<25s'
            pr.append({'priority':p,'area':'τ moderate','current':f'{tau_m:.1f}s [{tc_m}]' if tau_m else tc_m,
                'target':tgt,'method':'HIIT 30/30s i 60/60s w severe domain','frequency':'2-3x/tydz',
                'mechanism':'↑zdolność oksydacyjna mitochondriów (Inglis 2024)'})
        elif tc_m=='TRAINED':
            p+=1; pr.append({'priority':p,'area':'τ moderate (optymalizacja)',
                'current':f'{tau_m:.1f}s [{tc_m}]' if tau_m else tc_m,
                'target':f'{tau_m:.0f}→<20s [ELITE]' if tau_m else '→<20s',
                'method':'Tempo runs 20-30min @ VT1-VT2 + HIIT 30/30s','frequency':'2-3x/tydz',
                'mechanism':'↑mitochondria + O₂ extraction'})
        if tc_h in('ACTIVE','SLOW'):
            p+=1; tgt=f'{tau_h:.0f}→{max(20,tau_h*0.65):.0f}s' if tau_h else '→<35s'
            meth='Threshold runs 20-30min @ VT1' if lim=='DELIVERY_LIMITED' else 'Base endurance + threshold runs'
            if vt1s: meth+=f' ({vt1s:.1f} km/h ±0.5)'
            pr.append({'priority':p,'area':'τ heavy','current':f'{tau_h:.1f}s [{tc_h}]' if tau_h else tc_h,
                'target':tgt,'method':meth,'frequency':'2x/tydz',
                'mechanism':'↑O₂ delivery (SV, CO, BV)' if lim=='DELIVERY_LIMITED' else '↑oxidative capacity + perfuzja'})
        if sc_h in('HIGH','VERY_HIGH'):
            p+=1; pr.append({'priority':p,'area':'Slow Component','current':f'{sc_hp:.1f}% [{sc_h}]' if sc_hp is not None else sc_h,
                'target':'<5%','method':'LSD 60-90min <VT1 + technika biegu','frequency':'3-4x/tydz',
                'mechanism':'Adaptacja TypeII→IIa, ↑efektywność (Jones 2011)'})
        elif sc_h=='MINIMAL':
            p+=1; pr.append({'priority':p,'area':'SC (utrzymanie)','current':f'{sc_hp:.1f}% [{sc_h}]' if sc_hp is not None else sc_h,
                'target':'<3%','method':'Kontynuacja treningu bazowego','frequency':'2-3x/tydz','mechanism':'Utrzymanie adaptacji'})
        if rec in('SLOW','MODERATE'):
            p+=1; pr.append({'priority':p,'area':'Recovery kinetics','current':rec,
                'target':'GOOD→EXCELLENT','method':'Repeat-sprint 6-8×30s >VT2 / 3min rest + 4×4min severe',
                'frequency':'1-2x/tydz','mechanism':'↑resynteza PCr, ↑reperfuzja'})
        return pr
