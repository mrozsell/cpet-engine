# =========
import numpy as np
import pandas as pd

# Lazy import to avoid circular dependency
try:
    from engine_core import RAW_PROTOCOLS, compile_protocol_for_apply, PROTOCOLS_DB
except ImportError:
    RAW_PROTOCOLS = {}
    compile_protocol_for_apply = None
    PROTOCOLS_DB = {}

            # ═══════════════════════════════════════════════════════════════

# CHART_JS — interactive chart JavaScript (Canvas-based)
# Defined as global constant, used by _render_charts_html
try:
    CHART_JS
except NameError:
    CHART_JS = '''
// CPET Charts - Canvas-based interactive charts
let _activeChart = null;
const _btnOrigColors = {};

function toggleChart(type) {
    const c = document.getElementById('chart_container');
    if (!c) return;

    // Restore all buttons to original state
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        const origColor = b.getAttribute('data-color') || b.style.borderColor;
        b.style.background = '#fff';
        b.style.color = origColor;
        b.style.fontWeight = '600';
    });

    if (_activeChart === type) {
        c.style.display='none';
        _activeChart=null;
        return;
    }

    _activeChart = type;
    const btn = document.getElementById('btn_'+type);
    if (btn) {
        const origColor = btn.getAttribute('data-color') || btn.style.borderColor;
        btn.style.background = origColor;
        btn.style.color = '#fff';
    }

    c.style.display='block';
    c.innerHTML='<canvas id="cpet_canvas" width="860" height="420" style="max-width:100%;background:white;"></canvas>';
    const canvas=document.getElementById('cpet_canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    const M={t:40,r:60,b:50,l:70};
    const pW=W-M.l-M.r, pH=H-M.t-M.b;
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);

    // Zone colors for protocol steps (matching training zones)
    // warmup=gray, Z1=blue, Z2=green, Z3=yellow, Z4=orange, Z5=red, recovery=light-green
    const ZONE_COLORS = [
        'rgba(200,200,210,0.25)',   // 0: warmup/rest
        'rgba(147,197,253,0.30)',   // 1: Z1 blue
        'rgba(74,222,128,0.30)',    // 2: Z2 green
        'rgba(251,191,36,0.30)',    // 3: Z3 yellow
        'rgba(249,115,22,0.30)',    // 4: Z4 orange
        'rgba(239,68,68,0.30)',     // 5: Z5 red
    ];

    function getStepZoneColor(steps, stepIdx) {
        const s = steps[stepIdx];
        if (!s) return ZONE_COLORS[0];
        const tp = s.type || '';
        if (tp === 'warmup' || tp === 'rest') return ZONE_COLORS[0];
        if (tp === 'cooldown' || tp === 'recovery') return ZONE_COLORS[0];
        // For exercise steps: map by fractional position
        const exSteps = steps.filter(x => x.type !== 'warmup' && x.type !== 'rest' && x.type !== 'cooldown' && x.type !== 'recovery');
        const exIdx = exSteps.indexOf(s);
        if (exIdx < 0) return ZONE_COLORS[1];
        const nEx = exSteps.length;
        const frac = nEx > 1 ? exIdx / (nEx - 1) : 0.5;
        if (frac <= 0.15) return ZONE_COLORS[1];
        if (frac <= 0.35) return ZONE_COLORS[2];
        if (frac <= 0.55) return ZONE_COLORS[3];
        if (frac <= 0.75) return ZONE_COLORS[4];
        return ZONE_COLORS[5];
    }

    function drawProtocolBands(steps, tMin, tMax, tStop) {
        if (!steps || steps.length === 0) return;
        steps.forEach((s, i) => {
            const x1 = M.l + Math.max(0, (s.start_s - tMin)) / (tMax - tMin) * pW;
            const endS = s.end_s || (tStop || tMax);
            const x2 = M.l + Math.min(pW, (endS - tMin) / (tMax - tMin) * pW);
            ctx.fillStyle = getStepZoneColor(steps, i);
            ctx.fillRect(x1, M.t, x2 - x1, pH);
            // Step label
            ctx.fillStyle = '#475569';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('S' + (i + 1), (x1 + x2) / 2, M.t + pH - 6);
        });
    }

    function drawAxis(label, min, max, side, color) {
        ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1;
        const steps=5;
        for(let i=0;i<=steps;i++) {
            const y=M.t+pH-pH*i/steps;
            if(side==='left') {
                ctx.beginPath(); ctx.moveTo(M.l,y); ctx.lineTo(W-M.r,y); ctx.stroke();
                ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='right';
                ctx.fillText((min+(max-min)*i/steps).toFixed(1), M.l-6, y+4);
            } else {
                ctx.fillStyle=color||'#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='left';
                ctx.fillText((min+(max-min)*i/steps).toFixed(1), W-M.r+6, y+4);
            }
        }
        ctx.fillStyle=color||'#64748b'; ctx.font='12px sans-serif';
        if(side==='left') { ctx.textAlign='center'; ctx.save(); ctx.translate(14,M.t+pH/2); ctx.rotate(-Math.PI/2); ctx.fillText(label,0,0); ctx.restore(); }
        else { ctx.textAlign='center'; ctx.save(); ctx.translate(W-8,M.t+pH/2); ctx.rotate(-Math.PI/2); ctx.fillText(label,0,0); ctx.restore(); }
    }

    function drawLine(times,vals,tMin,tMax,vMin,vMax,color,width) {
        ctx.strokeStyle=color; ctx.lineWidth=width||2; ctx.beginPath();
        let started=false;
        for(let i=0;i<times.length;i++) {
            if(vals[i]===null||vals[i]===undefined) continue;
            const x=M.l+(times[i]-tMin)/(tMax-tMin)*pW;
            const y=M.t+pH-(vals[i]-vMin)/(vMax-vMin)*pH;
            if(!started){ctx.moveTo(x,y);started=true;}else{ctx.lineTo(x,y);}
        }
        ctx.stroke();
    }

    function drawVT(time,tMin,tMax,label,color) {
        if(!time) return;
        const x=M.l+(time-tMin)/(tMax-tMin)*pW;
        ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
        ctx.beginPath(); ctx.moveTo(x,M.t); ctx.lineTo(x,M.t+pH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=color; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
        ctx.fillText(label,x,M.t-6);
    }

    function fmtTime(s) { return Math.floor(s/60)+":"+("0"+Math.floor(s%60)).slice(-2); }

    function drawTimeAxis(tMin,tMax) {
        const steps=8;
        ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='center';
        for(let i=0;i<=steps;i++) {
            const t=tMin+(tMax-tMin)*i/steps;
            const x=M.l+pW*i/steps;
            ctx.fillText(fmtTime(t),x,H-M.b+18);
        }
        ctx.fillText('Czas (min:ss)',M.l+pW/2,H-6);
    }

    function drawDots(times,vals,tMin,tMax,vMin,vMax,color,radius) {
        for(let i=0;i<times.length;i++) {
            if(vals[i]===null||vals[i]===undefined) continue;
            const x=M.l+(times[i]-tMin)/(tMax-tMin)*pW;
            const y=M.t+pH-(vals[i]-vMin)/(vMax-vMin)*pH;
            ctx.beginPath(); ctx.arc(x,y,radius||4,0,Math.PI*2);
            ctx.fillStyle=color; ctx.fill();
            ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
        }
    }

    function drawTitle(text) {
        ctx.fillStyle='#1e293b'; ctx.font='bold 14px sans-serif'; ctx.textAlign='center';
        ctx.fillText(text,W/2,22);
    }

    function drawLegend(items) {
        let x = M.l + 10;
        items.forEach(([label, color]) => {
            ctx.fillStyle=color; ctx.fillRect(x,M.t+8,20,3);
            ctx.fillStyle='#374151'; ctx.font='11px sans-serif'; ctx.textAlign='left';
            ctx.fillText(label, x+24, M.t+12);
            x += ctx.measureText(label).width + 38;
        });
    }

    function drawStopLine(tStop, tMin, tMax) {
        if (!tStop) return;
        const xs=M.l+(tStop-tMin)/(tMax-tMin)*pW;
        ctx.strokeStyle='#1e293b'; ctx.lineWidth=2; ctx.setLineDash([8,4]);
        ctx.beginPath(); ctx.moveTo(xs,M.t); ctx.lineTo(xs,M.t+pH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle='#1e293b'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
        ctx.fillText('STOP',xs,M.t-6);
    }

    const d = CD;

    // ═══ PROTOCOL CHART ═══
    if(type==='proto' && d.kinetics) {
        const t=d.kinetics.time||[], vo2=d.kinetics.vo2||[];
        const steps=d.kinetics.protocol_steps||[];
        const spd=d.kinetics.speed||[], pwr=d.kinetics.power||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vo2Max=Math.max(...vo2.filter(v=>v!==null))*1.1;
        const hasSpd=spd.length>0&&spd.some(v=>v!==null&&v>0);
        const hasPwr=pwr.length>0&&pwr.some(v=>v!==null&&v>0);
        const load=hasSpd?spd:(hasPwr?pwr:null);
        const loadMax=load?Math.max(...load.filter(v=>v!==null&&v>0))*1.15:0;
        drawTitle('Protokol testu'+(d.kinetics.protocol_name?' \u2014 '+d.kinetics.protocol_name:''));
        drawProtocolBands(steps, tMin, tMax, d.kinetics.t_stop);
        drawAxis('VO2 ml/min',0,vo2Max,'left','#ef4444');
        if(load) drawAxis(hasSpd?'Speed km/h':'Power W',0,loadMax,'right','#8b5cf6');
        drawTimeAxis(tMin,tMax);
        if(load) drawLine(t,load,tMin,tMax,0,loadMax,'#8b5cf6',2);
        drawLine(t,vo2,tMin,tMax,0,vo2Max,'#ef4444',2);
        drawStopLine(d.kinetics.t_stop, tMin, tMax);
        drawVT(d.kinetics.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.kinetics.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['VO2','#ef4444']].concat(load?[[hasSpd?'Speed':'Power','#8b5cf6']]:[]));
    }

    // ═══ VO2 KINETICS ═══
    if(type==='kinetics' && d.kinetics) {
        const t=d.kinetics.time||[], vo2=d.kinetics.vo2||[];
        const spd=d.kinetics.speed||[], pwr=d.kinetics.power||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vo2Max=Math.max(...vo2.filter(v=>v!==null))*1.1;
        const hasSpd=spd.length>0&&spd.some(v=>v!==null&&v>0);
        const hasPwr=pwr.length>0&&pwr.some(v=>v!==null&&v>0);
        const load=hasSpd?spd:(hasPwr?pwr:null);
        const loadMax=load?Math.max(...load.filter(v=>v!==null&&v>0))*1.15:0;
        drawTitle('VO2 Kinetics'+(d.kinetics.protocol_name?' \u2014 '+d.kinetics.protocol_name:''));
        drawProtocolBands(d.kinetics.protocol_steps||[], tMin, tMax, d.kinetics.t_stop);
        drawAxis('VO2 ml/min',0,vo2Max,'left','#ef4444');
        if(load) drawAxis(hasSpd?'Speed km/h':'Power W',0,loadMax,'right',hasSpd?'#8b5cf6':'#f59e0b');
        drawTimeAxis(tMin,tMax);
        if(load) drawLine(t,load,tMin,tMax,0,loadMax,hasSpd?'#8b5cf6':'#f59e0b',1.5);
        drawLine(t,vo2,tMin,tMax,0,vo2Max,'#ef4444',2);
        drawStopLine(d.kinetics.t_stop, tMin, tMax);
        drawVT(d.kinetics.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.kinetics.vt2_time,tMin,tMax,'VT2','#f59e0b');
        if(d.kinetics.vo2peak) {
            const yp=M.t+pH-(d.kinetics.vo2peak)/vo2Max*pH;
            ctx.strokeStyle='#ef4444'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
            ctx.beginPath(); ctx.moveTo(M.l,yp); ctx.lineTo(W-M.r,yp); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle='#ef4444'; ctx.font='10px sans-serif'; ctx.textAlign='right';
            ctx.fillText('VO2peak '+Math.round(d.kinetics.vo2peak),W-M.r-4,yp-4);
        }
        drawLegend([['VO2','#ef4444']].concat(load?[[hasSpd?'Speed':'Power',hasSpd?'#8b5cf6':'#f59e0b']]:[]));
    }

    // ═══ V-SLOPE ═══
    if(type==='vslope' && d.gas) {
        const t=d.gas.time||[], vo2=d.gas.vo2||[], vco2=d.gas.vco2||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vMax=Math.max(...vo2,...vco2)*1.1;
        drawTitle('V-slope \u2014 VO2 & VCO2');
        drawAxis('ml/min',0,vMax,'left','#374151');
        drawTimeAxis(tMin,tMax);
        drawLine(t,vo2,tMin,tMax,0,vMax,'#ef4444',2);
        drawLine(t,vco2,tMin,tMax,0,vMax,'#3b82f6',2);
        drawVT(d.gas.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.gas.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['VO2','#ef4444'],['VCO2','#3b82f6']]);
    }

    // ═══ FAT/CHO ═══
    if(type==='fatchho' && d.substrate) {
        const t=d.substrate.time||[], cho=(d.substrate.cho||[]).map(v=>v*60), fat=(d.substrate.fat||[]).map(v=>v*60);
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vMax=Math.max(...cho,...fat)*1.2;
        drawTitle('Substrat \u2014 FAT & CHO (g/h)');
        drawAxis('g/h',0,vMax,'left','#374151');
        drawTimeAxis(tMin,tMax);
        ctx.globalAlpha=0.3;
        ctx.fillStyle='rgba(251,191,36,0.5)';
        ctx.beginPath(); ctx.moveTo(M.l,M.t+pH);
        for(let i=0;i<t.length;i++){const x=M.l+(t[i]-tMin)/(tMax-tMin)*pW;const y=M.t+pH-(fat[i]||0)/vMax*pH;ctx.lineTo(x,y);}
        ctx.lineTo(M.l+pW,M.t+pH); ctx.closePath(); ctx.fill();
        ctx.fillStyle='rgba(59,130,246,0.4)';
        ctx.beginPath(); ctx.moveTo(M.l,M.t+pH);
        for(let i=0;i<t.length;i++){const x=M.l+(t[i]-tMin)/(tMax-tMin)*pW;const y=M.t+pH-(cho[i]||0)/vMax*pH;ctx.lineTo(x,y);}
        ctx.lineTo(M.l+pW,M.t+pH); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=1;
        drawLine(t,fat,tMin,tMax,0,vMax,'#f59e0b',2);
        drawLine(t,cho,tMin,tMax,0,vMax,'#3b82f6',2);
        drawVT(d.substrate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.substrate.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['FAT','#f59e0b'],['CHO','#3b82f6']]);
    }

    // ═══ LACTATE ═══
    if(type==='lac' && d.lactate) {
        const pts=d.lactate.points||[];
        if(pts.length<2) return;
        const times=pts.map(p=>p.time_sec||p.time||0);
        const las=pts.map(p=>p.la||p.lactate||0);
        const tMin=Math.min(...times)-30, tMax=Math.max(...times)+30;
        const lMax=Math.max(...las)*1.2;
        drawTitle('Krzywa mleczanowa');
        drawAxis('Laktat [mmol/L]',0,lMax,'left','#dc2626');
        drawTimeAxis(tMin,tMax);
        const pt=d.lactate.poly_time||[], pf=d.lactate.poly_fit||[];
        if(pt.length>2) drawLine(pt,pf,tMin,tMax,0,lMax,'#fca5a5',1.5);
        drawDots(times,las,tMin,tMax,0,lMax,'#dc2626',5);
        drawVT(d.lactate.lt1_time,tMin,tMax,'LT1','#16a34a');
        drawVT(d.lactate.lt2_time,tMin,tMax,'LT2','#dc2626');
        drawVT(d.lactate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.lactate.vt2_time,tMin,tMax,'VT2','#f59e0b');
    }

    // ═══ NIRS ═══
    if(type==='nirs' && d.nirs) {
        const traces=d.nirs.traces||[];
        if(traces.length===0) return;
        const tr=traces[0];
        const times=tr.time_sec||tr.time||[];
        const vals=tr.smo2||tr.values||tr.SmO2||[];
        if(times.length<2||vals.length<2) return;
        const validVals=vals.filter(v=>v!==null&&v!==undefined&&!isNaN(v));
        if(validVals.length<2) return;
        const tMin=Math.min(...times), tMax=Math.max(...times);
        const vMin=Math.min(...validVals)*0.9;
        const vMax=Math.max(...validVals)*1.1;
        drawTitle('NIRS \u2014 SmO2 (%)');
        drawAxis('SmO2 [%]',vMin,vMax,'left','#2563eb');
        drawTimeAxis(tMin,tMax);
        drawLine(times,vals,tMin,tMax,vMin,vMax,'#2563eb',2);
        drawVT(d.nirs.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.nirs.vt2_time,tMin,tMax,'VT2','#f59e0b');
    }

    // ═══ DUAL (Lactate + SmO2) ═══
    if(type==='dual' && d.lactate && d.nirs && d.nirs.traces && d.nirs.traces.length>0) {
        const pts=d.lactate.points||[];
        const tr=d.nirs.traces[0];
        const nTimes=tr.time_sec||tr.time||[];
        const nVals=tr.smo2||tr.values||tr.SmO2||[];
        if(pts.length<2||nTimes.length<2) return;
        const allT=[...pts.map(p=>p.time_sec||p.time||0),...nTimes];
        const tMin=Math.min(...allT)-20, tMax=Math.max(...allT)+20;
        const lMax=Math.max(...pts.map(p=>p.la||0))*1.2;
        const validN=nVals.filter(v=>v!==null&&v!==undefined&&!isNaN(v));
        const sMin=validN.length>0?Math.min(...validN)*0.9:0;
        const sMax=validN.length>0?Math.max(...validN)*1.1:100;
        drawTitle('Laktat + SmO2');
        drawAxis('Laktat [mmol/L]',0,lMax,'left','#dc2626');
        drawAxis('SmO2 [%]',sMin,sMax,'right','#2563eb');
        drawTimeAxis(tMin,tMax);
        drawLine(nTimes,nVals,tMin,tMax,sMin,sMax,'#2563eb',2);
        const pt=d.lactate.poly_time||[], pf=d.lactate.poly_fit||[];
        if(pt.length>2) drawLine(pt,pf,tMin,tMax,0,lMax,'#fca5a5',1.5);
        drawDots(pts.map(p=>p.time_sec||p.time||0),pts.map(p=>p.la||0),tMin,tMax,0,lMax,'#dc2626',5);
        drawVT(d.lactate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.lactate.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['Laktat','#dc2626'],['SmO2','#2563eb']]);
    }
}

// Init: store original button colors
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        b.setAttribute('data-color', b.style.color || b.style.borderColor);
    });
});
// Also immediate init for non-DOMContentLoaded scenarios
setTimeout(function() {
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        if (!b.getAttribute('data-color')) {
            b.setAttribute('data-color', b.style.color || b.style.borderColor);
        }
    });
}, 100);
'''

# INTERPRETATION ENGINE — Normy wiekowe + auto-interpretacja CPET
# Źródła: Cooper Institute/ACSM 2021, FRIEND Registry, INSCYD 2024
# ═══════════════════════════════════════════════════════════════

COOPER_NORMS = {
    'male': {
        20: {'superior': 55.4, 'excellent': 51.1, 'good': 45.4, 'fair': 41.7},
        30: {'superior': 54.0, 'excellent': 48.3, 'good': 44.0, 'fair': 40.5},
        40: {'superior': 52.5, 'excellent': 46.4, 'good': 42.4, 'fair': 38.5},
        50: {'superior': 48.9, 'excellent': 43.4, 'good': 39.2, 'fair': 35.6},
        60: {'superior': 45.7, 'excellent': 39.5, 'good': 35.5, 'fair': 32.3},
        70: {'superior': 42.1, 'excellent': 36.7, 'good': 32.3, 'fair': 29.4},
    },
    'female': {
        20: {'superior': 49.6, 'excellent': 43.9, 'good': 39.5, 'fair': 36.1},
        30: {'superior': 47.4, 'excellent': 42.4, 'good': 37.8, 'fair': 34.4},
        40: {'superior': 45.3, 'excellent': 39.7, 'good': 36.3, 'fair': 33.0},
        50: {'superior': 41.1, 'excellent': 36.7, 'good': 33.0, 'fair': 30.1},
        60: {'superior': 37.8, 'excellent': 33.0, 'good': 30.0, 'fair': 27.5},
        70: {'superior': 36.7, 'excellent': 30.9, 'good': 28.1, 'fair': 25.9},
    }
}

ATHLETE_NORMS = {
    'male':   [('Elita', 70), ('Wyczynowy', 60), ('Zaawansowany', 52), ('Amator', 45), ('Początkujący', 38)],
    'female': [('Elita', 60), ('Wyczynowy', 50), ('Zaawansowany', 42), ('Amator', 36), ('Początkująca', 30)],
}

def _interp_get_decade(age):
    d = int(age // 10) * 10
    return max(20, min(70, d))

def _interp_lerp_thresholds(age, sex):
    """Interpolate Cooper thresholds for exact age."""
    s = sex if sex in ('male','female') else 'male'
    norms = COOPER_NORMS[s]
    d = max(20, min(70, age))
    d_lo = int(d // 10) * 10
    d_lo = max(20, min(70, d_lo))
    d_hi = min(70, d_lo + 10)
    if d_lo == d_hi:
        return norms[d_lo]
    frac = (d - d_lo) / (d_hi - d_lo)
    lo, hi = norms[d_lo], norms[d_hi]
    return {k: round(lo[k] + (hi[k] - lo[k]) * frac, 1) for k in lo}

def interpret_vo2max(vo2_mlkgmin, age, sex):
    thr = _interp_lerp_thresholds(age, sex)
    v = vo2_mlkgmin
    if v >= thr['superior']:
        cat, pct, col = 'Superior', 95, '#8b5cf6'
    elif v >= thr['excellent']:
        cat, pct, col = 'Excellent', 80, '#3b82f6'
    elif v >= thr['good']:
        cat, pct, col = 'Good', 60, '#22c55e'
    elif v >= thr['fair']:
        cat, pct, col = 'Fair', 40, '#f59e0b'
    else:
        cat, pct, col = 'Poor', 20, '#ef4444'
    # Refine percentile within band
    bands = [('superior',95,99), ('excellent',80,94), ('good',60,79), ('fair',40,59)]
    for lo_key, plo, phi in bands:
        hi_key_idx = ['fair','good','excellent','superior'].index(lo_key)
        keys = ['fair','good','excellent','superior']
        if lo_key == 'superior' and v >= thr['superior']:
            ext = min((v - thr['superior']) / max(thr['superior']*0.1, 1), 1.0)
            pct = int(95 + ext * 4)
            break
        if hi_key_idx < 3:
            nxt = keys[hi_key_idx + 1]
            if v >= thr[lo_key] and v < thr[nxt]:
                frac = (v - thr[lo_key]) / max(thr[nxt] - thr[lo_key], 0.1)
                pct = int(plo + frac * (phi - plo))
                break
    else:
        if v < thr['fair']:
            pct = max(5, int(40 * v / max(thr['fair'], 1)))
    # Athlete level
    s = sex if sex in ('male','female') else 'male'
    athlete_level = 'Nietrenujący/ca'
    for lvl, threshold in ATHLETE_NORMS[s]:
        if v >= threshold:
            athlete_level = lvl
            break
    return {'category': cat, 'percentile': min(99, max(1, pct)),
            'athlete_level': athlete_level, 'color': col, 'thresholds': thr}

def interpret_thresholds(vt1_vo2, vt2_vo2, vo2peak, vt1_hr, vt2_hr, hr_max):
    r = {}
    if vo2peak and vo2peak > 0:
        r['vt1_pct_vo2'] = round(vt1_vo2 / vo2peak * 100, 1) if vt1_vo2 else None
        r['vt2_pct_vo2'] = round(vt2_vo2 / vo2peak * 100, 1) if vt2_vo2 else None
    if hr_max and hr_max > 0:
        r['vt1_pct_hr'] = round(vt1_hr / hr_max * 100, 1) if vt1_hr else None
        r['vt2_pct_hr'] = round(vt2_hr / hr_max * 100, 1) if vt2_hr else None
    vp = r.get('vt1_pct_vo2')
    if vp:
        if vp >= 80: r['aerobic_base'] = 'Doskonała'
        elif vp >= 70: r['aerobic_base'] = 'Bardzo dobra'
        elif vp >= 60: r['aerobic_base'] = 'Dobra'
        elif vp >= 50: r['aerobic_base'] = 'Umiarkowana'
        else: r['aerobic_base'] = 'Słaba'
    r['gap_bpm'] = round(vt2_hr - vt1_hr) if vt1_hr and vt2_hr else None
    return r

def interpret_test_validity(rer_peak, hr_max, age, lactate_peak=None):
    criteria = []
    hr_pred = 220 - age
    if rer_peak and rer_peak >= 1.15: criteria.append('RER ≥1.15')
    elif rer_peak and rer_peak >= 1.10: criteria.append('RER ≥1.10')
    if hr_max and hr_max >= 0.90 * hr_pred: criteria.append(f'HR ≥90% pred ({hr_max}/{hr_pred})')
    if lactate_peak and lactate_peak >= 8.0: criteria.append(f'La ≥8 ({lactate_peak:.1f})')
    elif lactate_peak and lactate_peak >= 6.0: criteria.append(f'La ≥6 ({lactate_peak:.1f})')
    n = len(criteria)
    if n >= 2: conf = 'Wysoka'
    elif n == 1 and rer_peak and rer_peak >= 1.10: conf = 'Wysoka'
    elif n == 1: conf = 'Umiarkowana'
    else: conf = 'Niska'
    is_max = n >= 1 and (not rer_peak or rer_peak >= 1.05)
    if rer_peak and rer_peak < 1.00: is_max = False; conf = 'Niska'
    return {'is_maximal': is_max, 'confidence': conf, 'criteria': criteria,
            'rer_desc': 'Doskonały' if rer_peak and rer_peak>=1.15 else 'Dobry' if rer_peak and rer_peak>=1.10 else 'Akceptowalny' if rer_peak and rer_peak>=1.05 else 'Submaximalny'}

def generate_training_recs(ct):
    """Generate training recommendations based on CPET profile."""
    recs = []
    vt1p = ct.get('_interp_vt1_pct_vo2')
    vt2p = ct.get('_interp_vt2_pct_vo2')
    cat = ct.get('_interp_vo2_category', '')
    ath = ct.get('_interp_vo2_athlete_level', '')
    gap = ct.get('_interp_gap_bpm')

    # Zone distribution recommendation
    if vt1p and vt1p < 60:
        recs.append({
            'title': 'Priorytet: Baza tlenowa (Strefa 1-2)',
            'desc': f'VT1 przy {vt1p:.0f}% VO2max wskazuje na niedostateczną bazę aerobową. Zalecenie: 70-80% objętości treningowej poniżej VT1.',
            'zones': '80% Z1-Z2 | 15% Z3 | 5% Z4-Z5',
            'icon': '🏃', 'color': '#22c55e'
        })
    elif vt1p and vt2p and (vt2p - vt1p) < 20:
        recs.append({
            'title': 'Priorytet: Poszerzenie strefy progowej',
            'desc': f'Wąski gap VT1-VT2 ({vt2p-vt1p:.0f}% VO2max). Tempo runs i SST (Sweet Spot Training) mogą podnieść VT2.',
            'zones': '65% Z1-Z2 | 25% Z3 | 10% Z4-Z5',
            'icon': '⚡', 'color': '#f59e0b'
        })
    elif vt2p and vt2p >= 88:
        recs.append({
            'title': 'Rozwój: Zwiększenie VO2max (Strefa 5)',
            'desc': f'VT2 przy {vt2p:.0f}% VO2max — progi wysoko ustawione. Dalszy postęp przez interwały VO2max (3-5 min @ 95-100% VO2max).',
            'zones': '75% Z1-Z2 | 10% Z3 | 10% Z4 | 5% Z5',
            'icon': '🔥', 'color': '#ef4444'
        })
    else:
        recs.append({
            'title': 'Rozwój: Polaryzowany model treningowy',
            'desc': 'Balanced profil — kontynuuj model 80/20 (80% nisko, 20% wysoko) z akcentem na progi.',
            'zones': '75% Z1-Z2 | 15% Z3 | 10% Z4-Z5',
            'icon': '⚖️', 'color': '#3b82f6'
        })

    # Fat metabolism
    fat = ct.get('FATmax_g_min')
    try: fat_v = float(fat) if fat and str(fat) not in ('-','None','') else None
    except: fat_v = None
    if fat_v and fat_v < 0.4:
        recs.append({
            'title': 'Metabolizm: Trening spalania tłuszczów',
            'desc': f'FATmax {fat_v:.2f} g/min — niska oksydacja lipidów. Długie sesje Z2 (60-90 min) na czczo lub z niską CHO mogą poprawić.',
            'icon': '🍃', 'color': '#84cc16'
        })

    # Running economy
    re_val = ct.get('E06_RE_mlO2_per_kg_per_km')
    try: re_f = float(re_val) if re_val and str(re_val) not in ('-','None','') else None
    except: re_f = None
    if re_f and re_f > 220:
        recs.append({
            'title': 'Technika: Poprawa ekonomii biegu',
            'desc': f'RE {re_f:.0f} mlO2/kg/km — potencjał do poprawy. Drills techniczne, strides, plyometria, hill sprints.',
            'icon': '👟', 'color': '#06b6d4'
        })

    return recs

def generate_observations(ct):
    obs = []
    def add(text, otype, prio, icon=None):
        icons = {'positive':'🟢','neutral':'🔵','warning':'🟡','negative':'🔴'}
        obs.append({'text': text, 'type': otype, 'priority': prio, 'icon': icon or icons.get(otype,'⚪')})

    vo2_rel = ct.get('_interp_vo2_mlkgmin')
    vo2_cat = ct.get('_interp_vo2_category','')
    vo2_pct = ct.get('_interp_vo2_percentile','?')
    vo2_ath = ct.get('_interp_vo2_athlete_level','')

    if vo2_rel and vo2_cat:
        cat_pl = {'Superior':'Wybitna','Excellent':'Bardzo dobra','Good':'Dobra','Fair':'Przeciętna','Poor':'Niska'}.get(vo2_cat, vo2_cat)
        add(f'VO2peak {vo2_rel:.1f} ml/kg/min — percentyl ~{vo2_pct} dla wieku. Kategoria: {cat_pl}. Poziom sportowy: {vo2_ath}.',
            'positive' if vo2_cat in ('Superior','Excellent') else 'neutral' if vo2_cat == 'Good' else 'warning', 1)

    vt1p = ct.get('_interp_vt1_pct_vo2')
    if vt1p:
        base = ct.get('_interp_aerobic_base','')
        _pc = ct.get('_performance_context', {})
        _vt1_v = _pc.get('v_vt1_kmh')
        _vt1_mas_pct = _pc.get('vt1_pct_vref')
        _vt1_mas_src = _pc.get('v_ref_source', '')
        _vt1_level = _pc.get('level_by_speed', '')

        # Build VT1 text with velocity context
        _vt1_txt = f'VT1 przy {vt1p:.0f}% VO₂max'
        if _vt1_v:
            _vt1_txt += f' / {_vt1_v:.1f} km/h'
        if _vt1_mas_pct and 'MAS_external' in _vt1_mas_src:
            _vt1_txt += f' ({_vt1_mas_pct:.0f}% MAS)'

        # Qualify base using BOTH %VO2 AND speed
        if _vt1_v and _vt1_mas_pct:
            # Speed-validated assessment
            if vt1p >= 70 and _vt1_mas_pct >= 60:
                _vt1_txt += ' — bardzo dobra baza tlenowa. Efektywny metabolizm tłuszczowy.'
                _vt1_mood = 'positive'
            elif vt1p >= 70 and _vt1_mas_pct < 50:
                _vt1_txt += ' — wysoki %VO₂max ale niska prędkość absolutna, wskazuje na niski pułap tlenowy.'
                _vt1_mood = 'warning'
            elif vt1p >= 55 and _vt1_mas_pct >= 50:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'neutral'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa, prędkość do rozwinięcia.'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'
        elif _vt1_v:
            # Speed available but no MAS — use absolute speed only
            if vt1p >= 70:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'positive'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'
        else:
            # No speed data at all — fallback to %VO2 only with caveat
            if vt1p >= 70:
                _vt1_txt += f' — {base.lower()} baza tlenowa (brak danych prędkości do pełnej walidacji).'
                _vt1_mood = 'positive'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa (brak danych prędkości do pełnej walidacji).'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'

        add(_vt1_txt, _vt1_mood, 2)

    vt2p = ct.get('_interp_vt2_pct_vo2')
    # Performance Context — VT2 z odniesieniem do prędkości/MAS
    _pc = ct.get('_performance_context', {})
    _pc_v = _pc.get('v_vt2_kmh')
    _pc_mas_pct = _pc.get('vt2_pct_vref')
    _pc_mas_src = _pc.get('v_ref_source', '')
    _pc_level = _pc.get('level_by_speed', '')
    _pc_level_pct = _pc.get('level_by_pct_vo2', '')
    _pc_match = _pc.get('levels_match', True)

    if vt2p:
        # Base text — mood determined by BOTH %VO2 AND speed level
        _vt2_txt = f'VT2 przy {vt2p:.0f}% VO₂max'

        # Add speed context
        if _pc_v:
            _vt2_txt += f' / {_pc_v:.1f} km/h'

        # Add MAS% if available
        if _pc_mas_pct and 'MAS_external' in _pc_mas_src:
            _vt2_txt += f' ({_pc_mas_pct:.0f}% MAS)'

        # Determine mood using speed-validated logic
        if _pc_level and _pc_level in ('Elite', 'Well-trained'):
            _vt2_mood = 'positive'
        elif _pc_level and _pc_level == 'Trained':
            _vt2_mood = 'neutral'
        elif _pc_level and _pc_level in ('Recreational', 'Sedentary'):
            _vt2_mood = 'warning'
        else:
            # No speed classification — fallback to %VO2 only
            if vt2p >= 88: _vt2_mood = 'positive'
            elif vt2p >= 80: _vt2_mood = 'neutral'
            else: _vt2_mood = 'warning'

        # Add level classification
        if _pc_level:
            _vt2_txt += f' — poziom <b>{_pc_level}</b>'

        # Add mismatch warning / context
        if not _pc_match and _pc_level_pct:
            _vt2_txt += f' (uwaga: %VO₂max sugeruje {_pc_level_pct})'

        if vt2p >= 88 and _pc_level in ('Sedentary', 'Recreational'):
            _vt2_txt += '. Wysoki próg względny ale niska prędkość → niski pułap VO₂max.'
            _vt2_mood = 'warning'
        elif vt2p >= 88 and _pc_level == 'Trained':
            _vt2_txt += '. Wysoki próg względny, prędkość na poziomie Trained — dalszy postęp przez interwały VO₂max.'
        elif vt2p >= 88 and _pc_level in ('Well-trained', 'Elite'):
            _vt2_txt += '. Doskonały próg potwierdzona prędkością absolutną.'
        elif vt2p >= 88 and not _pc_level:
            _vt2_txt += '. Wysoki próg (brak danych prędkości do pełnej walidacji).'
        elif vt2p < 80:
            _vt2_txt += '. Potencjał do rozwoju przez trening tempo/SST.'

        add(_vt2_txt, _vt2_mood, 3)

    gap = ct.get('_interp_gap_bpm')
    if gap is not None:
        if gap < 10:
            add(f'Wąska strefa VT1-VT2 ({gap} bpm). Ograniczona przestrzeń do treningu progowego.', 'warning', 4)
        elif gap >= 25:
            add(f'Szeroka strefa VT1-VT2 ({gap} bpm) — duża przestrzeń do treningu tempo/threshold.', 'positive', 4)

    is_max = ct.get('_interp_test_maximal')
    conf = ct.get('_interp_test_confidence','')
    rer_p = ct.get('RER_peak')
    try:
        rer_val = float(rer_p) if rer_p and str(rer_p) != '-' else None
    except: rer_val = None
    if rer_val:
        if rer_val >= 1.15:
            add(f'RER peak {rer_val:.2f} — test maximalny, doskonały wysiłek metaboliczny.', 'positive', 2)
        elif rer_val >= 1.10:
            add(f'RER peak {rer_val:.2f} — test maximalny, dobry wysiłek.', 'positive', 2)
        elif rer_val >= 1.05:
            add(f'RER peak {rer_val:.2f} — akceptowalny wysiłek, ale bliski granicy submaximalności.', 'neutral', 2)
        else:
            add(f'RER peak {rer_val:.2f} — test prawdopodobnie SUBMAXIMALNY. VO2peak może być niedoszacowane.', 'negative', 1)

    ve_sl = ct.get('VE_VCO2_slope')
    try:
        ve_val = float(ve_sl) if ve_sl and str(ve_sl) != '-' else None
    except: ve_val = None
    if ve_val:
        if ve_val < 25:
            add(f'VE/VCO2 slope {ve_val:.1f} — doskonała efektywność wentylacyjna.', 'positive', 5)
        elif ve_val < 30:
            add(f'VE/VCO2 slope {ve_val:.1f} — prawidłowa efektywność wentylacyjna.', 'neutral', 5)
        elif ve_val < 36:
            add(f'VE/VCO2 slope {ve_val:.1f} — łagodnie podwyższony (norma <30).', 'warning', 5)
        else:
            add(f'VE/VCO2 slope {ve_val:.1f} — podwyższony. Obniżona efektywność wentylacyjna.', 'negative', 4)

    fatmax = ct.get('FATmax_g_min')
    try:
        fat_val = float(fatmax) if fatmax and str(fatmax) != '-' else None
    except: fat_val = None
    if fat_val and fat_val > 0:
        sex = ct.get('sex','male')
        thr = 0.7 if sex == 'male' else 0.5
        if fat_val >= thr:
            add(f'FATmax {fat_val:.2f} g/min — dobra zdolność oksydacji tłuszczów.', 'positive', 6)
        else:
            add(f'FATmax {fat_val:.2f} g/min — umiarkowana oksydacja tłuszczów. Trening Z2 może to poprawić.', 'neutral', 6)

    # O2-pulse analysis
    o2p = ct.get('O2pulse_peak')
    try: o2p_val = float(o2p) if o2p and str(o2p) not in ('-','None','') else None
    except: o2p_val = None
    if o2p_val:
        _sx = ct.get('_interp_sex', 'male')
        o2p_norm = 18 if _sx == 'male' else 13
        if o2p_val >= o2p_norm * 1.3:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — wysoki. Dobra objętość wyrzutowa.', 'positive', 5)
        elif o2p_val >= o2p_norm * 0.8:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — w normie.', 'neutral', 6)
        else:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — obniżony. Możliwy limiter centralny (SV).', 'warning', 4)

    # Running economy (E06)
    re_val = ct.get('E06_RE_mlO2_per_kg_per_km')
    try: re_f = float(re_val) if re_val and str(re_val) not in ('-','None','') else None
    except: re_f = None
    if re_f:
        if re_f < 180:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — doskonała (elitarna klasa).', 'positive', 5)
        elif re_f < 210:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — dobra.', 'neutral', 6)
        elif re_f < 240:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — przeciętna. Potencjał poprawy przez drills techniczne i plyometrię.', 'warning', 5)
        else:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — niska. Priorytet: praca nad techniką biegu.', 'negative', 4)

    # Lactate dynamics
    la_peak = ct.get('_interp_la_peak')
    if la_peak:
        if la_peak >= 12:
            add(f'Laktat peak {la_peak:.1f} mmol/L — wysoka zdolność buforowa i tolerancja kwasicy.', 'positive', 6)
        elif la_peak >= 8:
            add(f'Laktat peak {la_peak:.1f} mmol/L — dobra odpowiedź glikolityczna.', 'neutral', 6)

    # HR response
    hr_max = ct.get('_interp_hr_max')
    _ag = ct.get('_interp_age', 30)
    try: hr_m = float(hr_max) if hr_max else None
    except: hr_m = None
    try: age_f = float(_ag)
    except: age_f = 30
    if hr_m and age_f:
        hr_pred = 220 - age_f
        hr_pct = hr_m / hr_pred * 100
        if hr_pct > 105:
            add(f'HRmax {hr_m:.0f} bpm ({hr_pct:.0f}% predykcji) — znacząco powyżej normy wiekowej.', 'neutral', 7)
        elif hr_pct < 85 and ct.get('_interp_test_maximal'):
            add(f'HRmax {hr_m:.0f} bpm ({hr_pct:.0f}% predykcji) — chronotropowa niekompetencja? Rozważ ocenę kardiologiczną.', 'warning', 3)


    # ─── SPORT CONTEXT: BREATHING PATTERN ANALYSIS ──────────────────
    # References:
    #   Folinsbee et al. 1983 — elite cyclists BF 63/min, VT ~50% FVC
    #   ACC (2021) — elite RR 60-70, BR <10-15%, VE >200 L/min normal
    #   HUNT3 (2014) — VE men 20-29: 141.9±24.5, VT 2.94±0.46 L
    #   NOODLE (2024) — EA VE/VCO2 slope: M 26.1±2.0, F 27.7±2.6
    #   Carey et al. 2008 — triathletes: 86% use BF-dominant pattern, only 14% VT-dominant
    #   Naranjo et al. 2005 — athlete breathing nomogram: VT vs BF curvilinear relation
    # Sport-specific VO2max ranges (ml/kg/min):
    #   Runner recreational: 40-50, competitive: 50-60, elite: 60-80+
    #   CrossFit competitive: 45-55, elite: 50-60+
    #   HYROX recreational: 38-48, competitive: 45-55
    #   Cycling competitive: 50-65, elite: 65-85+
    #   Triathlon competitive: 55-65, elite: 65-80+

    e07 = ct.get('_e07_raw', {})
    if e07 and e07.get('status') == 'OK':
        # Read VO2max directly from ct
        _bp_vo2 = ct.get('_interp_vo2_mlkgmin')
        # Read VE/VCO2 slope directly from ct (ve_val may not be in scope here)
        _bp_ve_slope = None
        try:
            _ve_sl_raw = ct.get('VE_VCO2_slope')
            if _ve_sl_raw and str(_ve_sl_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_slope = float(_ve_sl_raw)
        except: pass
        _bp_strategy = e07.get('strategy', '')
        _bp_bf_peak = e07.get('bf_peak')
        _bp_vt_peak = e07.get('vt_peak_L')
        _bp_ve_peak = None
        try:
            _bp_ve_peak_raw = ct.get('VEpeak')
            if _bp_ve_peak_raw and str(_bp_ve_peak_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_peak = float(_bp_ve_peak_raw)
        except: pass
        _bp_vdvt_rest = e07.get('vdvt_rest')
        _bp_vdvt_peak = e07.get('vdvt_peak')
        _bp_flags = e07.get('flags', [])
        _bp_bf_vt_ratio = e07.get('ve_from_bf_pct', 50)

        # Determine if athlete context applies
        _is_athlete = (_bp_vo2 is not None and _bp_vo2 >= 45)
        _is_vent_efficient = (_bp_ve_slope is not None and _bp_ve_slope < 30)
        _has_sport_context = _is_athlete and _is_vent_efficient

        # Sport tier classification
        _sport_tier = 'sedentary'
        if _bp_vo2:
            _sex_bp = ct.get('_interp_sex', 'male')
            if _sex_bp == 'male':
                if _bp_vo2 >= 65: _sport_tier = 'elite_endurance'
                elif _bp_vo2 >= 55: _sport_tier = 'competitive_endurance'
                elif _bp_vo2 >= 45: _sport_tier = 'trained'
                elif _bp_vo2 >= 35: _sport_tier = 'recreational'
            else:
                if _bp_vo2 >= 55: _sport_tier = 'elite_endurance'
                elif _bp_vo2 >= 45: _sport_tier = 'competitive_endurance'
                elif _bp_vo2 >= 38: _sport_tier = 'trained'
                elif _bp_vo2 >= 30: _sport_tier = 'recreational'

        # Reference norms by tier (M/F combined approximate)
        _bp_norms = {
            'sedentary':             {'bf_peak': (35,45), 'vt_peak': (1.8,2.5), 've_peak': (70,110),  'bf_dom_normal': False},
            'recreational':          {'bf_peak': (38,50), 'vt_peak': (2.2,3.0), 've_peak': (90,130),  'bf_dom_normal': False},
            'trained':               {'bf_peak': (42,55), 'vt_peak': (2.5,3.2), 've_peak': (110,155), 'bf_dom_normal': True},
            'competitive_endurance': {'bf_peak': (45,60), 'vt_peak': (2.7,3.5), 've_peak': (130,175), 'bf_dom_normal': True},
            'elite_endurance':       {'bf_peak': (55,70), 'vt_peak': (2.9,4.0), 've_peak': (160,200), 'bf_dom_normal': True},
        }
        _norms = _bp_norms.get(_sport_tier, _bp_norms['sedentary'])

        # Generate sport-contextualized breathing observations
        if _has_sport_context:
            _tier_pl = {'elite_endurance':'elitarny wytrzymałościowy', 'competitive_endurance':'startujący wytrzymałościowy',
                        'trained':'wytrenowanego', 'recreational':'rekreacyjnego'}.get(_sport_tier, _sport_tier)

            # BF-dominant strategy reinterpretation
            if _bp_strategy == 'BF_DOMINANT' and _norms['bf_dom_normal']:
                add(f'Wzorzec BF-dominant ({_bp_bf_vt_ratio:.0f}% wentylacji z częstości) — typowa adaptacja sportowa u zawodnika '
                    f'{_tier_pl}. Kolarze elitarni osiągają BF 60-70/min (Folinsbee 1983, ACC 2021). '
                    f'86% wytrzymałościowców stosuje ten wzorzec (Carey 2008).', 'neutral', 7)
            elif _bp_strategy == 'BF_DOMINANT' and not _norms['bf_dom_normal']:
                add(f'Wzorzec BF-dominant ({_bp_bf_vt_ratio:.0f}% z częstości) — u zawodnika rekreacyjnego może wskazywać na '
                    f'ograniczenie mechaniczne lub niedostateczny trening mięśni oddechowych. '
                    f'Rozważ trening oddechowy (IMT/RMT).', 'warning', 7)

            # VD/VT reinterpretation for athletes with very low baseline
            if _bp_vdvt_rest is not None and _bp_vdvt_peak is not None:
                try:
                    vdvt_r = float(_bp_vdvt_rest)
                    vdvt_p = float(_bp_vdvt_peak)
                    if vdvt_r < 0.20 and vdvt_p < 0.20:
                        add(f'VD/VT {vdvt_r:.3f}→{vdvt_p:.3f} — wyjątkowo niska martwa przestrzeń wentylacyjna. '
                            f'Norma kliniczna 0.25-0.35. Wartości poniżej 0.20 wskazują na elitarną efektywność '
                            f'wentylacyjną. Ewentualny wzrost VD/VT w tym zakresie NIE jest patologiczny.', 'positive', 7)
                    elif vdvt_p > vdvt_r and (vdvt_p - vdvt_r) > 0.03:
                        if vdvt_p < 0.25:
                            add(f'VD/VT wzrost {vdvt_r:.3f}→{vdvt_p:.3f} — mimo wzrostu, wartości bezwzględne w normie '
                                f'(<0.25). Wzorzec „paradoxical rise" jest klinicznie nieistotny przy niskim baseline.', 'neutral', 8)
                except: pass

            # BF peak contextualization
            if _bp_bf_peak:
                try:
                    bf = float(_bp_bf_peak)
                    bf_lo, bf_hi = _norms['bf_peak']
                    if bf >= 55 and _sport_tier in ('competitive_endurance', 'elite_endurance'):
                        add(f'BF peak {bf:.0f}/min — w normie dla poziomu {_tier_pl} (ref: {bf_lo}-{bf_hi}/min). '
                            f'Elitarni sportowcy RR 60-70/min (ACC 2021).', 'neutral', 8)
                    elif bf > bf_hi:
                        add(f'BF peak {bf:.0f}/min — powyżej typowego zakresu ({bf_lo}-{bf_hi}/min) dla poziomu {_tier_pl}. '
                            f'Rozważ ocenę rezerwy oddechowej i mechaniki wentylacji.', 'warning', 8)
                except: pass

            # VE peak contextualization
            if _bp_ve_peak:
                ve_lo, ve_hi = _norms['ve_peak']
                if _bp_ve_peak > ve_hi * 1.1:
                    add(f'VE peak {_bp_ve_peak:.0f} L/min — bardzo wysoka. Ref {_tier_pl}: {ve_lo}-{ve_hi} L/min. '
                        f'Sprawdź rezerwę oddechową (BR). BR <15% jest normalne u sportowców (ACC 2021).', 'neutral', 8)
                elif _bp_ve_peak >= ve_lo:
                    add(f'VE peak {_bp_ve_peak:.0f} L/min — adekwatna do poziomu {_tier_pl} (ref: {ve_lo}-{ve_hi} L/min).', 'positive', 9)

            # Flag reinterpretation summary
            _clinical_flags = [f for f in _bp_flags if f in ('BF_DOMINANT_STRAT', 'EARLY_VT_PLATEAU', 'VDVT_NO_DECREASE', 'VDVT_PARADOXICAL_RISE', 'HIGH_BF_VAR_REST', 'HIGH_BF_VAR_EX')]
            if _clinical_flags and _has_sport_context:
                add(f'Flagi oddechowe [{", ".join(_clinical_flags)}] — reinterpretacja sportowa: '
                    f'przy VO2max {_bp_vo2:.1f} ml/kg/min i VE/VCO2 slope {_bp_ve_slope:.1f} (<30) '
                    f'te wzorce stanowią ADAPTACJĘ SPORTOWĄ, nie patologię. '
                    f'Kontekst kliniczny wymaga VE/VCO2 >34 i/lub VD/VT >0.30.', 'neutral', 7, '🏃')

        else:
            # Non-athlete: standard breathing pattern observations
            if _bp_strategy == 'BF_DOMINANT':
                add(f'Wzorzec oddechowy BF-dominant ({_bp_bf_vt_ratio:.0f}% z częstości) — '
                    f'wentylacja głównie przez częstość oddechów. Może zwiększać martwą przestrzeń. '
                    f'Rozważ trening oddychania przeponowego.', 'warning', 7)

            if _bp_vdvt_rest is not None and _bp_vdvt_peak is not None:
                try:
                    vdvt_r = float(_bp_vdvt_rest)
                    vdvt_p = float(_bp_vdvt_peak)
                    if vdvt_p > 0.30:
                        add(f'VD/VT peak {vdvt_p:.3f} — podwyższony (norma <0.30). Obniżona efektywność wentylacyjna.', 'warning', 7)
                    elif vdvt_p > vdvt_r and (vdvt_p - vdvt_r) > 0.05 and vdvt_p > 0.25:
                        add(f'VD/VT wzrost {vdvt_r:.3f}→{vdvt_p:.3f} — paradoksalny wzrost. Rozważ ocenę kardiologiczną.', 'warning', 7)
                except: pass

    obs.sort(key=lambda x: x['priority'])
    return obs



"""
Unified Profile Scoring Engine for CPET Reports.
Single source of truth for all scoring, grading, limiter/superpower detection.
Used by both PRO and LITE reports.
"""

"""
Unified Profile Scoring Engine for CPET Reports.
Single source of truth for all scoring, grading, limiter/superpower detection.
Used by both PRO and LITE reports.
"""

def compute_profile_scores(ct):
    """
    Compute unified profile scores from canon_table.
    Returns dict with: categories, overall, grade, limiter, superpower, flags, gauge_scores.
    """
    g = ct.get

    def _sf(val, default=None):
        """Safe float conversion."""
        try: return float(val) if val not in (None, '', '-', '[BRAK]', 'None', 'nan') else default
        except: return default

    # ─── EXTRACT INPUTS ───
    e15 = g('_e15_raw', {})
    e10 = g('_e10_raw', {})
    e08 = g('_e08_raw', {})
    e03 = g('_e03_raw', {})
    e06 = g('_e06_raw', {})
    e05 = g('_e05_raw', {})
    e07 = g('_e07_raw', {})
    e16 = g('_e16_raw', {})
    _pc = g('_performance_context', {})
    modality = g('modality', 'run')

    vo2_rel = g('VO2max_ml_kg_min')
    vo2_pctile = e15.get('vo2_percentile_approx') or g('vo2_percentile_approx')
    vo2_pct_pred = g('VO2_pct_predicted') or e15.get('vo2_pct_predicted')
    vo2_class = e15.get('vo2_class_pop', '')
    vo2_class_sport = e15.get('vo2_class_sport_desc', '')
    sport_class_raw = e15.get('vo2_class_sport', '')  # UNTRAINED/RECREATIONAL/TRAINED/COMPETITIVE/SUB_ELITE/ELITE
    vo2_det = e15.get('vo2_determination', '') or g('vo2_determination', '')

    vt1_pct = g('VT1_pct_VO2peak')
    vt2_pct = g('VT2_pct_VO2peak')
    vt1_speed = g('VT1_Speed')
    vt2_speed = g('VT2_Speed')
    thr_gap = e16.get('threshold_gap_bpm') or g('threshold_gap_bpm')

    ve_vco2_slope = g('VE_VCO2_slope') or e03.get('slope_to_vt2')
    vent_class = e03.get('ventilatory_class', '')

    gain_z = g('GAIN_z') or e06.get('gain_z_score')
    re_mlkgkm = g('RE_mlkgkm') or e06.get('re_at_vt2')
    re_meas_speed = g('re_measurement_speed_kmh') or e06.get('re_measurement_speed_kmh') or e06.get('load_at_vt2')

    o2p_pct = e05.get('pct_predicted_friend') or g('O2pulse_pct_predicted')
    o2p_trajectory = g('O2pulse_trajectory')
    o2p_ff = g('O2pulse_ff')

    hrr1 = e08.get('hrr_1min') or g('HRR_1min')

    fatmax = e10.get('mfo_gmin')
    fatmax_pct_vo2 = e10.get('fatmax_pct_vo2peak')
    cop_pct_vo2 = g('COP_pct_vo2peak') or e10.get('cop_pct_vo2peak')

    # FATmax protocol confidence (Chrzanowski-Smith 2018, SJMSS; Amaro-Gahete 2019)
    # Gold standard: ≥3-min step protocol (ICC 0.94). Ramp/2-min: ICC 0.58-0.76.
    # Detect from protocol_name: RAMP → low, STEP with ≥3min → high
    _prot_name_sc = (g('protocol_name') or g('_prot_name') or '').upper()
    if any(k in _prot_name_sc for k in ('FATMAX', 'STEP_3', 'STEP_4', 'STEP_5')):
        _fatmax_confidence = 1.0   # dedicated FATmax protocol or long steps
    elif 'STEP' in _prot_name_sc:
        _fatmax_confidence = 0.65  # step protocol but likely 2-min
    else:
        _fatmax_confidence = 0.45  # ramp or unknown → approximate

    bf_peak = _sf(e07.get('bf_peak')) if e07 else None
    breathing_flags = e07.get('flags', []) if e07 else []

    # ─── PERCENTILE ───
    try: pctile_val = float(vo2_pctile) if vo2_pctile else 50
    except: pctile_val = 50

    _is_athlete = sport_class_raw in ('TRAINED', 'COMPETITIVE', 'SUB_ELITE', 'ELITE')
    # Also treat as athlete if population fitness is high (SUPERIOR/EXCELLENT or ≥80th pctile)
    # A 95th-percentile person IS athletic even if sport-specific class is RECREATIONAL
    _pop_class = e15.get('vo2_class_pop', '')
    if not _is_athlete and (pctile_val >= 80 or _pop_class in ('SUPERIOR', 'EXCELLENT')):
        _is_athlete = True
    # Sport context label for human-readable texts
    _sport_label_map = {'ELITE': 'elitarny', 'SUB_ELITE': 'subelitarny', 'COMPETITIVE': 'zawodniczy', 'TRAINED': 'wytrenowany', 'RECREATIONAL': 'rekreacyjny', 'UNTRAINED': 'początkujący'}
    _sport_lbl = _sport_label_map.get(sport_class_raw, '')
    _vo2_ctx = f'poziom {_sport_lbl}' if _sport_lbl else f'~{pctile_val:.0f} percentyl populacyjny'

    # ═══════════════════════════════════════════════════════
    # CATEGORY SCORING (10 categories, each 0-100)
    # ═══════════════════════════════════════════════════════
    _cat = {}

    # ── 1. VO2max CEILING ──
    # Blend population percentile with sport class for balanced scoring
    _sport_score_map = {'ELITE': 95, 'SUB_ELITE': 85, 'COMPETITIVE': 72, 'TRAINED': 58, 'RECREATIONAL': 38, 'UNTRAINED': 20}
    _sport_sc = _sport_score_map.get(sport_class_raw, None)
    if _sport_sc is not None:
        # Weighted blend: 40% pop percentile + 60% sport level
        _vo2_sc = min(100, max(0, pctile_val * 0.4 + _sport_sc * 0.6))
    else:
        # No sport class available — fall back to pop percentile only
        _vo2_sc = min(100, max(0, pctile_val))
    _vo2_pp = _sf(vo2_pct_pred)
    if _vo2_pp is not None and _vo2_pp < 85:
        _vo2_sc = min(_vo2_sc, _vo2_pp * 0.9)
    _vo2_v = _sf(vo2_rel, 0)
    _cat['vo2max'] = {
        'score': _vo2_sc,
        'label': 'Wydolność tlenowa',
        'icon': '🏋',
        'limiter_text': f'VO₂max ({_vo2_v:.1f} ml/kg/min) ogranicza Twoją wydolność. Więcej treningów interwałowych VO₂max (3-5 min @ 95-100% HRmax) i zwiększenie objętości treningowej pomogą przebić ten sufit.',
        'super_text': f'VO₂max ({_vo2_v:.1f} ml/kg/min, ~{pctile_val:.0f} percentyl) to Twój fundament — masz silny silnik tlenowy, na którym możesz budować resztę.',
        'tip': 'Interwały VO₂max + objętość Z2'
    }

    # ── 2. VT2 THRESHOLD ──
    _vt2p = _sf(vt2_pct, 75)
    # Sigmoid scoring: plateau in 75-85% range (physiological sweet spot),
    # steep drop below 70%, gentle gain above 85%
    import math
    _vt2_sc = max(0, min(100, 100 / (1 + math.exp(-0.18 * (_vt2p - 74)))))
    # Sport-specific VT2 expectation for athletes
    _vt2_sport_thr = {'run':80,'bike':78,'triathlon':80,'rowing':78,'crossfit':72,'hyrox':75,'swimming':76,'xc_ski':84,'soccer':78,'mma':72}.get(modality, 80)
    if _is_athlete and _vt2p < _vt2_sport_thr:
        _vt2_sc *= 0.85
    _cat['vt2'] = {
        'score': _vt2_sc,
        'label': 'Próg mleczanowy',
        'icon': '⚡',
        'limiter_text': f'Próg mleczanowy (VT2) przy {_vt2p:.0f}% VO₂max — organizm zaczyna "kwasić się" relatywnie wcześnie. Trening tempo (Z3-Z4) i interwały progowe podniosą ten próg.',
        'super_text': f'Próg mleczanowy (VT2) aż przy {_vt2p:.0f}% VO₂max — możesz utrzymywać wysokie tempo przez długi czas zanim organizm się "zakwasi".',
        'tip': 'Trening tempo Z3-Z4 + interwały SST'
    }

    # ── 3. VT1 AEROBIC BASE ──
    _vt1p = _sf(vt1_pct, 55)
    _vt1_sc = max(0, min(100, (_vt1p - 40) / 35 * 100))
    _gap = _sf(thr_gap, 20)
    if _gap < 12:
        _vt1_sc *= 0.8

    _vt1_trap = False
    # Ceiling trap: high VT1% but low absolute fitness — sport-specific thresholds
    _vt1_trap_thr = {'run':70,'bike':68,'triathlon':70,'rowing':68,'crossfit':62,'hyrox':65,'swimming':66,'xc_ski':74,'soccer':68,'mma':62}.get(modality, 70)
    _vt1_trained_thr = _vt1_trap_thr + 8  # higher bar for TRAINED athletes
    if sport_class_raw in ('UNTRAINED', 'RECREATIONAL') and _vt1p >= _vt1_trap_thr:
        _vt1_sc *= 0.6
        _vt1_trap = True
    elif sport_class_raw == 'TRAINED' and _vt1p >= _vt1_trained_thr:
        _vt1_sc *= 0.8
        _vt1_trap = True
    elif pctile_val < 50 and _vt1p >= _vt1_trap_thr:
        # Fallback: no sport class, use pop pctile
        _vt1_sc *= 0.65
        _vt1_trap = True

    _vt1_spd = _sf(vt1_speed, 0)
    if g('test_device', 'treadmill') == 'treadmill' and 0 < _vt1_spd < 9.5 and _vt1p >= 70:
        _vt1_sc *= 0.8
        _vt1_trap = True

    if _vt1_trap:
        _vt1_lim = f'VT1 przy {_vt1p:.0f}% VO₂max wygląda dobrze procentowo, ale absolutna wydolność ({_vo2_ctx}' + (f', VT1 przy {_vt1_spd:.1f} km/h' if _vt1_spd > 0 else '') + ') jest niska. Buduj sufit tlenowy — baza pójdzie w górę automatycznie.'
        _vt1_sup = ''
    else:
        _vt1_lim = f'Próg tlenowy (VT1) przy {_vt1p:.0f}% VO₂max' + (f' z wąskim gapem VT2-VT1 ({_gap:.0f} bpm)' if _gap < 15 else '') + ' — baza aerobowa wymaga wzmocnienia. Więcej długich, spokojnych treningów w Z2.'
        _vt1_sup = f'Solidna baza tlenowa — VT1 przy {_vt1p:.0f}% VO₂max' + (f' z szerokim gapem {_gap:.0f} bpm między progami' if _gap >= 25 else '') + '. Twój fundament aerobowy jest mocny.'

    _cat['vt1'] = {
        'score': _vt1_sc,
        'label': 'Baza tlenowa',
        'icon': '💚',
        'limiter_text': _vt1_lim,
        'super_text': _vt1_sup,
        'tip': 'Długie biegi Z2 + objętość'
    }

    # ── 4. ECONOMY ──
    _gz = _sf(gain_z, 0)
    _re_v = _sf(re_mlkgkm)
    _re_info = f' (RE: {_re_v:.0f} ml/kg/km)' if _re_v else ''
    _e06_flags = e06.get('flags', []) if isinstance(e06, dict) else []
    _e06_no_data = 'INSUFFICIENT_DATA' in _e06_flags and not _re_v and not _gz
    # test_device determines RE vs GAIN (physics of the test, not the sport)
    # treadmill → RE is always valid regardless of sport (crossfit on treadmill → RE)
    _test_device = g('test_device', 'treadmill')
    _use_re = _test_device == 'treadmill' and _re_v and _re_v > 0

    # RE-based scoring when test is on treadmill (lower RE = better economy)
    if _use_re:
        # If RE measured at very low speed (<8 km/h), it's walk-to-run transition
        # and RE is unreliable (engine flags RE_LOW_SPEED_UNRELIABLE)
        _re_speed = _sf(re_meas_speed, 12)
        if _re_speed < 8:
            _use_re = False  # fall through to GAIN-only or neutral

    if _use_re:
        # Speed-adjusted RE thresholds (literature: RE improves with speed due to
        # walk-run transition costs being amortized; RE at 10 km/h is naturally worse
        # than at 14+ km/h). Adjust norms: +5 ml/kg/km for each km/h below 12.
        _re_speed = _sf(re_meas_speed, 12)
        _speed_adj = max(0, (12 - _re_speed) * 5) if _re_speed < 12 else 0
        _re_adj = _re_v - _speed_adj  # adjusted RE (as if measured at 12+ km/h)
        if _re_adj < 175: _re_sc = 95
        elif _re_adj < 185: _re_sc = 80 + (185 - _re_adj) / 10 * 15
        elif _re_adj < 200: _re_sc = 60 + (200 - _re_adj) / 15 * 20
        elif _re_adj < 215: _re_sc = 40 + (215 - _re_adj) / 15 * 20
        elif _re_adj < 235: _re_sc = 20 + (235 - _re_adj) / 20 * 20
        else: _re_sc = max(5, 20 - (_re_adj - 235) / 20 * 15)
        # Blend with GAIN_z when available
        # GAIN_z reliability depends on speed diversity between VT1 and VT2:
        # If speed barely changed (protocol saturation), GAIN is unreliable
        # Also: when RE is elite (<185), GAIN should have minimal impact
        _gz_sc = max(15, min(100, 50 + _gz * 22)) if _gz else None
        if _gz_sc is not None:
            # Speed diversity factor: GAIN is only meaningful when speed increased
            _vt1_spd_e = _sf(vt1_speed, 0)
            _vt2_spd_e = _sf(vt2_speed, 0)
            if _vt1_spd_e > 4 and _vt2_spd_e > 4:
                _spd_ratio = _vt2_spd_e / _vt1_spd_e
                _diversity = min(1.0, max(0.0, (_spd_ratio - 1.0) / 0.20))
            else:
                _diversity = 0.5  # unknown speeds → moderate weight
            # Base weight: lower for RECREATIONAL/UNTRAINED (confounded with fitness)
            _sport_trained = sport_class_raw in ('TRAINED', 'COMPETITIVE', 'SUB_ELITE', 'ELITE')
            _base_gz_w = 0.40 if _sport_trained else 0.20
            # Reduce by speed diversity (saturated protocol → low GAIN weight)
            _gz_w = _base_gz_w * _diversity
            # Elite RE cap: when RE is excellent, don't let GAIN drag it down
            if _re_adj < 185:
                _gz_w = min(_gz_w, 0.15)
            _econ_sc = (1 - _gz_w) * _re_sc + _gz_w * _gz_sc
        else:
            _econ_sc = _re_sc
        _re_ctx = ' (test na bieżni)' if modality != 'run' else ''
        _econ_label = f'Ekonomia biegu{_re_ctx}'
        # Limiter text should reflect ACTUAL RE quality, not blended score
        if _re_v < 185:
            _econ_lim = f'Ekonomia biegu{_re_ctx} na dobrym poziomie (RE: {_re_v:.0f} ml/kg/km — elitarna){f", z-score: {_gz:+.1f}" if _gz else ""}. Efektywność O₂/km wysoka, ale inne kategorie wymagają uwagi.'
        elif _re_v < 200:
            _econ_lim = f'Ekonomia biegu{_re_ctx} do poprawy (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""}. Plyometria, trening siłowy i ćwiczenia techniczne poprawią efektywność.'
        else:
            _econ_lim = f'Ekonomia biegu{_re_ctx} poniżej przeciętnej (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""}. Zużywasz dużo energii na danym tempie. Plyometria, trening siłowy i ćwiczenia techniczne poprawią efektywność.'
        _econ_sup = f'Doskonała ekonomia biegu{_re_ctx} (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""} — Twój organizm zużywa mniej energii na danym tempie niż przeciętna osoba.'
    else:
        # GAIN_z-based for cycling or when RE unavailable
        _econ_sc = max(0, min(100, 50 + _gz * 22)) if _gz else 50
        _mod_names = {'bike':'pedałowania','rowing':'wioślarstwa','swimming':'pływania','triathlon':'ruchu','crossfit':'ruchu','hyrox':'ruchu'}
        _mod_name = _mod_names.get(modality, 'ruchu')
        _mod_tips = {'bike':'Praca nad kadencją, pozycją i siłą specjalną poprawi efektywność.',
                     'rowing':'Trening techniczny wioślarstwa i siła specjalna poprawią efektywność.',
                     'swimming':'Drills techniczne i trening siły specyficznej poprawią efektywność.'}
        _mod_tip = _mod_tips.get(modality, 'Trening techniczny i siłowy poprawi efektywność.')
        _econ_label = f'Ekonomia {_mod_name}'
        _econ_lim = f'Ekonomia {_mod_name} poniżej przeciętnej (z-score: {_gz:+.1f}){_re_info}. {_mod_tip}'
        _econ_sup = f'Doskonała ekonomia {_mod_name} (z-score: {_gz:+.1f}){_re_info} — Twój organizm zużywa mniej energii na danym obciążeniu niż przeciętna osoba.'

    _econ_sc = max(0, min(100, _econ_sc))
    _econ_tip_map = {
        'run': 'Plyometria + trening siłowy + technika biegu',
        'bike': 'Kadencja + siła specjalna + pozycja na rowerze',
        'swimming': 'Technika + drills + trening paddles',
        'rowing': 'Technika wioślarstwa + siła + rate control',
    }
    _econ_tip = _econ_tip_map.get(modality, 'Trening techniczny + siła specjalna')
    if _e06_no_data:
        _econ_sc = None  # No speed/power data → exclude from scoring (not neutral 50)
        _econ_lim = f'Brak danych prędkości/mocy w pliku CPET — ekonomia ruchu nie mogła być obiektywnie oceniona. Aby uzyskać pełną analizę, wykonaj test z pomiarem prędkości lub mocy.'
        _econ_sup = _econ_lim
        _econ_tip = 'Powtórz test z pomiarem prędkości/mocy'
    _cat['economy'] = {
        'score': _econ_sc,
        'label': _econ_label,
        'icon': '⚙️',
        'limiter_text': _econ_lim,
        'super_text': _econ_sup,
        'tip': _econ_tip,
        'no_data': _e06_no_data
    }

    # ── 5. VENTILATORY EFFICIENCY ──
    _slp = _sf(ve_vco2_slope, 30)
    if _slp <= 20: _vent_sc = 98
    elif _slp <= 25: _vent_sc = 85 + (25 - _slp) / 5 * 13
    elif _slp <= 28: _vent_sc = 72 + (28 - _slp) / 3 * 13
    elif _slp <= 30: _vent_sc = 60 + (30 - _slp) / 2 * 12
    elif _slp <= 34: _vent_sc = 35 + (34 - _slp) / 4 * 25
    else: _vent_sc = max(5, 35 - (_slp - 34) / 6 * 30)
    _vent_sc = max(0, min(100, _vent_sc))
    # Ventilation limiter text: severity-dependent
    if _slp >= 34:
        _vent_lim = f'Efektywność wentylacyjna istotnie ograniczona (VE/VCO₂ slope: {_slp:.1f}). Oddychasz znacząco więcej niż potrzeba na daną produkcję CO₂. Trening oddechowy i konsultacja ze specjalistą.'
    elif _slp >= 30:
        _vent_lim = f'Efektywność wentylacyjna umiarkowanie obniżona (VE/VCO₂ slope: {_slp:.1f}). Trening oddechowy i interwały tlenowe mogą poprawić ekonomię oddychania.'
    else:
        _vent_lim = f'Efektywność wentylacyjna w normie, ale najniższa z Twoich kategorii (VE/VCO₂ slope: {_slp:.1f}). To nie jest istotne ograniczenie — Twoja wentylacja działa prawidłowo. Marginalny potencjał poprawy przez trening oddechowy.'
    _cat['ventilation'] = {
        'score': _vent_sc,
        'label': 'Wentylacja',
        'icon': '💨',
        'limiter_text': _vent_lim,
        'super_text': f'Wyjątkowo efektywne oddychanie (VE/VCO₂ slope: {_slp:.1f}) — Twoje płuca doskonale radzą sobie z wymianą gazową przy minimalnym wysiłku wentylacyjnym.',
        'tip': 'Trening oddechowy + technika oddychania'
    }

    # ── 6. CARDIAC / O2 PULSE ──
    _o2p = _sf(o2p_pct)
    if _o2p is not None:
        _o2p_sc = min(100, max(0, _o2p))
        _traj = str(o2p_trajectory).lower() if o2p_trajectory else ''
        _ff_v = _sf(o2p_ff)
        if 'plateau' in _traj or 'flat' in _traj:
            if _is_athlete and _o2p >= 120:
                pass
            else:
                _o2p_sc *= 0.85
        if _ff_v is not None and _ff_v < 0.5 and _o2p < 110:
            _o2p_sc *= 0.9
        _cat['cardiac'] = {
            'score': _o2p_sc,
            'label': 'Serce (O\u2082 pulse)',
            'icon': '\u2764\ufe0f',
            'limiter_text': f'O\u2082 pulse na {_o2p:.0f}% normy' + (' z p\u0142askim przebiegiem' if 'plateau' in _traj or 'flat' in _traj else '') + ' \u2014 sugeruje mniejsz\u0105 obj\u0119to\u015b\u0107 wyrzutow\u0105 serca. Interwa\u0142y VO\u2082max i trening wytrzyma\u0142o\u015bciowy poprawi\u0105 serce sportowe.',
            'super_text': f'Silne serce sportowe \u2014 O\u2082 pulse na {_o2p:.0f}% normy przewidywanej (wg Wassermana, dla Twojego wieku, p\u0142ci i masy cia\u0142a). Ka\u017cde uderzenie serca dostarcza du\u017co tlenu do mi\u0119\u015bni.',
            'tip': 'Interwa\u0142y VO\u2082max + d\u0142ugi trening Z2'
        }

    # ── 7. RECOVERY ──
    _hrr_v = _sf(hrr1, 25)
    _rec_mode = g('rec_recovery_mode', 'auto')
    # Active recovery (cooldown walking) typically yields HRR 5-10 bpm LOWER
    # than passive (sudden stop). Adjust HRR upward for active recovery protocol.
    if _rec_mode == 'active' and _hrr_v < 40:
        _hrr_v_adj = _hrr_v + 8  # compensate for active cooldown
    else:
        _hrr_v_adj = _hrr_v
    if _hrr_v_adj >= 50: _hrr_sc = 95
    elif _hrr_v_adj >= 40: _hrr_sc = 85 + (_hrr_v_adj - 40) / 10 * 10
    elif _hrr_v_adj >= 30: _hrr_sc = 70 + (_hrr_v_adj - 30) / 10 * 15
    elif _hrr_v_adj >= 22: _hrr_sc = 55 + (_hrr_v_adj - 22) / 8 * 15
    elif _hrr_v_adj >= 15: _hrr_sc = 38 + (_hrr_v_adj - 15) / 7 * 17
    elif _hrr_v_adj >= 8: _hrr_sc = 15 + (_hrr_v_adj - 8) / 7 * 23
    else: _hrr_sc = max(5, _hrr_v_adj / 8 * 15)
    _hrr_sc = max(0, min(100, _hrr_sc))
    # Athletes should have better recovery — graduated penalty
    # HRR depends on protocol (active/passive), position, HR strap lag
    # so penalty is mild for borderline cases
    if _is_athlete:
        if _hrr_v < 15:
            _hrr_sc *= 0.75   # serious concern
        elif _hrr_v < 20:
            _hrr_sc *= 0.85   # below expected for athlete
        elif _hrr_v < 28:
            _hrr_sc *= 0.93   # mild: borderline, protocol-dependent
    if _hrr_v < 20:
        _rec_lim = f'Regeneracja istotnie osłabiona (HRR₁ / Heart Rate Recovery — spadek tętna w 1. min po wysiłku: {_hrr_v:.0f} bpm). Norma: >25 bpm. Tętno spada bardzo wolno po wysiłku. Zadbaj o sen, nawodnienie, trening Z1-Z2 i periodyzację. Rozważ konsultację.'
    elif _hrr_v < 28:
        _rec_lim = f'Regeneracja nieco poniżej normy sportowej (HRR₁ / Heart Rate Recovery — spadek tętna w 1. min po wysiłku: {_hrr_v:.0f} bpm). Norma: >25 bpm. Może wynikać z protokołu testu, zmęczenia lub stylu życia. Zadbaj o sen, nawodnienie i periodyzację.'
    else:
        _rec_lim = f'Regeneracja w normie, ale najniższa z Twoich kategorii (HRR₁ / Heart Rate Recovery — spadek tętna w 1. min po wysiłku: {_hrr_v:.0f} bpm). Norma: >25 bpm, >40 = bardzo dobrze. Potencjał poprawy przez lepszą higienę snu i periodyzację.'
    _cat['recovery'] = {
        'score': _hrr_sc,
        'label': 'Regeneracja',
        'icon': '🔄',
        'limiter_text': _rec_lim,
        'super_text': f'Błyskawiczna regeneracja (HRR₁ / Heart Rate Recovery — spadek tętna w 1. min po wysiłku: {_hrr_v:.0f} bpm) — Twój układ nerwowy i serce szybko wracają do normy po wysiłku.',
        'tip': 'Sen + nawodnienie + periodyzacja'
    }

    # ── 8. SUBSTRATE / FAT OXIDATION ──
    _fat_v = _sf(fatmax, 0)  # g/min
    _fat_pct = _sf(fatmax_pct_vo2, 45)
    _cop_v = _sf(cop_pct_vo2, 60)
    _bm = _sf(g('body_mass_kg'), 75)
    _sub_sc = 50
    if _fat_v > 0:
        # Normalize FATmax to body mass (mg/kg/min) for fairer comparison
        # Literature: Randell 2017 uses FFM, but BM is always available
        # MFO norms (g/min): 0.61 M / 0.50 F athletes (Randell 2017)
        # ~7-8 mg/kg/min is good for trained, ~5 recreational
        _fat_norm = (_fat_v * 1000) / _bm  # mg/kg/min
        if _fat_norm >= 8.0 and _fat_pct >= 48: _sub_raw = 92
        elif _fat_norm >= 6.5 and _fat_pct >= 43: _sub_raw = 78
        elif _fat_norm >= 5.0 and _fat_pct >= 38: _sub_raw = 65
        elif _fat_norm >= 3.5: _sub_raw = 50
        else: _sub_raw = 30
        # Crossover penalty: <40% is common in sedentary (Lima-Silva 2010)
        # Only penalize when very early (<35%) suggesting metabolic inflexibility
        if _cop_v < 35: _sub_raw *= 0.8
        elif _cop_v < 42: _sub_raw *= 0.9
        # Protocol confidence: compress score toward 50 (neutral) for unreliable protocols
        # Ramp/2-min steps: MFO ICC 0.58-0.76 (Chrzanowski-Smith 2018)
        # Score = 50 + (raw - 50) * confidence → low confidence = closer to neutral
        _sub_sc = 50 + (_sub_raw - 50) * _fatmax_confidence
    else:
        _sub_sc = None

    _fatmax_approx = _fatmax_confidence < 0.8
    _approx_note = ' (orientacyjnie — protokół ramp/krótkie stopnie)' if _fatmax_approx else ''
    if _sub_sc is not None:
        _cat['substrate'] = {
            'score': _sub_sc,
            'label': 'Spalanie tłuszczów',
            'icon': '🔥',
            'confidence': _fatmax_confidence,
            'limiter_text': f'Metabolizm tłuszczowy ograniczony{_approx_note} (FATmax: {_fat_v * 60:.0f} g/h' + (f', crossover przy {_cop_v:.0f}% VO₂' if _cop_v < 55 else '') + '). ' + ('Wynik orientacyjny z protokołu ramp — dedykowany test FATmax (stopnie ≥3 min) da dokładniejszy obraz. ' if _fatmax_approx else '') + 'Więcej treningów Z2 oraz okazjonalne sesje \u201etrain low\u201d (Z1-Z2 na czczo, 1-2\u00d7/tyg.) poprawi\u0105 spalanie. NIE na treningach kluczowych.',
            'super_text': f'Dobry metabolizm tłuszczowy{_approx_note} (FATmax: {_fat_v * 60:.0f} g/h przy {_fat_pct:.0f}% VO₂max) — Twój organizm spala tłuszcze efektywnie.' + (' Wynik orientacyjny — dedykowany test potwierdzi.' if _fatmax_approx else ' Daje przewagę na długich dystansach.'),
            'tip': 'Długie Z2 + periodyzacja węglowodanów'
        }

    # ── 9. BREATHING PATTERN ──
    _bp_sc = None
    if bf_peak is not None and e07:
        # Sport-class-aware BF peak scoring
        # Elite athletes normally have BF 55-70 (Carey 2008, Naranjo 2005)
        if _is_athlete:
            # TRAINED/COMPETITIVE/ELITE — higher BF is normal adaptation
            if bf_peak < 50: _bp_sc = 92
            elif bf_peak < 58: _bp_sc = 80
            elif bf_peak < 66: _bp_sc = 68
            elif bf_peak < 73: _bp_sc = 58  # upper normal for athletes
            else: _bp_sc = 42
        else:
            # UNTRAINED/RECREATIONAL — standard thresholds
            if bf_peak < 40: _bp_sc = 90
            elif bf_peak < 48: _bp_sc = 75
            elif bf_peak < 55: _bp_sc = 60
            elif bf_peak < 65: _bp_sc = 45
            else: _bp_sc = 30
        # Only penalize for REAL clinical flags, not estimation artifacts or normal athlete adaptations
        _artifact_flags = {'VDVT_EST_ARTIFACT', 'EST_ARTIFACT', 'VT_ESTIMATED', 'VDVT_EST_LOW'}
        # In athletes: BF variability and early VT plateau are normal adaptations (Carey 2008)
        if _is_athlete:
            _artifact_flags |= {'HIGH_BF_VAR_EX', 'EARLY_VT_PLATEAU'}
        _real_flags = [f for f in (breathing_flags or []) if f not in _artifact_flags]
        if _real_flags:
            _n_flags = len(_real_flags)
            # Athletes: smaller penalty per flag (some variability is normal)
            _pen = 0.05 if _is_athlete else 0.08
            _bp_sc *= max(0.75, 1 - _n_flags * _pen)
        _bp_sc = max(0, min(100, _bp_sc))
        _cat['breathing'] = {
            'score': _bp_sc,
            'label': 'Wzorzec oddechowy',
            'icon': '🌬️',
            'limiter_text': f'Wzorzec oddechowy z ostrzeżeniami (BF peak: {bf_peak:.0f}/min' + (f', flagi: {", ".join(breathing_flags[:2])}' if breathing_flags else '') + '). Praca nad głębokim, kontrolowanym oddychaniem może poprawić efektywność.',
            'super_text': f'Efektywny wzorzec oddechowy (BF peak: {bf_peak:.0f}/min) — oddychasz głęboko i ekonomicznie, co wspiera wydolność.',
            'tip': 'Trening oddechowy + medytacja'
        }

    # ── 10. DECONDITIONING ──
    _vo2_pp_v = _sf(vo2_pct_pred, 100)
    if _vo2_pp_v is not None and _vo2_pp_v < 80 and _vent_sc > 55 and _o2p_sc > 60:
        _decon_sc = max(5, _vo2_pp_v * 0.8)
        _cat['deconditioning'] = {
            'score': _decon_sc,
            'label': 'Dekondycjonowanie',
            'icon': '🛋️',
            'limiter_text': f'VO₂max na {_vo2_pp_v:.0f}% normy przy prawidłowych proporcjach fizjologicznych — wskazuje na dekondycjonowanie. Systematyczny, stopniowy trening przyniesie szybkie efekty.',
            'super_text': '',
            'tip': 'Systematyczny trening od podstaw'
        }

    # ═══════════════════════════════════════════════════════
    # CONTEXTUAL OVERRIDES
    # ═══════════════════════════════════════════════════════
    _vt2_trap = False
    if _vt2p >= 88 and _pc.get('level_by_speed') in ('Sedentary', 'Recreational'):
        _vt2_trap = True
    elif _vt2p >= 88 and sport_class_raw in ('UNTRAINED', 'RECREATIONAL'):
        _vt2_trap = True
    elif _vt2p >= 85 and sport_class_raw in ('UNTRAINED', 'RECREATIONAL') and pctile_val < 80:
        _vt2_trap = True

    if _vt2_trap:
        _cat['vo2max']['score'] *= 0.75
        _cat['vt2']['score'] *= 0.8
        _cat['vo2max']['limiter_text'] = f'Próg wysoki ({_vt2p:.0f}% VO₂max) ale wydolność absolutna niska ({_vo2_ctx}) — prawdziwym limiterem jest niski VO₂max. Buduj sufit tlenowy interwałami i objętością.'
        _cat['vt2']['super_text'] = f'Próg przy {_vt2p:.0f}% VO₂max — dobry stosunek, ale sufit (VO₂max) wymaga podniesienia.'

    _econ_mask = False
    _sport_trained = sport_class_raw in ('TRAINED', 'COMPETITIVE', 'SUB_ELITE', 'ELITE')
    _vt1_spd_m = _sf(vt1_speed, 0)
    _vt2_spd_m = _sf(vt2_speed, 0)
    _spd_div_ok = (_vt1_spd_m > 4 and _vt2_spd_m > 4 and _vt2_spd_m / _vt1_spd_m > 1.10)
    if _sport_trained and _gz and _gz < -0.8 and _spd_div_ok:
        # Genuinely trained athlete with reliable bad economy → real technique limiter
        # Only apply when speed diversity confirms GAIN is meaningful
        if not (_use_re and _re_v < 185):  # Don't mask when RE is elite
            _cat['economy']['score'] *= 0.85
            _econ_mask = True

    # ═══════════════════════════════════════════════════════
    # FIND LIMITER & SUPERPOWER
    # ═══════════════════════════════════════════════════════
    _valid = {k: v for k, v in _cat.items() if v.get('score') is not None}

    # ── Sport demand profiles (literature-based priority multipliers) ──
    # Higher = more important for that sport's performance.
    # Joyner & Coyle 2008: VO2max+VT2+RE = "big 3" for running
    # Lucía 2001: VT2 = key differentiator in cycling
    # San Millán 2018: substrate critical for ultra-endurance
    # Laursen 2002: sport-specific training demands
    _sport_demands = {
        'run':       {'vo2max':1.0, 'vt2':0.9, 'economy':0.85, 'substrate':0.6, 'vt1':0.5, 'recovery':0.4, 'ventilation':0.3, 'cardiac':0.5, 'breathing':0.1},
        'bike':      {'vt2':1.0, 'economy':0.9, 'vo2max':0.8, 'cardiac':0.7, 'substrate':0.5, 'recovery':0.4, 'vt1':0.4, 'ventilation':0.3, 'breathing':0.1},
        'triathlon':  {'economy':1.0, 'vt2':0.9, 'vo2max':0.8, 'substrate':0.7, 'recovery':0.5, 'cardiac':0.5, 'vt1':0.4, 'ventilation':0.3, 'breathing':0.1},
        'hyrox':     {'vt2':1.0, 'recovery':0.9, 'vo2max':0.8, 'economy':0.6, 'substrate':0.5, 'vt1':0.4, 'cardiac':0.5, 'ventilation':0.3, 'breathing':0.1},
        'crossfit':  {'vt2':1.0, 'recovery':0.9, 'vo2max':0.8, 'ventilation':0.5, 'economy':0.4, 'cardiac':0.5, 'vt1':0.3, 'substrate':0.2, 'breathing':0.1},
        'mma':       {'vt2':1.0, 'recovery':0.9, 'vo2max':0.8, 'ventilation':0.5, 'economy':0.4, 'cardiac':0.5, 'vt1':0.3, 'substrate':0.2, 'breathing':0.1},
        'xc_ski':    {'vo2max':1.0, 'vt2':0.9, 'economy':0.7, 'cardiac':0.7, 'substrate':0.6, 'vt1':0.5, 'recovery':0.4, 'ventilation':0.3, 'breathing':0.1},
        'swimming':  {'economy':1.0, 'vo2max':0.8, 'vt2':0.7, 'ventilation':0.6, 'cardiac':0.5, 'vt1':0.4, 'recovery':0.4, 'substrate':0.3, 'breathing':0.1},
        'rowing':    {'economy':1.0, 'vt2':0.9, 'vo2max':0.8, 'cardiac':0.7, 'substrate':0.5, 'recovery':0.4, 'vt1':0.4, 'ventilation':0.3, 'breathing':0.1},
        'soccer':    {'vt2':1.0, 'recovery':0.9, 'vo2max':0.7, 'vt1':0.6, 'economy':0.4, 'cardiac':0.5, 'ventilation':0.3, 'substrate':0.2, 'breathing':0.1},
    }
    _demands = _sport_demands.get(modality, {'vo2max':1.0, 'vt2':0.8, 'vt1':0.5, 'economy':0.5, 'ventilation':0.3, 'cardiac':0.5, 'recovery':0.5, 'substrate':0.3, 'breathing':0.1})

    # ── Detect "no real limiter" mode ──
    # Threshold 75: below = genuinely actionable weakness, above = marginal gains only
    _min_score = min((v.get('score', 100) for v in _valid.values()), default=100)
    _no_real_limiter = _min_score >= 75

    # ── Sport-weighted limiter selection ──
    # For each category: priority = (100 - score) × sport_demand
    # This means: a small deficit in a CRITICAL sport category outweighs
    # a larger deficit in an irrelevant one.
    # Exclude 'deconditioning' from sport-demand weighting (it's a meta-category)
    # ── Priority ranking (V2: replaces binary LIMITER/SUPERMOC) ──
    # Sort ALL categories by (100-score)*demand → top = biggest training priority
    # Limiter = priority #1 (with safety redirects applied after)
    _priority_ranking = []
    for k, v in _valid.items():
        if k == 'deconditioning':
            continue
        sc = v.get('score')
        if sc is None:
            continue
        demand = _demands.get(k, 0.3)
        deficit = max(0, 100 - sc) * demand
        stars = 3 if demand >= 0.8 else (2 if demand >= 0.5 else 1)
        _priority_ranking.append({
            'key': k,
            'label': v.get('label', k),
            'icon': v.get('icon', '●'),
            'score': sc,
            'demand': demand,
            'deficit': deficit,
            'stars': stars,
            'limiter_text': v.get('limiter_text', ''),
            'super_text': v.get('super_text', ''),
            'tip': v.get('tip', ''),
        })
    _priority_ranking.sort(key=lambda x: -x['deficit'])

    # ── Limiter = priority #1 from ranking (replaces old standalone logic) ──
    if _priority_ranking:
        _limiter_key = _priority_ranking[0]['key']

    # ── Safety redirects (measurement reliability) ──

    # Economy redirect for RECREATIONAL/UNTRAINED: GAIN confounded with fitness
    if _limiter_key == 'economy' and sport_class_raw in ('UNTRAINED', 'RECREATIONAL'):
        _vo2_sc = _valid.get('vo2max', {}).get('score', 100)
        _vt1_sc = _valid.get('vt1', {}).get('score', 100)
        if _vo2_sc < 70:
            _limiter_key = 'vo2max'
        elif _vt1_sc < 65:
            _limiter_key = 'vt1'

    # FATmax redirect for unreliable protocol (ramp / <3min steps)
    if _limiter_key == 'substrate' and _fatmax_confidence < 0.8:
        _non_sub = [p for p in _priority_ranking if p['key'] != 'substrate']
        if _non_sub:
            _limiter_key = _non_sub[0]['key']

    # Breathing redirect: low-weight auxiliary, score ≥65 = normal
    if _limiter_key == 'breathing' and _valid.get('breathing', {}).get('score', 0) >= 65:
        _non_bp = [p for p in _priority_ranking if p['key'] != 'breathing']
        if _non_bp:
            _limiter_key = _non_bp[0]['key']

    _super_candidates = {k: v for k, v in _valid.items() if k != 'deconditioning' and v.get('super_text')}
    _super_key = max(_super_candidates, key=lambda k: _super_candidates[k]['score']) if _super_candidates else None

    _limiter = _valid.get(_limiter_key, {}) if _limiter_key else {}
    _superpower = _valid.get(_super_key, {}) if _super_key else {}

    # ── No-real-limiter: override limiter text to "marginal" tone ──
    if _no_real_limiter and _limiter_key and _limiter:
        _lim_label = _limiter.get('label', _limiter_key)
        _lim_sc = _limiter.get('score', 0)
        _limiter['limiter_text'] = (
            f'{_lim_label} ({_lim_sc:.0f}/100) to Twoje relatywnie najsłabsze ogniwo, '
            f'ale przy wszystkich kategoriach ≥75 nie masz wyraźnego limitera. '
            f'Profil fizjologiczny jest zbalansowany — kontynuuj obecny trening.'
        )

    # ═══════════════════════════════════════════════════════
    # OVERALL SCORE (weighted composite)
    # ═══════════════════════════════════════════════════════
    # Sport-specific category weights
    _base_weights = {'vo2max': 0.25, 'vt2': 0.20, 'vt1': 0.10, 'economy': 0.12,
                'ventilation': 0.08, 'cardiac': 0.10, 'recovery': 0.08,
                'substrate': 0.05, 'breathing': 0.02}
    _sport_weight_mods = {
        # Endurance: substrate matters more, economy critical
        'run':       {'substrate': 0.07, 'economy': 0.14},
        'xc_ski':    {'substrate': 0.08, 'economy': 0.10, 'cardiac': 0.12},
        'triathlon': {'substrate': 0.08, 'economy': 0.12},
        # Cycling: economy (watts efficiency) and cardiac critical
        'bike':      {'economy': 0.14, 'cardiac': 0.12, 'substrate': 0.06},
        'rowing':    {'economy': 0.14, 'cardiac': 0.12, 'substrate': 0.06},
        'swimming':  {'economy': 0.16, 'substrate': 0.04},
        # Mixed/glycolytic: recovery and VT2 matter more, substrate less
        'crossfit':  {'recovery': 0.12, 'vt2': 0.22, 'substrate': 0.02, 'economy': 0.08},
        'hyrox':     {'recovery': 0.10, 'substrate': 0.06, 'economy': 0.10},
        'mma':       {'recovery': 0.14, 'vt2': 0.22, 'substrate': 0.02, 'economy': 0.06},
        'soccer':    {'recovery': 0.12, 'vt2': 0.22, 'substrate': 0.03, 'economy': 0.06},
    }
    _weights = dict(_base_weights)
    _weights.update(_sport_weight_mods.get(modality, {}))
    _overall = 0
    try:
        _w_sum = 0
        for k, w in _weights.items():
            if k in _valid and _valid[k].get('score') is not None:
                _overall += _valid[k]['score'] * w
                _w_sum += w
        if _w_sum > 0:
            _overall /= _w_sum  # normalize to 0-100 based on present categories only
    except:
        pass

    # Grade (Polish descriptive)
    if _overall >= 85: _grade = 'Elitarny'
    elif _overall >= 72: _grade = 'Bardzo dobry'
    elif _overall >= 58: _grade = 'Dobry'
    elif _overall >= 42: _grade = 'Przeciętny'
    else: _grade = 'Do poprawy'

    # Grade (letter, for PRO backward compat)
    if _overall >= 85: _grade_letter = 'A+'
    elif _overall >= 75: _grade_letter = 'A'
    elif _overall >= 65: _grade_letter = 'B+'
    elif _overall >= 55: _grade_letter = 'B'
    elif _overall >= 45: _grade_letter = 'C+'
    elif _overall >= 35: _grade_letter = 'C'
    else: _grade_letter = 'D'

    _grade_col = '#16a34a' if _overall >= 72 else ('#d97706' if _overall >= 50 else '#dc2626')

    # ═══════════════════════════════════════════════════════
    # AEROBIC BASE LABEL (replaces interpret_thresholds logic)
    # ═══════════════════════════════════════════════════════
    if _vt1_trap:
        _aerobic_base = 'Pozornie dobra'  # ceiling trap
    elif _vt1p >= 80: _aerobic_base = 'Doskonała'
    elif _vt1p >= 70: _aerobic_base = 'Bardzo dobra'
    elif _vt1p >= 60: _aerobic_base = 'Dobra'
    elif _vt1p >= 50: _aerobic_base = 'Umiarkowana'
    else: _aerobic_base = 'Słaba'

    # ═══════════════════════════════════════════════════════
    # INTERPRETATIONS (unified text generation)
    # Each key → ready HTML string or None (=skip)
    # ═══════════════════════════════════════════════════════
    _interp = {}
    _sc = sport_class_raw.upper() if sport_class_raw else ''
    _sc_pl = vo2_class_sport or _sc  # e.g. "RECREATIONAL (run)"

    # ── VO₂max interpretation ──
    # Uses BOTH population percentile AND sport class for balanced assessment
    try:
        if _sc in ('ELITE', 'SUB_ELITE'):
            _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) to <b>poziom elitarny</b> ({_sc_pl}) \u2014 wybitna wydolność tlenowa. \U0001f4aa'
        elif _sc == 'COMPETITIVE':
            _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) to <b>poziom zawodniczy</b> ({_sc_pl}) \u2014 solidna baza do dalszego rozwoju.'
        elif _sc == 'TRAINED':
            if pctile_val >= 85:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) to <b>dobry poziom sportowy</b> ({_sc_pl}) \u2014 powyżej przeciętnej, solidna baza.'
            else:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) \u2014 <b>poziom wytrenowany</b> ({_sc_pl}). Regularne treningi przynoszą efekty.'
        elif _sc == 'RECREATIONAL':
            if pctile_val >= 80:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) jest <b>powyżej przeciętnej populacyjnej</b>, ale na <b>rekreacyjnym poziomie sportowym</b> ({_sc_pl}). Jest duży potencjał na poprawę treningiem interwałowym i objętościowym.'
            elif pctile_val >= 40:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) jest <b>w normie</b> ({_sc_pl}) \u2014 jest duży potencjał na poprawę regularnym treningiem.'
            else:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) wskazuje na <b>potencjał do poprawy</b> \u2014 regularny trening przyniesie szybkie efekty.'
        elif _sc == 'UNTRAINED':
            _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) wskazuje na <b>potencjał do poprawy</b> \u2014 systematyczny trening przyniesie szybkie i widoczne efekty.'
        else:
            # Fallback — no sport class available, use population only
            if pctile_val >= 85:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) plasuje Cię <b>powyżej przeciętnej</b> (~{pctile_val:.0f} percentyl).'
            elif pctile_val >= 40:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) jest <b>w normie</b> \u2014 jest duży potencjał na poprawę regularnym treningiem.'
            else:
                _interp['vo2max'] = f'Twój VO\u2082max ({_vo2_v:.1f} ml/kg/min) wskazuje na <b>potencjał do poprawy</b> \u2014 regularny trening przyniesie szybkie efekty.'
    except:
        pass

    # ── Thresholds (VT2 + VT1 combined) ──
    try:
        if _vt2p > 0:
            if _vt2_trap:
                # Threshold trap: high % but low absolute → misleading
                _interp['thresholds'] = f'VT2 przy {_vt2p:.0f}% VO\u2082max \u2014 Tw\u00f3j pr\u00f3g kwasowy jest bardzo blisko maksimum (doskonała tolerancja na kwasicę). Ale sam \u201esufit\u201d ({_vo2_ctx}) jest na ni\u017cszym poziomie sportowym. Analogia: silnik 100 KM, kt\u00f3ry wchodzi na czerwone dopiero przy {_vt2p:.0f} KM \u2014 \u015bwietna tolerancja, ale sam silnik m\u00f3g\u0142by by\u0107 wi\u0119kszy. <b>Priorytet: podnie\u015b VO\u2082max</b> \u2014 progi automatycznie p\u00f3jd\u0105 wy\u017cej.'
            elif _vt2p >= 85:
                _msg = f'Twoje progi są <b>wysoko ustawione</b> (VT2 przy {_vt2p:.0f}% VO\u2082max)'
                if _gap >= 25:
                    _msg += f' z szerokim gapem {_gap:.0f} bpm między progami \u2014 świetna elastyczność stref.'
                else:
                    _msg += ' \u2014 możesz utrzymywać wysokie tempo przez długi czas.'
                _interp['thresholds'] = _msg
            elif _vt2p >= 75:
                _interp['thresholds'] = f'Twoje progi są na <b>dobrym poziomie</b> (VT2 przy {_vt2p:.0f}%). Trening tempo i interwały progowe mogą je jeszcze podnieść.'
            elif _vt2p > 0:
                _interp['thresholds'] = f'Twoje progi mają <b>przestrzeń do poprawy</b> (VT2 przy {_vt2p:.0f}%). Systematyczny trening w Z3-Z4 pomoże je podnieść.'
    except:
        pass

    # ── VT1 (aerobic base) — only if meaningful ──
    try:
        if _vt1p > 0 and _vt1_trap:
            _spd_info = f', VT1 przy {_vt1_spd:.1f} km/h' if _vt1_spd > 0 else ''
            _interp['vt1'] = f'VT1 przy {_vt1p:.0f}% VO\u2082max wygląda dobrze procentowo, ale przy <b>niskiej wydolności absolutnej</b> ({_vo2_ctx}{_spd_info}). To \"pozornie dobra\" baza \u2014 buduj sufit tlenowy, baza pójdzie w górę automatycznie.'
        elif _vt1p > 0 and _vt1p < 55:
            _interp['vt1'] = f'Baza tlenowa (VT1 przy {_vt1p:.0f}% VO\u2082max) <b>wymaga wzmocnienia</b>. Więcej długich, spokojnych treningów w Z2 pomoże.'
    except:
        pass

    # ── Recovery ──
    try:
        if _hrr_v >= 40:
            _interp['recovery'] = f'Twoja regeneracja jest <b>bardzo dobra</b> \u2014 tętno spada szybko po wysiłku (HRR\u2081: {_hrr_v:.0f} bpm).'
        elif _hrr_v >= 25:
            _interp['recovery'] = f'Regeneracja na <b>dobrym poziomie</b> (HRR\u2081: {_hrr_v:.0f} bpm).'
        elif _hrr_v > 0:
            _interp['recovery'] = f'Regeneracja <b>wymaga poprawy</b> (HRR\u2081: {_hrr_v:.0f} bpm). Zadbaj o sen, nawodnienie i trening Z1-Z2.'
    except:
        pass

    # ── Economy ──
    try:
        if _use_re:
            if _re_v < 185:
                _interp['economy'] = f'Ekonomia biegu <b>ponadprzeciętna</b> (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""}.'
            elif _re_v < 200:
                _interp['economy'] = f'Ekonomia biegu <b>dobra</b> (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""} \u2014 w normie, z potencjałem na poprawę.'
            elif _re_v < 215:
                _interp['economy'] = f'Ekonomia biegu <b>przeciętna</b> (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""}. Plyometria i trening siłowy pomogą.'
            else:
                _interp['economy'] = f'Ekonomia biegu <b>do poprawy</b> (RE: {_re_v:.0f} ml/kg/km){f", z-score: {_gz:+.1f}" if _gz else ""}. Plyometria, trening siłowy i ćwiczenia techniczne poprawią efektywność.'
        else:
            if _gz > 0.5:
                _interp['economy'] = f'Ekonomia ruchu <b>ponadprzeciętna</b> (z-score: {_gz:+.1f}){" \u2014 RE: " + str(int(_re_v)) + " ml/kg/km" if _re_v else ""}.'
            elif _gz < -0.5:
                _interp['economy'] = f'Ekonomia ruchu <b>do poprawy</b> (z-score: {_gz:+.1f}){" \u2014 RE: " + str(int(_re_v)) + " ml/kg/km" if _re_v else ""}. Plyometria i trening siłowy pomogą.'
    except:
        pass

    # ── Ventilation ──
    try:
        if _slp < 25:
            _vc = vent_class
            _interp['ventilation'] = f'Efektywność wentylacyjna <b>wybitna</b> (VE/VCO\u2082 slope: {_slp:.1f}) \u2014 Twoje płuca pracują bardzo ekonomicznie.'
        elif _slp > 34:
            _vc = vent_class
            _interp['ventilation'] = f'Efektywność wentylacyjna <b>wymaga uwagi</b> (VE/VCO\u2082 slope: {_slp:.1f}{", " + _vc if _vc else ""}). Trening oddechowy może pomóc.'
        # 25-34 → skip (normal range)
    except:
        pass

    # ── Cardiac ──
    try:
        if _o2p > 115:
            _interp['cardiac'] = f'O\u2082 pulse na <b>{_o2p:.0f}% normy</b> \u2014 silne serce sportowe z wysoką objętością wyrzutową.'
        elif _o2p < 85:
            _interp['cardiac'] = f'O\u2082 pulse na <b>{_o2p:.0f}% normy</b> \u2014 objętość wyrzutowa serca może być limiterem. Interwały VO\u2082max pomogą.'
        # 85-115 → skip (normal)
    except:
        pass

    # ── Substrate ──
    try:
        _sub_score = _cat.get('substrate', {}).get('score', 50)
        if _sub_score >= 75:
            _interp['substrate'] = f'Metabolizm tłuszczowy <b>bardzo dobry</b> (FATmax: {_fat_v * 60:.0f} g/h) \u2014 świetna adaptacja do długich dystansów.'
        elif _sub_score >= 55:
            _interp['substrate'] = f'Metabolizm tłuszczowy <b>w normie</b> (FATmax: {_fat_v * 60:.0f} g/h). Więcej treningów Z2 pomoże przesunąć crossover w prawo.'
        elif _fat_v > 0 and _cop_v < 40:
            _interp['substrate'] = f'Wczesny crossover CHO/FAT (przy {_cop_v:.0f}% VO\u2082max) \u2014 wi\u0119cej trening\u00f3w Z2 oraz okazjonalne sesje \u201etrain low\u201d (Z1-Z2 na czczo, 1-2\u00d7/tyg.) poprawi\u0105 spalanie t\u0142uszcz\u00f3w. NIE na treningach kluczowych.'
        elif _fat_v > 0 and _fat_v < 0.25:
            _interp['substrate'] = f'Metabolizm tłuszczowy <b>ograniczony</b> (FATmax: {_fat_v * 60:.0f} g/h). Więcej treningów Z2 i periodyzacja węglowodanów pomogą.'
    except:
        pass
    return {
        'categories': _cat,
        'overall': _overall,
        'grade': _grade,
        'grade_letter': _grade_letter,
        'grade_color': _grade_col,
        'limiter_key': _limiter_key,
        'super_key': _super_key,
        'limiter': _limiter,
        'superpower': _superpower,
        'priority_ranking': _priority_ranking,
        'no_real_limiter': _no_real_limiter,
        'limiter_type': 'marginal' if _no_real_limiter else 'real',
        'sport_demands': _demands,
        'pctile_val': pctile_val,
        'aerobic_base': _aerobic_base,
        'sport_class': sport_class_raw,
        'interpretations': _interp,
        'flags': {
            'vt1_ceiling_trap': _vt1_trap,
            'vt2_threshold_trap': _vt2_trap,
            'economy_masking': _econ_mask,
        },
        'gauge_scores': {
            'profil': _overall,
            'vt2': _cat.get('vt2', {}).get('score', 0) or 0,
            'ventilation': _cat.get('ventilation', {}).get('score', 0) or 0,
            'economy': _cat.get('economy', {}).get('score', 0) or 0,
            'recovery': _cat.get('recovery', {}).get('score', 0) or 0,
        },
        'raw': {
            'vo2_rel': _vo2_v,
            'vt1_pct': _vt1p,
            'vt2_pct': _vt2p,
            'vt1_speed': _vt1_spd,
            'gap_bpm': _gap,
            'slope': _slp,
            'gain_z': _gz,
            're_mlkgkm': _re_v,
            'o2p_pct': _o2p,
            'hrr1': _hrr_v,
            'fatmax_gmin': _fat_v,
            'cop_pct': _cop_v,
            'bf_peak': bf_peak,
            'vo2_class': vo2_class,
            'vo2_class_sport': vo2_class_sport,
            'vent_class': vent_class,
        }
    }


# ════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════

class ReportAdapter:
    """
    Rola: Adapter / Render analityczny.
    Zgodność: [ANALYSIS_REPORT_BUILDER v1.1]
    Przywraca pełny format raportu dla Trenera AI.
    """

    @staticmethod
    def _safe_get(dictionary, key, default="[BRAK]"):
        val = dictionary.get(key)
        if val is None or val == "" or val == "nan":
            return default

        # Bezpieczne formatowanie liczb (float/int + numpy typy)
        if isinstance(val, (int, float, np.integer, np.floating)):
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                return default
            return f"{float(val):.2f}"

        return str(val)

    @staticmethod
    def _first_non_null(*values):
        for v in values:
            if v is not None and v != "" and str(v).lower() != "nan":
                return v
        return None


    @staticmethod
    def describe_protocol(df, cfg, results):
        """Generate protocol description and interactive table HTML."""

        proto_name = getattr(cfg, 'protocol_name', '') or ''
        modality = getattr(cfg, 'modality', 'run')
        mod_pl = 'Bieżnia' if modality == 'run' else ('Rower' if modality == 'bike' else modality.capitalize())
        # Infer test device from protocol name
        _prot_upper = proto_name.upper()
        if any(k in _prot_upper for k in ('RUN_', 'BRUCE', 'BIEZNIA', 'TREADMILL', 'STEP_3MIN', 'HYROX')):
            mod_pl = 'Bieżnia'
        elif any(k in _prot_upper for k in ('BIKE_', 'CYCLE', 'ROWER', 'WATT', 'ECHO')):
            mod_pl = 'Rower'
        elif any(k in _prot_upper for k in ('ROW_', 'WIOSLARZ')):
            mod_pl = 'Ergometr wiośl.'

        t_stop = results.get('E00', {}).get('t_stop', 0)
        dur_min = t_stop / 60 if t_stop else 0
        rec_avail = results.get('E00', {}).get('recovery_0_60_available', False)

        # Try RAW_PROTOCOLS first
        raw_segments = None
        try:
            if 'RAW_PROTOCOLS' in dir(__builtins__) or True:
                import builtins
                rp = globals().get('RAW_PROTOCOLS', {})
                if proto_name in rp:
                    raw_segments = rp[proto_name]
        except: pass

        # Build short description
        parts = [mod_pl]
        if raw_segments:
            speeds = [s.get('speed_kmh') for s in raw_segments if s.get('speed_kmh') is not None and s.get('speed_kmh',0) > 0]
            inclines = [s.get('incline_pct') for s in raw_segments if s.get('incline_pct',0) > 0]
            powers = [s.get('power_w') for s in raw_segments if s.get('power_w') is not None]

            # Determine type
            types = set(s.get('type','') for s in raw_segments)
            if any('ramp' in t for t in types):
                parts.append('rampowy')
            else:
                parts.append('stopniowany')

            if speeds:
                parts.append(f'{min(speeds):.0f}–{max(speeds):.0f} km/h')
            if powers:
                parts.append(f'{min(powers):.0f}–{max(powers):.0f} W')
            if inclines:
                parts.append(f'do {max(inclines):.0f}%')

            # Step duration
            const_segs = [s for s in raw_segments if s.get('type')=='const' and s.get('speed_kmh',0) > 0]
            if const_segs:
                import re
                def parse_t(t_str):
                    try:
                        p = t_str.split(':')
                        return int(p[0])*60 + int(p[1])
                    except: return 0
                durs = [parse_t(s['end']) - parse_t(s['start']) for s in const_segs]
                if durs:
                    from statistics import median
                    med = median(durs)
                    parts.append(f'{med/60:.0f}min/stopień')
        else:
            # Fallback to marker detection
            markers = []
            if 'Marker' in df.columns:
                mk = df[df['Marker'].notna() & (df['Marker'].astype(str).str.strip() != '')]
                for _, row in mk.iterrows():
                    markers.append({'label': str(row['Marker']).strip()})
            speeds_m = []
            for m in markers:
                lbl = m['label']
                if lbl.lower() not in ('ko','end','stop') and '%' not in lbl:
                    try: speeds_m.append(float(lbl.replace(',','.')))
                    except: pass
            if speeds_m:
                parts.append(f'stopniowany, {min(speeds_m):.0f}–{max(speeds_m):.0f} km/h')

        if dur_min > 0:
            parts.append(f'{dur_min:.0f} min')
        if rec_avail:
            t_max = df['Time_s'].max() if 'Time_s' in df.columns else 0
            rec_s = t_max - t_stop if t_max > t_stop else 0
            if rec_s > 30:
                parts.append(f'recovery {rec_s/60:.0f} min')

        return ', '.join(parts)

    @staticmethod
    def protocol_table_html(cfg, t_stop=None):
        """Generate protocol stages table HTML - only up to t_stop, with stop marker."""
        proto_name = getattr(cfg, 'protocol_name', '') or ''
        rp = globals().get('RAW_PROTOCOLS', {})
        segs = rp.get(proto_name, [])
        if not segs:
            return ''

        modality = getattr(cfg, 'modality', 'run')
        is_run = modality in ('run', 'walk')

        def parse_t(t_str):
            try:
                p = str(t_str).split(':')
                return int(p[0])*60 + int(p[1])
            except: return 99999

        rows = ''
        last_active_row = None
        for s in segs:
            stype = s.get('type','')
            start = s.get('start','')
            end_t = s.get('end','')
            start_s = parse_t(start)
            end_s = parse_t(end_t)

            # Skip stages entirely after t_stop
            if t_stop and start_s >= t_stop:
                break

            if is_run:
                speed = s.get('speed_kmh', s.get('speed_from',''))
                speed_to = s.get('speed_to','')
                incl = s.get('incline_pct', 0)
                if stype == 'ramp_speed':
                    desc = f'{speed}\u2192{speed_to} km/h'
                elif stype == 'dynamic_incline_steps_to_stop':
                    desc = f'{speed} km/h + nachylenie co {s.get("incline_step_pct",2)}%'
                else:
                    desc = f'{speed} km/h'
                    if incl and float(incl) > 0:
                        desc += f' / {incl}%'
            else:
                pwr = s.get('power_w', s.get('power_from',''))
                pwr_to = s.get('power_to','')
                if 'ramp' in stype:
                    desc = f'{pwr}\u2192{pwr_to} W'
                else:
                    desc = f'{pwr} W'

            # Determine if this is the stop stage (t_stop falls within it)
            is_stop_stage = t_stop and start_s < t_stop and end_s >= t_stop

            # Truncate end time to t_stop if stage extends beyond
            if t_stop and end_s > t_stop:
                mins = int(t_stop // 60)
                secs = int(t_stop % 60)
                end_t = f'{mins}:{secs:02d}'

            # Color
            if stype == 'const' and float(s.get('speed_kmh',0) or 0) == 0 and float(s.get('power_w',0) or 0) == 0:
                bg = '#f1f5f9'
            elif is_stop_stage:
                bg = '#fef2f2'  # red tint for stop
            elif 'ramp' in stype or 'dynamic' in stype:
                bg = '#fef3c7'
            else:
                bg = 'white'

            # Stop marker
            stop_mark = ' \u26d4' if is_stop_stage else ''

            rows += f'<tr style="background:{bg};"><td style="padding:3px 8px;">{start}</td><td style="padding:3px 8px;">{end_t}{stop_mark}</td><td style="padding:3px 8px;">{desc}</td></tr>'

        if not rows:
            return ''

        return f"""<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;">
<tr style="background:#f8fafc;font-weight:600;"><th style="padding:4px 8px;text-align:left;">Start</th><th style="padding:4px 8px;text-align:left;">End</th><th style="padding:4px 8px;text-align:left;">{'Prędkość / Nachylenie' if is_run else 'Moc'}</th></tr>
{rows}</table>"""


    @staticmethod
    def build_canon_table(df, results: 'Dict[str, Any]', cfg) -> 'Dict[str, Any]':
        """Tworzy płaską tabelę ze wszystkich silników."""

        # Pobieramy wyniki (zabezpieczenie pustymi słownikami)
        r01 = results.get('E01', {})  # QC/Max
        r02 = results.get('E02', {})  # Progi
        r03 = results.get('E03', {})  # Slope
        r04 = results.get('E04', {})  # OUES
        r05 = results.get('E05', {})  # O2Pulse
        r07 = results.get('E07', {})  # BreathingPattern
        r08 = results.get('E08', {})  # HRR
        r10 = results.get('E10', {})  # Substrate
        r11 = results.get('E11', {})  # Lactate
        r12 = results.get('E12', {})  # NIRS
        r15 = results.get('E15', {})  # Norm
        r16 = results.get('E16', {})  # Strefy

        ct = {}

        # --- A. META DANE ---
        ct['athlete_name'] = getattr(cfg, 'athlete_name', 'AUTO')
        ct['athlete_id'] = getattr(cfg, 'athlete_id', 'ID')
        ct['test_date'] = getattr(cfg, 'test_date', '-')
        ct['location'] = getattr(cfg, 'location', 'Lab')
        ct['operator'] = getattr(cfg, 'operator', 'Auto')
        ct['protocol'] = getattr(cfg, 'protocol_name', '-')
        ct['age'] = getattr(cfg, 'age_y', '-')
        ct['height'] = getattr(cfg, 'height_cm', '-')
        ct['weight'] = getattr(cfg, 'body_mass_kg', '-')

        # --- B. PROGI (ANCHORS) ---
        ct['VT1_HR'] = ReportAdapter._safe_get(r02, 'vt1_hr')
        vt1_spd = r02.get('vt1_speed_kmh') or r02.get('vt1_speed')
        ct['VT1_Speed'] = ReportAdapter._safe_get(r02, 'vt1_speed_kmh') if vt1_spd not in [None, ""] else ReportAdapter._safe_get(r02, 'vt1_speed') if r02.get('vt1_speed') not in [None, ""] else ReportAdapter._safe_get(r02, 'vt1_load')
        ct['VT1_Power'] = ReportAdapter._safe_get(r02, 'vt1_power_w') if r02.get('vt1_power_w') is not None else ReportAdapter._safe_get(r02, 'vt1_power')

        ct['VT2_HR'] = ReportAdapter._safe_get(r02, 'vt2_hr')
        vt2_spd = r02.get('vt2_speed_kmh') or r02.get('vt2_speed')
        ct['VT2_Speed'] = ReportAdapter._safe_get(r02, 'vt2_speed_kmh') if vt2_spd not in [None, ""] else ReportAdapter._safe_get(r02, 'vt2_speed') if r02.get('vt2_speed') not in [None, ""] else ReportAdapter._safe_get(r02, 'vt2_load')
        ct['VT2_Power'] = ReportAdapter._safe_get(r02, 'vt2_power_w') if r02.get('vt2_power_w') is not None else ReportAdapter._safe_get(r02, 'vt2_power')

        # --- B2. PROGI — %VO2max, VO2 values, plateau ---
        ct['VT1_VO2_mlmin'] = ReportAdapter._safe_get(r02, 'vt1_vo2_mlmin')
        ct['VT2_VO2_mlmin'] = ReportAdapter._safe_get(r02, 'vt2_vo2_mlmin')
        ct['VT1_pct_VO2max'] = ReportAdapter._safe_get(r15, 'vt1_pct_vo2peak')
        ct['VT2_pct_VO2max'] = ReportAdapter._safe_get(r15, 'vt2_pct_vo2peak')
        ct['VT1_pct_HRmax'] = ReportAdapter._safe_get(r15, 'vt1_pct_hrmax')
        ct['VT2_pct_HRmax'] = ReportAdapter._safe_get(r15, 'vt2_pct_hrmax')
        ct['VO2_determination'] = r15.get('vo2_determination', r01.get('vo2_determination', 'VO2peak'))
        ct['plateau_detected'] = r15.get('plateau_detected', r01.get('plateau_detected', False))
        ct['plateau_delta'] = ReportAdapter._safe_get(r15, 'plateau_delta_mlmin') or ReportAdapter._safe_get(r01, 'plateau_delta_mlmin')

        # --- B3. CROSS-VALIDATION VT↔LT (E18) ---
        r18 = results.get('E18', {})
        ct['E18_status'] = r18.get('status', 'N/A')
        ct['E18_overall_class'] = r18.get('summary', {}).get('overall_class', '')
        ct['E18_overall_score'] = r18.get('summary', {}).get('overall_score_pct', '')
        ct['E18_summary'] = r18.get('summary', {})
        ct['E18_layer1'] = r18.get('layer1_lactate_at_vt', {})
        ct['E18_layer2'] = r18.get('layer2_domain_confirmation', {})
        ct['E18_layer3'] = r18.get('layer3_concordance', {})
        ct['E18_flags'] = r18.get('flags', [])

        # --- CHART DATA (for interactive plots) ---
        # E11 lactate curve data
        ct['_chart_lactate_points'] = r11.get('raw_points', [])
        ct['_chart_lactate_poly'] = r11.get('poly_curve', {})
        ct['_chart_lt1_time'] = r11.get('lt1_time_sec')
        ct['_chart_lt1_la'] = r11.get('lt1_la_mmol')
        ct['_chart_lt2_time'] = r11.get('lt2_time_sec')
        ct['_chart_lt2_la'] = r11.get('lt2_la_mmol')
        ct['_chart_lt1_method'] = r11.get('lt1_method', '')
        ct['_chart_lt2_method'] = r11.get('lt2_method', '')

        # E12 NIRS data
        r12_full = results.get('E12', {})
        ct['_chart_nirs_traces'] = r12_full.get('traces', [])
        ct['_chart_nirs_channels'] = r12_full.get('channels_used', [])

        # --- INTERPRETATION ENGINE ---
        _age = float(ct.get('age', 30) or 30)
        _sex = getattr(cfg, 'sex', 'male') or 'male'
        ct['_interp_sex'] = _sex
        _weight = float(ct.get('weight', 70) or 70)

        # VO2peak relative
        _vo2_abs = results.get('E01', {}).get('vo2_peak_mlmin')
        _vo2_rel = None
        if _vo2_abs and _weight and _weight > 0:
            _vo2_rel = _vo2_abs / _weight
        ct['_interp_vo2_mlkgmin'] = round(_vo2_rel, 1) if _vo2_rel else None

        # Classify VO2max
        if _vo2_rel:
            _vc = interpret_vo2max(_vo2_rel, _age, _sex)
            ct['_interp_vo2_category'] = _vc['category']
            ct['_interp_vo2_percentile'] = _vc['percentile']
            ct['_interp_vo2_athlete_level'] = _vc['athlete_level']
            ct['_interp_vo2_color'] = _vc['color']
            ct['_interp_vo2_thresholds'] = _vc['thresholds']

        # Thresholds analysis
        _vt1_vo2 = results.get('E02', {}).get('vt1_vo2_mlmin') or results.get('E01', {}).get('vt1_vo2_mlmin')
        _vt2_vo2 = results.get('E02', {}).get('vt2_vo2_mlmin') or results.get('E01', {}).get('vt2_vo2_mlmin')
        _vt1_hr_val = results.get('E02', {}).get('vt1_hr')
        _vt2_hr_val = results.get('E02', {}).get('vt2_hr')
        if not _vt1_hr_val:
            try:
                _vh = ct.get('VT1_HR')
                _vt1_hr_val = float(_vh) if _vh and str(_vh) not in ('-','None','0','') else None
            except: _vt1_hr_val = None
        if not _vt2_hr_val:
            try:
                _vh = ct.get('VT2_HR')
                _vt2_hr_val = float(_vh) if _vh and str(_vh) not in ('-','None','0','') else None
            except: _vt2_hr_val = None
        _hr_max_val = results.get('E01', {}).get('hr_peak')
        if not _hr_max_val:
            try:
                _hm = ct.get('HR_max')
                _hr_max_val = float(_hm) if _hm and str(_hm) not in ('-','None','0','') else None
            except: _hr_max_val = None

        if _vo2_abs and (_vt1_vo2 or _vt2_vo2):
            _ti = interpret_thresholds(_vt1_vo2, _vt2_vo2, _vo2_abs, _vt1_hr_val, _vt2_hr_val, _hr_max_val)
            ct['_interp_vt1_pct_vo2'] = _ti.get('vt1_pct_vo2')
            ct['_interp_vt2_pct_vo2'] = _ti.get('vt2_pct_vo2')
            ct['_interp_vt1_pct_hr'] = _ti.get('vt1_pct_hr')
            ct['_interp_vt2_pct_hr'] = _ti.get('vt2_pct_hr')
            ct['_interp_aerobic_base'] = _ti.get('aerobic_base', '-')
            ct['_interp_gap_bpm'] = _ti.get('gap_bpm')

        # Test validity
        _rer_val = results.get('E01', {}).get('rer_peak')
        if _rer_val is None:
            try:
                _rr = ct.get('RER_peak')
                _rer_val = float(_rr) if _rr and str(_rr) not in ('-','None','') else None
            except: _rer_val = None
        _la_peak = None
        _e04 = results.get('E04', {})
        if _e04 and _e04.get('status') == 'OK':
            _la_vals = [p.get('la', 0) for p in _e04.get('points', []) if p.get('la')]
            if _la_vals: _la_peak = max(_la_vals)

        _tv = interpret_test_validity(_rer_val, _hr_max_val, _age, _la_peak)
        ct['_interp_test_maximal'] = _tv['is_maximal']
        ct['_interp_test_confidence'] = _tv['confidence']
        ct['_interp_test_criteria'] = _tv['criteria']
        ct['_interp_rer_desc'] = _tv['rer_desc']
        ct['_interp_la_peak'] = round(_la_peak, 1) if _la_peak else None
        ct['_interp_hr_max'] = _hr_max_val
        ct['_interp_age'] = _age

        # Generate observations
        # [observations moved to end of build_canon_table]

        # --- PROTOCOL SUMMARY ---
        _df_ex_p = results.get('_df_ex', pd.DataFrame())
        _df_full_p = results.get('_df_full', pd.DataFrame())
        _e00_p = results.get('E00', {})
        _e01_p = results.get('E01', {})
        _t_stop_p = _e00_p.get('t_stop', 0)

        ct['_prot_name'] = getattr(cfg, 'protocol_name', '-')
        ct['_prot_modality'] = getattr(cfg, 'modality', '-')
        ct['_gc_manual'] = getattr(cfg, 'gc_manual', None)

        # Protocol steps from RAW_PROTOCOLS
        try:
            from engine_core import RAW_PROTOCOLS as _RP
            _prot_segs = _RP.get(ct['_prot_name'], [])
        except ImportError:
            _prot_segs = []
        _prot_steps = []
        for _seg in _prot_segs:
            _st = _seg.get('type', '?')
            _s = _seg.get('start', '0:00')
            _e = _seg.get('end', '?')
            # Parse start time to seconds
            def _pts(ts):
                if isinstance(ts, (int,float)): return float(ts)
                parts = str(ts).split(':')
                if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
                if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                return 0
            _step = {
                'type': _st,
                'start_s': _pts(_s),
                'end_s': _pts(_e) if _e != '?' else None,
                'speed': _seg.get('speed_kmh', _seg.get('speed_from')),
                'speed_to': _seg.get('speed_to'),
                'incline': _seg.get('incline_pct', _seg.get('incline_start', 0)),
                'incline_step': _seg.get('incline_step'),
                'step_every': _seg.get('step_every_sec'),
                'power': _seg.get('power_w', _seg.get('power_from')),
                'power_to': _seg.get('power_to'),
            }
            _prot_steps.append(_step)
        ct['_prot_steps'] = _prot_steps

        if not _df_ex_p.empty and 'Time_sec' in _df_ex_p.columns:
            _t_ex = pd.to_numeric(_df_ex_p['Time_sec'], errors='coerce')
            _t_full = pd.to_numeric(_df_full_p.get('Time_sec', pd.Series()), errors='coerce') if not _df_full_p.empty else pd.Series()

            ct['_prot_ex_duration_s'] = round(_t_stop_p, 0) if _t_stop_p else round(float(_t_ex.max() - _t_ex.min()), 0)
            ct['_prot_total_duration_s'] = round(float(_t_full.max()), 0) if _t_full.notna().sum() > 0 else ct['_prot_ex_duration_s']
            ct['_prot_rec_duration_s'] = round(ct['_prot_total_duration_s'] - ct['_prot_ex_duration_s'], 0)

            _hr_ex = pd.to_numeric(_df_ex_p.get('HR_bpm', pd.Series()), errors='coerce')
            _vo2_ex = pd.to_numeric(_df_ex_p.get('VO2_mlmin', _df_ex_p.get('VO2_ml_min', pd.Series())), errors='coerce')
            ct['_prot_hr_start'] = round(float(_hr_ex.iloc[:20].mean()), 0) if _hr_ex.notna().sum() > 20 else None
            ct['_prot_hr_end'] = round(float(_hr_ex.iloc[-20:].mean()), 0) if _hr_ex.notna().sum() > 20 else None
            ct['_prot_vo2_start'] = round(float(_vo2_ex.iloc[:20].mean()), 0) if _vo2_ex.notna().sum() > 20 else None
            ct['_prot_vo2_end'] = round(float(_vo2_ex.iloc[-20:].mean()), 0) if _vo2_ex.notna().sum() > 20 else None

            # Phases
            if 'Faza' in _df_full_p.columns:
                _phases = _df_full_p['Faza'].dropna().unique().tolist()
                ct['_prot_phases'] = [str(p) for p in _phases]
            else:
                ct['_prot_phases'] = []

            # Speed/power range if available
            for col, key in [('Speed_kmh', '_prot_speed'), ('Power_W', '_prot_power')]:
                _vals = pd.to_numeric(_df_ex_p.get(col, pd.Series()), errors='coerce')
                _nz = _vals[_vals > 0]
                if len(_nz) > 10:
                    ct[f'{key}_min'] = round(float(_nz.min()), 1)
                    ct[f'{key}_max'] = round(float(_nz.max()), 1)

        # --- CHART DATA: Exercise time series (V-slope, Fat/CHO) ---
        # Downsample to ~150 points for chart
        _df_ex = results.get('_df_ex', pd.DataFrame())
        if not _df_ex.empty and 'Time_sec' in _df_ex.columns:
            _step = max(1, len(_df_ex) // 150)
            _ds = _df_ex.iloc[::_step]
            _t = pd.to_numeric(_ds.get('Time_sec', pd.Series()), errors='coerce')
            _vo2 = pd.to_numeric(_ds.get('VO2_mlmin', _ds.get('VO2_ml_min', pd.Series())), errors='coerce')
            _vco2 = pd.to_numeric(_ds.get('VCO2_mlmin', _ds.get('VCO2_ml_min', pd.Series())), errors='coerce')
            _hr = pd.to_numeric(_ds.get('HR_bpm', pd.Series()), errors='coerce')
            _cho = pd.to_numeric(_ds.get('CHO_g_min', pd.Series()), errors='coerce')
            _fat = pd.to_numeric(_ds.get('FAT_g_min', pd.Series()), errors='coerce')

            _valid_gas = _t.notna() & _vo2.notna() & _vco2.notna()
            if _valid_gas.sum() > 10:
                ct['_chart_gas_time'] = [round(v,1) for v in _t[_valid_gas].tolist()]
                ct['_chart_gas_vo2'] = [round(v,0) for v in _vo2[_valid_gas].tolist()]
                ct['_chart_gas_vco2'] = [round(v,0) for v in _vco2[_valid_gas].tolist()]
                ct['_chart_gas_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr[_valid_gas].tolist()]

            _valid_sub = _t.notna() & (_cho.notna() | _fat.notna())
            if _valid_sub.sum() > 10:
                ct['_chart_sub_time'] = [round(v,1) for v in _t[_valid_sub].tolist()]
                ct['_chart_sub_cho'] = [round(v,2) if pd.notna(v) else 0 for v in _cho[_valid_sub].tolist()]
                ct['_chart_sub_fat'] = [round(v,2) if pd.notna(v) and v > 0 else 0 for v in _fat[_valid_sub].tolist()]
                ct['_chart_sub_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr[_valid_sub].tolist()]
        # --- CHART DATA: Full VO2 kinetics (exercise + recovery) ---
        _df_full = results.get('_df_full', pd.DataFrame())
        if _df_full.empty:
            _df_full = _df_ex  # fallback to exercise only
        if not _df_full.empty and 'Time_sec' in _df_full.columns:
            _step_f = max(1, len(_df_full) // 200)
            _ds_f = _df_full.iloc[::_step_f]
            _t_f = pd.to_numeric(_ds_f.get('Time_sec', pd.Series()), errors='coerce')
            _vo2_f = pd.to_numeric(_ds_f.get('VO2_mlmin', _ds_f.get('VO2_ml_min', pd.Series())), errors='coerce')
            _hr_f = pd.to_numeric(_ds_f.get('HR_bpm', pd.Series()), errors='coerce')
            _spd_f = pd.to_numeric(_ds_f.get('Speed_kmh', pd.Series()), errors='coerce')
            _pwr_f = pd.to_numeric(_ds_f.get('Power_W', pd.Series()), errors='coerce')
            _valid_f = _t_f.notna() & _vo2_f.notna()
            if _valid_f.sum() > 10:
                ct['_chart_full_time'] = [round(v,1) for v in _t_f[_valid_f].tolist()]
                ct['_chart_full_vo2'] = [round(v,0) for v in _vo2_f[_valid_f].tolist()]
                ct['_chart_full_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr_f[_valid_f].tolist()]
                _s = _spd_f[_valid_f]
                _p = _pwr_f[_valid_f]
                if _s.notna().sum() > 10 and (_s > 0).sum() > 10:
                    ct['_chart_full_speed'] = [round(v,1) if pd.notna(v) and v > 0 else None for v in _s.tolist()]
                if _p.notna().sum() > 10 and (_p > 0).sum() > 10:
                    ct['_chart_full_power'] = [round(v,0) if pd.notna(v) and v > 0 else None for v in _p.tolist()]
        ct['_chart_t_stop'] = results.get('E00', {}).get('t_stop')
        ct['_chart_vo2peak'] = results.get('E01', {}).get('vo2_peak_mlmin')



        # E02 thresholds for overlay
        ct['_chart_vt1_time'] = r02.get('vt1_time_sec')
        ct['_chart_vt2_time'] = r02.get('vt2_time_sec')
        ct['_chart_vt1_hr'] = r02.get('vt1_hr')
        ct['_chart_vt2_hr'] = r02.get('vt2_hr')

        # E18 cross-validation for annotations
        r18 = results.get('E18', {})
        ct['_chart_e18_l1'] = r18.get('layer1_lactate_at_vt', {})

        # --- C. STREFY ---
        zones_list = ['z1', 'z2', 'z3', 'z4', 'z5']
        _z_container = r16.get('zones', r16) if isinstance(r16, dict) else {}
        for z in zones_list:
            z_data = _z_container.get(z, {}) if isinstance(_z_container, dict) else {}
            prefix = z.upper()
            ct[f'{prefix}_HR_low'] = ReportAdapter._safe_get(z_data, 'hr_low', '...')
            ct[f'{prefix}_HR_high'] = ReportAdapter._safe_get(z_data, 'hr_high', '...')
            ct[f'{prefix}_speed_low'] = ReportAdapter._safe_get(z_data, 'speed_low', '')
            ct[f'{prefix}_speed_high'] = ReportAdapter._safe_get(z_data, 'speed_high', '')
            ct[f'{prefix}_name'] = z_data.get('name_pl', '')

        # --- D. METRYKI FIZJOLOGICZNE ---
        # VO2peak rel: preferuj E15 (znormalizowane), fallback E01
        vo2_rel = ReportAdapter._first_non_null(
            r15.get('vo2_rel'),
            r01.get('vo2_peak_mlkgmin'),
            r01.get('vo2_rel')
        )
        ct['VO2peak_rel'] = ReportAdapter._safe_get({'v': vo2_rel}, 'v')

        # VO2peak abs (L/min): preferuj E15, fallback E01 (ml/min -> L/min)
        vo2_abs_lmin = ReportAdapter._first_non_null(
            r15.get('vo2_abs'),
            r01.get('vo2_peak_lmin'),
            r01.get('vo2_abs_lmin')
        )
        if vo2_abs_lmin is None:
            vo2_abs_mlmin = ReportAdapter._first_non_null(
                r01.get('vo2_peak_mlmin'),
                r01.get('vo2_abs_mlmin')
            )
            if vo2_abs_mlmin is not None:
                try:
                    vo2_abs_lmin = float(vo2_abs_mlmin) / 1000.0
                except Exception:
                    vo2_abs_lmin = None
        ct['VO2peak_abs'] = ReportAdapter._safe_get({'v': vo2_abs_lmin}, 'v')

        # E15 klasyfikacja
        ct['VO2_class_pop'] = r15.get('vo2_class_pop', '-')
        ct['VO2_class_sport'] = r15.get('vo2_class_sport', '-')
        ct['VO2_pct_predicted'] = ReportAdapter._safe_get(r15, 'vo2_pct_predicted')
        ct['VO2_pred_method'] = r15.get('vo2_predicted_method', 'Wasserman')
        ct['VO2_pred_wass_mlkg'] = ReportAdapter._safe_get(r15, 'vo2_predicted_wasserman_mlkg')
        ct['RER_class'] = r15.get('rer_class', '-')
        ct['VE_VCO2_class'] = r15.get('ve_vco2_class', '-')
        ct['VE_VCO2_class_desc'] = r15.get('ve_vco2_desc', '-')
        ct['test_quality'] = r15.get('test_quality', '-')

        ct['HRmax'] = ReportAdapter._safe_get(r01, 'hr_peak')
        ct['RERpeak'] = ReportAdapter._safe_get(r01, 'rer_peak')

        # VEpeak: obsługa różnych nazw kluczy z E01
        ve_peak = ReportAdapter._first_non_null(
            r01.get('ve_peak_lmin'),
            r01.get('ve_peak'),
            r01.get('VE_peak')
        )
        ct['VEpeak'] = ReportAdapter._safe_get({'v': ve_peak}, 'v')

        # E03 VentSlope v2.0 — multi-method
        ct['VE_VCO2_slope_full']  = ReportAdapter._safe_get(r03, 'slope_full')
        ct['VE_VCO2_slope_vt2']   = ReportAdapter._safe_get(r03, 'slope_to_vt2')
        ct['VE_VCO2_slope_vt1']   = ReportAdapter._safe_get(r03, 'slope_to_vt1')
        ct['VE_VCO2_intercept']   = ReportAdapter._safe_get(r03, 'intercept_to_vt2') if r03.get('intercept_to_vt2') is not None else ReportAdapter._safe_get(r03, 'intercept_full')
        ct['VE_VCO2_nadir']       = ReportAdapter._safe_get(r03, 've_vco2_nadir')
        ct['VE_VCO2_at_vt1']     = ReportAdapter._safe_get(r03, 've_vco2_at_vt1')
        ct['VE_VCO2_peak']       = ReportAdapter._safe_get(r03, 've_vco2_peak')
        ct['VE_VCO2_vent_class'] = r03.get('ventilatory_class', '-')
        ct['VE_VCO2_vc_desc']    = r03.get('vc_description', '-')
        ct['VE_VCO2_predicted']  = ReportAdapter._safe_get(r03, 'predicted_slope')
        ct['VE_VCO2_pct_pred']   = ReportAdapter._safe_get(r03, 'pct_predicted')
        ct['VE_VCO2_r2']        = ReportAdapter._safe_get(r03, 'r2_to_vt2') if r03.get('r2_to_vt2') is not None else ReportAdapter._safe_get(r03, 'r2_full')
        ct['PETCO2_rest']        = ReportAdapter._safe_get(r03, 'petco2_rest', '-')
        ct['PETCO2_vt1']         = ReportAdapter._safe_get(r03, 'petco2_vt1', '-')
        ct['PETCO2_peak']        = ReportAdapter._safe_get(r03, 'petco2_peak', '-')
        ct['VE_VCO2_flags']      = ', '.join(r03.get('flags', [])) if r03.get('flags') else 'Brak'
        # Legacy compat
        ct['VE_VCO2_slope'] = ct['VE_VCO2_slope_vt2'] if ct['VE_VCO2_slope_vt2'] != '[BRAK]' else ct['VE_VCO2_slope_full']
        ct['OUES'] = ReportAdapter._safe_get(r04, 'oues100', ReportAdapter._safe_get(r04, 'oues_val'))
        ct['OUES_90'] = ReportAdapter._safe_get(r04, 'oues90')
        ct['OUES_75'] = ReportAdapter._safe_get(r04, 'oues75')
        ct['OUES_toVT1'] = ReportAdapter._safe_get(r04, 'oues_to_vt1')
        ct['OUES_toVT2'] = ReportAdapter._safe_get(r04, 'oues_to_vt2')
        ct['OUES_per_kg'] = ReportAdapter._safe_get(r04, 'oues_per_kg')
        ct['OUES_per_BSA'] = ReportAdapter._safe_get(r04, 'oues_per_bsa')
        ct['OUES_pred_Holl'] = ReportAdapter._safe_get(r04, 'oues_pred_hollenberg')
        ct['OUES_pct_Holl'] = ReportAdapter._safe_get(r04, 'oues_pct_hollenberg')
        ct['OUES_R2'] = ReportAdapter._safe_get(r04, 'oues_r2_100')
        ct['OUES_submax'] = ReportAdapter._safe_get(r04, 'oues_submax_stability')
        ct['OUES_flags'] = ', '.join(r04.get('oues_flags', [])) if isinstance(r04.get('oues_flags'), list) else ReportAdapter._safe_get(r04, 'oues_flags')
        ct['BR_pct'] = '-'

        # O2 Pulse: fallbacki nazw
        o2p = ReportAdapter._first_non_null(
            r15.get('o2pulse_peak'),
            r01.get('o2pulse_peak'),
            r01.get('o2_pulse_peak'),
            r01.get('O2Pulse_peak')
        )
        ct['O2pulse_peak'] = ReportAdapter._safe_get({'v': o2p}, 'v')
        # E05 v2.0 fields
        ct['O2pulse_at_vt1'] = ReportAdapter._safe_get(r05, 'o2pulse_at_vt1')
        ct['O2pulse_at_vt2'] = ReportAdapter._safe_get(r05, 'o2pulse_at_vt2')
        ct['O2pulse_pred_friend'] = ReportAdapter._safe_get(r05, 'predicted_friend')
        ct['O2pulse_pred_noodle'] = ReportAdapter._safe_get(r05, 'predicted_noodle')
        ct['O2pulse_pct_friend'] = ReportAdapter._safe_get(r05, 'pct_predicted_friend')
        ct['O2pulse_pct_noodle'] = ReportAdapter._safe_get(r05, 'pct_predicted_noodle')
        ct['O2pulse_est_sv'] = ReportAdapter._safe_get(r05, 'estimated_sv_peak_ml')
        ct['O2pulse_trajectory'] = ReportAdapter._safe_get(r05, 'trajectory', '-')
        ct['O2pulse_traj_desc'] = ReportAdapter._safe_get(r05, 'trajectory_desc', '-')
        ct['O2pulse_ff'] = ReportAdapter._safe_get(r05, 'flattening_fraction')
        ct['O2pulse_o2prr'] = ReportAdapter._safe_get(r05, 'o2prr')
        ct['O2pulse_flags'] = ', '.join(r05.get('flags', [])) if r05.get('flags') else 'BRAK'

        ct['HRR_60s'] = ReportAdapter._safe_get(r08, 'hrr_1min')
        ct['HRR_180s'] = ReportAdapter._safe_get(r08, 'hrr_3min')
        ct['HRR_mode'] = ReportAdapter._safe_get(r08, 'recovery_mode', '-')
        ct['HRR_quality'] = ReportAdapter._safe_get(r08, 'quality', '-')

        # --- E. METABOLICZNE / INNE ---
        ct['SmO2_min'] = ReportAdapter._safe_get(r12, 'smo2_min')
        ct['_e12_raw'] = r12

        # E13 Drift
        r13 = results.get('E13', {})
        ct['_e13_raw'] = r13
        r17 = results.get('E17', {})
        ct['_e17_raw'] = r17

        # E19 Concordance
        r19 = results.get('E19', {})
        ct['_e19_raw'] = r19
        ct['validity_score'] = r19.get('validity_score', '-')
        ct['validity_grade'] = r19.get('validity_grade', '-')
        ct['concordance_score'] = r19.get('concordance_score', '-')
        ct['concordance_grade'] = r19.get('concordance_grade', '-')

        # QC metadata for HTML
        qc = results.get('_qc_log', {})
        ct['_qc_status'] = qc.get('status', 'PASS')
        ct['_engines_ok_n'] = str(len(qc.get('engines_executed_ok', [])))
        ct['_engines_fail_n'] = str(len(qc.get('engines_failed', [])))
        lim = qc.get('engines_limited', [])
        ct['_engines_lim_list'] = ', '.join(lim) if lim else ''
        ct['Lactate_peak'] = ReportAdapter._safe_get(r11, 'lt_peak')
        ct['_e11_raw'] = r11
        if r11.get('status') == 'OK':
            ct['La_baseline'] = str(round(r11.get('la_baseline', 0), 2)) if r11.get('la_baseline') else '-'
            ct['LT1_method'] = r11.get('lt1_method', '-') or '-'
            ct['LT1_HR'] = str(round(r11['lt1_hr_bpm'], 0)) if r11.get('lt1_hr_bpm') else '-'
            ct['LT1_Speed'] = str(round(r11['lt1_speed_kmh'], 1)) if r11.get('lt1_speed_kmh') else '-'
            ct['LT1_La'] = str(round(r11['lt1_la_mmol'], 2)) if r11.get('lt1_la_mmol') else '-'
            ct['LT2_method'] = r11.get('lt2_method', '-') or '-'
            ct['LT2_HR'] = str(round(r11['lt2_hr_bpm'], 0)) if r11.get('lt2_hr_bpm') else '-'
            ct['LT2_Speed'] = str(round(r11['lt2_speed_kmh'], 1)) if r11.get('lt2_speed_kmh') else '-'
            ct['LT2_La'] = str(round(r11['lt2_la_mmol'], 2)) if r11.get('lt2_la_mmol') else '-'
            ct['La_n_points'] = str(r11.get('n_points', 0))

        # FATmax / crossover fallbacki
        fat_max = ReportAdapter._first_non_null(r10.get('mfo_gmin'), r10.get('fat_max'), r10.get('fatmax_g_min'))
        fat_max_hr = ReportAdapter._first_non_null(r10.get('fat_max_hr'), r10.get('fatmax_hr'))
        cho_cross = ReportAdapter._first_non_null(r10.get('cop_hr'), r10.get('cho_crossover_hr'), r10.get('crossover_hr'))

        # E06
        r06 = results.get('E06', {})
        ct['GAIN_below_vt1'] = r06.get('gain_below_vt1')
        ct['GAIN_below_vt1_r2'] = r06.get('gain_below_vt1_r2')
        ct['GAIN_full'] = r06.get('gain_full')
        ct['GAIN_unit'] = r06.get('gain_unit', '')
        ct['GAIN_modality'] = r06.get('modality', '')
        ct['RE_mlkgkm'] = r06.get('running_economy_mlkgkm')
        ct['RE_class'] = r06.get('re_classification')
        ct['DELTA_EFF'] = r06.get('delta_efficiency_pct')
        ct['LIN_BREAK_time'] = r06.get('linearity_break_time_s')
        ct['GAIN_z'] = r06.get('gain_z_score')
        ct['GAIN_norm_ref'] = r06.get('norm_gain_ref')
        ct['GAIN_norm_src'] = r06.get('norm_src')
        ct['GAIN_at_vt1'] = r06.get('gain_at_vt1')
        ct['GAIN_at_vt2'] = r06.get('gain_at_vt2')
        ct['RE_at_vt1'] = r06.get('re_at_vt1')
        ct['RE_at_vt2'] = r06.get('re_at_vt2')
        ct['VO2_at_vt1_e06'] = r06.get('vo2_at_vt1')
        ct['VO2_at_vt2_e06'] = r06.get('vo2_at_vt2')
        ct['LOAD_at_vt1'] = r06.get('load_at_vt1')
        ct['LOAD_at_vt2'] = r06.get('load_at_vt2')
        ct['EFF_at_vt1'] = r06.get('eff_at_vt1')
        ct['EFF_at_vt2'] = r06.get('eff_at_vt2')
        ct['FATmax_g'] = ReportAdapter._safe_get({'v': fat_max}, 'v')
        ct['FATmax_HR'] = ReportAdapter._safe_get({'v': fat_max_hr}, 'v')
        ct['CHO_Cross_HR'] = ReportAdapter._safe_get({'v': cho_cross}, 'v')

        # Zone substrate rates
        zs = r10.get('zone_substrate', {})
        for zn in ('z2', 'z4', 'z5'):
            zd = zs.get(zn, {})
            ct[f'{zn}_fat_gh'] = zd.get('fat_gh')
            ct[f'{zn}_cho_gh'] = zd.get('cho_gh')
            ct[f'{zn}_fat_pct'] = zd.get('fat_pct')
            ct[f'{zn}_cho_pct'] = zd.get('cho_pct')
            ct[f'{zn}_kcal_h'] = zd.get('kcal_h')
            ct[f'{zn}_rer_valid'] = zd.get('rer_valid', True)

        # Additional E10 v2 fields
        ct['FATmax_pct_vo2peak'] = r10.get('fatmax_pct_vo2peak')
        ct['FATmax_pct_hrmax'] = r10.get('fatmax_pct_hrmax')
        ct['FATmax_zone_hr_low'] = r10.get('fatmax_zone_hr_low')
        ct['FATmax_zone_hr_high'] = r10.get('fatmax_zone_hr_high')
        ct['MFO_mgkg_min'] = r10.get('mfo_mgkg_min')
        ct['COP_pct_vo2peak'] = r10.get('cop_pct_vo2peak')
        ct['COP_RER'] = r10.get('cop_rer')
        ct['COP_note'] = r10.get('cop_note')
        ct['fat_pct_at_vt1'] = r10.get('fat_pct_at_vt1')
        ct['cho_pct_at_vt1'] = r10.get('cho_pct_at_vt1')


        # ─── FINAL: Generate observations & training recs (after all ct values populated) ───
        ct['_e07_raw'] = r07
        ct['_interp_observations'] = generate_observations(ct)
        ct['_interp_training_recs'] = generate_training_recs(ct)
        # E14 Kinetics
        r14 = results.get('E14', {})
        ct['T_half_VO2'] = r14.get('T_half_VO2_simple_s')
        ct['kinetics_tau'] = r14.get('tau_s')
        ct['kinetics_r2'] = r14.get('r_squared')
        ct['kinetics_class'] = r14.get('classification')
        ct['kinetics_VO2DR'] = r14.get('VO2_delay_recovery_s')
        ct['T_half_VCO2'] = r14.get('T_half_VCO2_s')
        ct['T_half_VE'] = r14.get('T_half_VE_s')
        ct['kinetics_MRT'] = r14.get('MRT_s')
        ct['kinetics_flags'] = r14.get('flags', [])
        ct['_protocol_description'] = ReportAdapter.describe_protocol(df, cfg, results)
        ct['_protocol_table_html'] = ReportAdapter.protocol_table_html(cfg, t_stop=results.get('E00', {}).get('t_stop'))

        # ═══════════════════════════════════════════════════════════════
        # ALIAS BLOCK: mapowanie kluczy dla render_html_report
        # ═══════════════════════════════════════════════════════════════

        # Meta
        ct['age_y'] = ct.get('age')
        ct['body_mass_kg'] = ct.get('weight')
        ct['height_cm'] = ct.get('height')
        ct['sex'] = getattr(cfg, 'sex', 'male')
        ct['modality'] = getattr(cfg, 'modality', 'run')
        # Infer test_device from protocol (determines RE vs GAIN, independent of sport)
        _prot = getattr(cfg, 'protocol_name', '').upper()
        if any(k in _prot for k in ('RUN_', 'BRUCE', 'BIEZNIA', 'TREADMILL', 'STEP_3MIN')):
            ct['test_device'] = 'treadmill'
        elif any(k in _prot for k in ('BIKE_', 'CYCLE', 'ROWER', 'ERGOM')):
            ct['test_device'] = 'bike_erg'
        elif any(k in _prot for k in ('ROW_', 'WIOSLARZ')):
            ct['test_device'] = 'rowing_erg'
        else:
            ct['test_device'] = 'treadmill' if ct.get('_chart_full_speed') else 'other'
        ct['protocol_name'] = ct.get('protocol') or getattr(cfg, 'protocol_name', '-')

        # VO2 peak
        ct['VO2max_ml_kg_min'] = ct.get('VO2peak_rel')
        ct['VO2max_L_min'] = ct.get('VO2peak_abs')
        ct['VO2max_abs_Lmin'] = ct.get('VO2peak_abs')
        ct['VO2max_class_label'] = ct.get('VO2_class_pop') or ct.get('_interp_vo2_category')
        ct['VO2max_percentile'] = ct.get('_interp_vo2_percentile')
        ct['VO2max_determination'] = ct.get('VO2_determination') or ct.get('plateau_detected')

        # Peaks
        ct['HR_peak'] = ct.get('HRmax')
        ct['RER_peak'] = ct.get('RERpeak')
        ct['VE_peak'] = ct.get('VEpeak')

        # Thresholds
        ct['VT1_HR_bpm'] = ct.get('VT1_HR')
        ct['VT1_Time_s'] = r02.get('vt1_time_sec')
        ct['VT1_pct_VO2peak'] = ct.get('VT1_pct_VO2max') or ct.get('_interp_vt1_pct_vo2')
        ct['VT1_confidence'] = r02.get('vt1_confidence', 0)
        ct['VT2_HR_bpm'] = ct.get('VT2_HR')
        ct['VT2_Time_s'] = r02.get('vt2_time_sec')
        ct['VT2_pct_VO2peak'] = ct.get('VT2_pct_VO2max') or ct.get('_interp_vt2_pct_vo2')
        ct['VT2_confidence'] = r02.get('vt2_confidence', 0)

        # VE/VCO2
        ct['VE_VCO2_slope'] = ct.get('VE_VCO2_slope_vt2') or ct.get('VE_VCO2_slope_full')
        ct['VE_VCO2_ventilatory_class'] = ct.get('VE_VCO2_vent_class')

        # Additional metrics
        ct['OUES'] = r04.get('oues') or r04.get('OUES')
        ct['O2pulse_peak'] = r05.get('o2pulse_peak_ml') or r05.get('o2_pulse_peak') or r05.get('o2pulse_peak')
        ct['BR_pct'] = results.get('E09', {}).get('br_pct')
        ct['T_half_VO2'] = results.get('E06', {}).get('t_half_s')
        ct['HRR_60s'] = r08.get('hrr_1min')
        ct['LOAD_at_vt1'] = ct.get('VT1_Speed') or ct.get('VT1_Power')
        ct['LOAD_at_vt2'] = ct.get('VT2_Speed') or ct.get('VT2_Power')

        # Zones
        ct['_zones_data'] = r16.get('zones', r16) if isinstance(r16, dict) else {}

        # Raw engine results
        ct['_e02_raw'] = results.get('E02', {})
        for _eid in ['E00','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','E16','E17','E19']:
            ct['_' + _eid.lower() + '_raw'] = results.get(_eid, {})

        # E06 efficiency aliases
        _r06 = results.get('E06', {})
        for _k06 in ['gain_below_vt1','gain_below_vt1_r2','gain_at_vt1','gain_at_vt2','gain_full','gain_modality','gain_norm_ref','gain_norm_src','gain_unit','gain_z','re_at_vt1','re_at_vt2','re_class','eff_at_vt1','eff_at_vt2','delta_eff','lin_break_time']:
            _parts = _k06.split('_')
            _ct_key = _parts[0].upper() + '_' + '_'.join(_parts[1:])
            ct[_ct_key] = _r06.get(_k06)

        # Performance Context passthrough (from results, not ct!)
        _pc = results.get('_performance_context', {})
        if _pc:
            ct['_performance_context'] = _pc

        # ── UNIFIED PROFILE SCORING (computed once, used by all reports) ──
        try:
            ct['_profile'] = compute_profile_scores(ct)
            # Override aerobic_base with ceiling-trap-aware version
            if ct['_profile'] and ct['_profile'].get('aerobic_base'):
                ct['_interp_aerobic_base'] = ct['_profile']['aerobic_base']
        except Exception:
            ct['_profile'] = None

        return ct

    @staticmethod
    def _oues_interpretation(ct):
        """Build OUES interpretation string for report."""
        oues = ct.get('OUES')
        if oues in (None, '-', '', '[BRAK]'):
            return '• OUES: [BRAK]'

        lines = []
        lines.append(f"• OUES100: [{oues}] ml/min/log(L/min)")

        # Submaximal variants
        o90 = ct.get('OUES_90', '-')
        o75 = ct.get('OUES_75', '-')
        ovt1 = ct.get('OUES_toVT1', '-')
        ovt2 = ct.get('OUES_toVT2', '-')
        if o90 not in (None, '-'):
            lines.append(f"  ↳ OUES90: [{o90}] | OUES75: [{o75}] | do VT1: [{ovt1}] | do VT2: [{ovt2}]")

        # Stability
        stab = ct.get('OUES_submax', '-')
        if stab not in (None, '-'):
            try:
                sv = float(stab)
                stab_txt = 'dobra' if 0.95 <= sv <= 1.05 else ('umiarkowana' if 0.90 <= sv <= 1.10 else 'niska')
                lines.append(f"  ↳ Stabilność submaksymalna (OUES90/100): [{stab}] — {stab_txt}")
            except (ValueError, TypeError):
                pass

        # Normalization
        opkg = ct.get('OUES_per_kg', '-')
        opbsa = ct.get('OUES_per_BSA', '-')
        if opkg not in (None, '-'):
            lines.append(f"  ↳ OUES/kg: [{opkg}] | OUES/BSA (OUESI): [{opbsa}]")

        # Predicted (Hollenberg)
        pred = ct.get('OUES_pred_Holl', '-')
        pct = ct.get('OUES_pct_Holl', '-')
        if pct not in (None, '-'):
            try:
                pv = float(pct)
                if pv < 70:
                    interp = 'PONIŻEJ NORMY — wskazuje na patologię'
                elif pv < 85:
                    interp = 'Poniżej średniej — obniżona efektywność'
                elif pv < 100:
                    interp = 'W normie — prawidłowa efektywność'
                elif pv < 115:
                    interp = 'Powyżej normy — wysoka efektywność'
                else:
                    interp = 'Znacznie powyżej normy — bardzo wysoka efektywność'
                lines.append(f"  ↳ Predicted (Hollenberg): [{pred}] | % predicted: [{pct}]% — {interp}")
            except (ValueError, TypeError):
                lines.append(f"  ↳ Predicted (Hollenberg): [{pred}] | % predicted: [{pct}]%")

        # R²
        r2 = ct.get('OUES_R2', '-')
        if r2 not in (None, '-'):
            try:
                r2v = float(r2)
                r2_txt = 'doskonałe dopasowanie' if r2v >= 0.95 else ('dobre' if r2v >= 0.90 else 'niskie — ostrożna interpretacja')
                lines.append(f"  ↳ R²: [{r2}] — {r2_txt}")
            except (ValueError, TypeError):
                pass

        # Flags
        fl = ct.get('OUES_flags', '-')
        if fl and fl not in (None, '-', ''):
            lines.append(f"  ↳ Flagi: [{fl}]")

        return '\n'.join(lines)

    @staticmethod
    def _breathing_pattern_report(ct):
        """E07 Breathing Pattern report section — with sport context."""
        e07 = ct.get("_e07_raw", {})
        if not e07 or e07.get("status") != "OK":
            return ""

        # ─── Determine sport context ───
        _bp_vo2 = ct.get('_interp_vo2_mlkgmin')
        _bp_ve_slope = None
        try:
            _ve_raw = ct.get('VE_VCO2_slope')
            if _ve_raw and str(_ve_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_slope = float(_ve_raw)
        except: pass
        _is_sport = (_bp_vo2 is not None and _bp_vo2 >= 45 and _bp_ve_slope is not None and _bp_ve_slope < 30)
        _sex_bp = ct.get('_interp_sex', 'male')
        _sport_tier = 'sedentary'
        if _bp_vo2:
            if _sex_bp == 'male':
                if _bp_vo2 >= 65: _sport_tier = 'elite'
                elif _bp_vo2 >= 55: _sport_tier = 'competitive'
                elif _bp_vo2 >= 45: _sport_tier = 'trained'
            else:
                if _bp_vo2 >= 55: _sport_tier = 'elite'
                elif _bp_vo2 >= 45: _sport_tier = 'competitive'
                elif _bp_vo2 >= 38: _sport_tier = 'trained'

        L = []
        _ctx_tag = f" [kontekst: sportowiec {_sport_tier}, VO2max {_bp_vo2:.0f}]" if _is_sport else ""
        L.append(f"3b. Wzorzec oddychania \u2014 Breathing Pattern (E07 v2.0){_ctx_tag}:")
        L.append(f"\u2022 BF: rest [{e07.get('bf_rest','?')}] \u2192 peak [{e07.get('bf_peak','?')}] /min"
                  f"  |  VT: rest [{e07.get('vt_rest_L','?')}] \u2192 peak [{e07.get('vt_peak_L','?')}] L")
        if e07.get("bf_at_vt1"):
            L.append(f"  \u21b3 przy VT1: BF [{e07['bf_at_vt1']}] /min  |  VT [{e07.get('vt_at_vt1_L','?')}] L")
        if e07.get("bf_at_vt2"):
            L.append(f"  \u21b3 przy VT2: BF [{e07['bf_at_vt2']}] /min  |  VT [{e07.get('vt_at_vt2_L','?')}] L")

        # VT Plateau — sport context
        plat = e07.get("vt_plateau_vs_vt2", "?")
        plat_t = e07.get("vt_plateau_time_s")
        plat_l = e07.get("vt_plateau_level_L")
        if _is_sport:
            _pd = {"AT_VT2": "prawid\u0142owe \u2014 plateau VT przy VT2",
                   "BEFORE_VT2": "wczesne plateau VT (<VT2) \u2014 typowe u sportowc\u00f3w, VE ro\u015bnie przez BF (86% EA, Carey 2008)",
                   "AFTER_VT2": "p\u00f3\u017ane plateau VT (>VT2) \u2014 wysoka rezerwa oddechowa",
                   "VT2_UNKNOWN": "brak VT2 do por\u00f3wnania"}.get(plat, "")
        else:
            _pd = {"AT_VT2": "prawid\u0142owe \u2014 plateau VT przy VT2",
                   "BEFORE_VT2": "\u26a0 wczesne plateau VT (<VT2) \u2014 ograniczenie mechaniczne?",
                   "AFTER_VT2": "p\u00f3\u017ane plateau VT (>VT2) \u2014 wysoka rezerwa oddechowa",
                   "VT2_UNKNOWN": "brak VT2 do por\u00f3wnania"}.get(plat, "")
        if plat_t:
            m, s = int(plat_t // 60), int(plat_t % 60)
            L.append(f"\u2022 Plateau VT: t={m}:{s:02d} ({e07.get('vt_plateau_pct_exercise',0):.0f}% testu) "
                     f"| poziom [{plat_l}] L | {_pd}")

        # Strategy — sport context
        strat = e07.get("strategy", "?")
        if _is_sport:
            _sd = {"VT_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez obj\u0119to\u015b\u0107 oddechow\u0105 \u2014 efektywne",
                   "BF_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez cz\u0119sto\u015b\u0107 \u2014 ADAPTACJA SPORTOWA (Folinsbee 1983: elitarni kolarze BF 63/min; ACC 2021: BF 60-70 norma u EA)",
                   "BALANCED": "zr\u00f3wnowa\u017cony wk\u0142ad VT i BF"}.get(strat, "")
        else:
            _sd = {"VT_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez obj\u0119to\u015b\u0107 oddechow\u0105 \u2014 efektywne",
                   "BF_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez cz\u0119sto\u015b\u0107 \u2014 mniej efektywne, zwi\u0119ksza dead space",
                   "BALANCED": "zr\u00f3wnowa\u017cony wk\u0142ad VT i BF"}.get(strat, "")
        L.append(f"\u2022 Strategia wentylacji: [{strat}] (VT {e07.get('ve_from_vt_pct',0):.0f}% / BF {e07.get('ve_from_bf_pct',0):.0f}%)")
        L.append(f"  \u21b3 {_sd}")

        # Timing
        if e07.get("duty_cycle_rest") is not None:
            L.append(f"\u2022 Timing oddechowy:")
            L.append(f"  \u21b3 tI: rest [{e07['ti_mean_rest_s']}] \u2192 peak [{e07['ti_mean_peak_s']}] s"
                     f"  |  tE: rest [{e07['te_mean_rest_s']}] \u2192 peak [{e07['te_mean_peak_s']}] s")
            dc_peak = e07['duty_cycle_peak']
            dc_desc = ""
            if dc_peak and dc_peak < 0.35: dc_desc = " \u26a0 niski"
            elif dc_peak and dc_peak > 0.50: dc_desc = " \u2014 wyd\u0142u\u017cony wdech"
            else: dc_desc = " \u2014 norma"
            L.append(f"  \u21b3 Duty cycle (tI/tTot): rest [{e07['duty_cycle_rest']}] \u2192 peak [{dc_peak}]{dc_desc} (ref: 0.40-0.50)")
            if e07.get("mean_insp_flow_peak_Ls"):
                L.append(f"  \u21b3 Mean inspiratory flow peak: [{e07['mean_insp_flow_peak_Ls']}] L/s")

        # VD/VT — sport context
        if e07.get("vdvt_rest") is not None:
            traj = e07.get("vdvt_trajectory", "?")
            vdvt_r = e07.get('vdvt_rest', 0)
            vdvt_p = e07.get('vdvt_peak', 0)
            try:
                vr = float(vdvt_r)
                vp = float(vdvt_p)
            except:
                vr, vp = 0.3, 0.3

            if _is_sport and vp < 0.20:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje (prawid\u0142owy V/Q matching)",
                       "FLAT": "VD/VT stabilne \u2014 przy warto\u015bciach <0.20 to NORMA SPORTOWA (elitarna efektywno\u015b\u0107)",
                       "ABNORMAL_INCREASE": f"VD/VT wzrost {vr:.3f}\u2192{vp:.3f} \u2014 przy <0.20 klinicznie nieistotny"}.get(traj, "")
            elif _is_sport and vp < 0.25:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje",
                       "FLAT": "VD/VT stabilne \u2014 poni\u017cej progu klinicznego (<0.25), prawdopodobnie norma sportowa",
                       "ABNORMAL_INCREASE": f"VD/VT wzrost \u2014 ale warto\u015bci bezwzgl\u0119dne w normie ({vp:.3f}<0.25)"}.get(traj, "")
            else:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje (prawid\u0142owy V/Q matching)",
                       "FLAT": "\u26a0 VD/VT nie maleje \u2014 mo\u017cliwy V/Q mismatch",
                       "ABNORMAL_INCREASE": "\u26a0 VD/VT ro\u015bnie \u2014 patologiczny V/Q mismatch"}.get(traj, "")

            parts = [f"rest [{vdvt_r}]"]
            if e07.get("vdvt_at_vt1"): parts.append(f"VT1 [{e07['vdvt_at_vt1']}]")
            if e07.get("vdvt_at_vt2"): parts.append(f"VT2 [{e07['vdvt_at_vt2']}]")
            parts.append(f"peak [{vdvt_p}]")
            _vdvt_str = ' \u2192 '.join(parts)
            L.append(f"\u2022 Dead space (VD/VT est.): {_vdvt_str}")
            est_note = " (warto\u015bci estymowane z PETCO2)" if vr < 0.15 else ""
            _ref = "Norma ABG: rest ~0.30 \u2192 peak <0.20 (ATS 2003)"
            if _is_sport and vp < 0.20:
                _ref += " | U sportowc\u00f3w: warto\u015bci <0.10-0.15 cz\u0119ste"
            L.append(f"  \u21b3 {_ref}{est_note}")
            L.append(f"  \u21b3 Trajektoria: {_td}")

        # Tachypnea — sport context for BF threshold
        tachy = e07.get("tachypnea_vs_vt2", "NONE")
        if tachy != "NONE" and e07.get("tachypnea_time_s"):
            tt = e07["tachypnea_time_s"]
            m2, s2 = int(tt // 60), int(tt % 60)
            if _is_sport:
                _tach = {"BEFORE_VT2": "tachypnea przed VT2 \u2014 u sportowc\u00f3w BF>55 norma (ACC 2021), por\u00f3wnaj z BR",
                         "AT_VT2": "tachypnea przy VT2 \u2014 oczekiwany punkt przej\u015bcia",
                         "AFTER_VT2": "p\u00f3\u017ana \u2014 dobra rezerwa oddechowa"}.get(tachy, "")
            else:
                _tach = {"BEFORE_VT2": "\u26a0 wczesna tachypnea \u2014 potencjalny limiter wentylacyjny",
                         "AT_VT2": "prawid\u0142owe \u2014 tachypnea przy VT2",
                         "AFTER_VT2": "p\u00f3\u017ana \u2014 dobra rezerwa oddechowa"}.get(tachy, "")
            L.append(f"\u2022 Tachypnea (BF>50): t={m2}:{s2:02d} ({e07.get('tachypnea_pct_exercise',0):.0f}% testu) \u2014 {_tach}")
        elif tachy == "NONE":
            L.append(f"\u2022 Tachypnea (BF>50): nie wyst\u0105pi\u0142a \u2014 dobra kontrola oddychania")

        # RSBI
        if e07.get("rsbi_peak"):
            _rd = ""
            if e07["rsbi_peak"] > 40: _rd = " \u26a0 wysoki (rapid shallow breathing)"
            elif e07["rsbi_peak"] > 25: _rd = " \u2014 podwy\u017cszony"
            else: _rd = " \u2014 prawid\u0142owy"
            L.append(f"\u2022 RSBI (BF/VT): rest [{e07.get('rsbi_rest','?')}] \u2192 peak [{e07['rsbi_peak']}] bpm/L{_rd}")

        # PTVV
        if e07.get("ptvv") is not None:
            ptvv = e07["ptvv"]
            if ptvv > 0.25: _pv = "\u26a0 wysoka nieregularno\u015b\u0107 \u2014 podejrzenie BPD"
            elif ptvv > 0.15: _pv = "umiarkowana zmienno\u015b\u0107"
            else: _pv = "regularny wzorzec"
            L.append(f"\u2022 PTVV (nieregularno\u015b\u0107 VT): [{ptvv}] \u2014 {_pv}")
            L.append(f"  \u21b3 Ref: Kn\u00f6pfel 2024 (norma <0.15)")

        # Sighs
        if e07.get("sigh_count", 0) > 0:
            L.append(f"\u2022 Westchnienia: [{e07['sigh_count']}] ({e07['sigh_rate_per_min']}/min)")
            if e07["sigh_count"] > 3:
                L.append(f"  \u21b3 \u26a0 Cz\u0119ste westchnienia \u2014 Periodic Deep Sighing?")

        # Breathing pattern classification — sport context
        bp = e07.get("breathing_pattern", "?")
        if _is_sport:
            _bp = {"NORMAL": "Prawid\u0142owy wzorzec oddychania",
                   "BORDERLINE": "Graniczny \u2014 jeden marker nieprawid\u0142owy (mo\u017ce by\u0107 norma sportowa)",
                   "DYSFUNCTIONAL_BPD": "Algorytm: DYSFUNKCYJNY \u2014 REINTERPRETACJA SPORTOWA: "
                       f"przy VO2max {_bp_vo2:.0f} i VE/VCO2 {_bp_ve_slope:.0f} wzorzec jest prawdopodobnie adaptacj\u0105, nie patologi\u0105",
                   "RAPID_SHALLOW": "Szybki p\u0142ytki oddech \u2014 u sportowca mo\u017ce wynika\u0107 z BF-dominant strategy",
                   "PERIODIC": "\u26a0 Oddychanie periodyczne"}.get(bp, bp)
        else:
            _bp = {"NORMAL": "Prawid\u0142owy wzorzec oddychania",
                   "BORDERLINE": "\u26a0 Graniczny \u2014 jeden marker nieprawid\u0142owy",
                   "DYSFUNCTIONAL_BPD": "\u26a0 DYSFUNKCYJNY \u2014 erratyczny VT/BF (BPD)",
                   "RAPID_SHALLOW": "\u26a0 Szybki p\u0142ytki oddech \u2014 wczesna dominacja BF",
                   "PERIODIC": "\u26a0 Oddychanie periodyczne"}.get(bp, bp)
        L.append(f"\u2022 Klasyfikacja: [{bp}] \u2014 {_bp}")

        # Flags — sport context annotation
        fl = e07.get("flags", [])
        if fl:
            if _is_sport:
                _sport_flags = {'BF_DOMINANT_STRAT','EARLY_VT_PLATEAU','VDVT_NO_DECREASE','VDVT_PARADOXICAL_RISE','HIGH_BF_VAR_REST','HIGH_BF_VAR_EX','VDVT_EST_LOW'}
                _clinical = [f for f in fl if f not in _sport_flags]
                _sport = [f for f in fl if f in _sport_flags]
                parts = []
                if _sport:
                    parts.append(f"sportowe (norma): {', '.join(_sport)}")
                if _clinical:
                    parts.append(f"do oceny: {', '.join(_clinical)}")
                L.append(f"\u2022 Flagi: [{' | '.join(parts)}]")
            else:
                L.append(f"\u2022 Flagi: [{', '.join(fl)}]")

        return "\n".join(L)

    @staticmethod
    def _nirs_report(ct):
        r12 = ct.get('_e12_raw', {})
        if not r12 or r12.get('status') == 'NO_SIGNAL':
            return ""
        def _n(key, fb=None):
            v = r12.get(key)
            if v is None: return fb
            try: return float(v)
            except: return fb
        L = []
        L.append("")
        L.append("3c. Analiza NIRS — Saturacja mięśniowa SmO2 (E12 v2.0)")
        L.append(f"    Sensor: Moxy via MetaMax | Kanał: {r12.get('channel','?')} | Jakość: {r12.get('signal_quality','?')}")
        L.append("")
        rest=_n('smo2_rest'); mn=_n('smo2_min'); pk=_n('smo2_at_peak')
        d_abs=_n('desat_total_abs'); d_pct=_n('desat_total_pct')
        if rest is not None and mn is not None:
            L.append("    WARTOŚCI KLUCZOWE:")
            L.append(f"    • SmO2 spoczynek: {rest:.1f}% → min: {mn:.1f}% (peak exercise: {(pk if pk else mn):.1f}%)")
            ds = f"↓ {d_abs:.1f} p.p." if d_abs else "?"
            dp = f" ({d_pct:.1f}% desaturacji)" if d_pct else ""
            L.append(f"    • Desaturacja: {ds}{dp}")
            if d_pct is not None:
                if d_pct > 50: L.append("      ✓ Bardzo wysoka ekstrakcja O2 — profil elitarny")
                elif d_pct > 30: L.append("      ✓ Dobra ekstrakcja O2 — profil sportowca")
                elif d_pct < 15: L.append("      ⚠ Niska desaturacja — możliwy wysoki ATT lub słaba ekstrakcja")
        t27=_n('time_below_27_s')
        if t27 and t27 > 0:
            L.append(f"    • Czas poniżej 27% SmO2: {t27:.0f}s (ref: <27% = strefa MMSS)")
        v1=_n('smo2_at_vt1'); v2=_n('smo2_at_vt2')
        if v1 or v2:
            L.append("    SmO2 W PROGACH:")
            if v1: L.append(f"    • Przy VT1: {v1:.1f}%")
            if v2: L.append(f"    • Przy VT2: {v2:.1f}%")
        b1=_n('bp1_time_s'); b2=_n('bp2_time_s')
        if b1 is not None:
            L.append("    BREAKPOINTY SmO2 (regresja segmentowa):")
            bs1=_n('bp1_smo2'); bv1=_n('bp1_vs_vt1_s')
            vs1 = ""
            if bv1 is not None: vs1 = f" | {abs(bv1):.0f}s {'PRZED' if bv1<0 else 'PO'} VT1"
            m,s=divmod(int(b1),60)
            L.append(f"    • BP1 (Faza1→2): t={m}:{s:02d}, SmO2={(f'{bs1:.1f}' if bs1 else '?')}%{vs1}")
        if b2 is not None:
            bs2=_n('bp2_smo2'); bv2=_n('bp2_vs_vt2_s')
            vs2 = ""
            if bv2 is not None: vs2 = f" | {abs(bv2):.0f}s {'PRZED' if bv2<0 else 'PO'} VT2"
            m,s=divmod(int(b2),60)
            L.append(f"    • BP2 (Faza2→3): t={m}:{s:02d}, SmO2={(f'{bs2:.1f}' if bs2 else '?')}%{vs2}")
        rate=_n('desat_rate_phase2')
        if rate is not None: L.append(f"    • Tempo desaturacji (Faza 2): {rate:.2f} %/min")
        rec=_n('smo2_recovery_peak'); hrt=_n('hrt_s'); ov=_n('overshoot_abs'); rx=_n('reox_rate')
        if rec is not None:
            L.append("    RECOVERY (hyperemia powysiłkowa):")
            os = f" (overshoot: +{ov:.1f} p.p.)" if ov else ""
            L.append(f"    • SmO2 peak recovery: {rec:.1f}%{os}")
            if hrt is not None:
                hi=""
                if hrt<15: hi=" ✓ bardzo szybka reperfuzja"
                elif hrt<30: hi=" ✓ dobra reperfuzja"
                elif hrt>60: hi=" ⚠ wolna reperfuzja"
                L.append(f"    • Half-Recovery Time: {hrt:.1f}s{hi}")
            if rx is not None: L.append(f"    • Tempo reoxygenacji: {rx:.3f} %/s")
        fl=r12.get('flags',[])
        if fl: L.append(f"    Flagi: {', '.join(fl)}")
        return "\n".join(L)


    @staticmethod
    def _drift_report(ct):
        """E13 Drift/Coupling report section."""
        r13 = ct.get('_e13_raw', {})
        if not r13 or r13.get('status') != 'OK':
            return ''
        L = []
        # Sport context for E13
        _d_vo2 = ct.get('_interp_vo2_mlkgmin')
        _d_vs = None
        try:
            _d_vsr = ct.get('VE_VCO2_slope')
            if _d_vsr and str(_d_vsr) not in ('-','[BRAK]','None',''): _d_vs = float(_d_vsr)
        except: pass
        _d_sport = (_d_vo2 is not None and _d_vo2 >= 45 and _d_vs is not None and _d_vs < 30)
        _d_sc = r13.get('slope_class', '-')
        _d_o2pt = r13.get('o2pulse_trajectory', '')
        _d_o2p_late = r13.get('o2pulse_late')
        # Reinterpret slope_class for athletes
        _d_sc_txt = _d_sc
        if _d_sport and _d_sc in ('VERY_LOW', 'LOW_HIGH_SV') and _d_o2pt == 'RISING':
            _d_sc_txt = f"{_d_sc} → ADAPTACJA SPORTOWA (wysoka SV, O2-pulse rosnący)"
        L.append("3e. CV Drift / Coupling (E13 v2.0):")
        L.append(f"  HR-VO2 slope: [{r13.get('hr_vo2_slope','-')}] bpm/L/min (R2={r13.get('hr_vo2_r2','-')}) | {_d_sc_txt} | Coupling: {r13.get('coupling_quality','-')}")
        bp = r13.get('breakpoint_time_sec')
        if bp:
            m, s = int(bp // 60), int(bp % 60)
            _bp_vs = r13.get('bp_vs_vt2', '-')
            _bp_txt = _bp_vs
            if _d_sport and _bp_vs == 'BEFORE_VT2' and _d_o2pt == 'RISING':
                _bp_txt = f"BEFORE_VT2 — u sportowca z rosnącym O2-pulse: wczesna optymalizacja SV, nie dekompensacja"
            L.append(f"  HR-VO2 breakpoint: t={m}:{s:02d} ({r13.get('breakpoint_pct_exercise','-')}%) | vs VT2: {_bp_txt}")
        ci = r13.get('chronotropic_index')
        ci_txt = f"{ci}" if ci else '-'
        L.append(f"  Chronotropic: CI={ci_txt} ({r13.get('ci_class','-')}) | HR reserve: {r13.get('hr_reserve_pct','-')}% | {r13.get('hr_pattern','-')}")
        L.append(f"  HR: rest {r13.get('hr_rest','-')} -> peak {r13.get('hr_peak','-')} bpm | pred {r13.get('hr_pred_max','-')} ({r13.get('pct_pred_hr','-')}%)")
        L.append(f"  O2P drift: {r13.get('o2pulse_early','-')} -> {r13.get('o2pulse_mid','-')} -> {r13.get('o2pulse_late','-')} ml/beat ({r13.get('o2pulse_drift_pct','-')}%)")
        L.append(f"  Trajektoria O2P: {r13.get('o2pulse_trajectory','-')} | Klinicznie: {r13.get('o2pulse_clinical','-')}")
        vwr = r13.get('vo2_wr')
        if vwr:
            L.append(f"  VO2/WR: {vwr.get('vo2_wr_slope','-')} {vwr.get('vo2_wr_unit','')} (R2={vwr.get('vo2_wr_r2','-')}) | {vwr.get('slope_class','-')}")
        sd = r13.get('step_drift')
        if sd:
            L.append(f"  Intra-step drift: max {sd.get('max_drift_pct','-')}% | mean {sd.get('mean_drift_pct','-')}%")
        fl = r13.get('flags', [])
        _d_ov = r13.get('overall_cv_coupling', '-')
        if _d_sport and _d_o2pt == 'RISING':
            _d_sport_fl = {'VERY_LOW_HR_VO2_SLOPE', 'EARLY_LINEARITY_BREAK'}
            _d_cfl = [f for f in fl if f not in _d_sport_fl]
            _d_sfl = [f for f in fl if f in _d_sport_fl]
            if _d_sfl and not _d_cfl:
                _d_ov = f"SPORT_NORM (reinterpretacja: niski slope + wczesny BP = wysoka SV u sportowca)"
            _d_fl_parts = []
            if _d_sfl: _d_fl_parts.append(f"sport (wysoka SV): {', '.join(_d_sfl)}")
            if _d_cfl: _d_fl_parts.append(f"kliniczne: {', '.join(_d_cfl)}")
            L.append(f"  Overall: {_d_ov} | Flagi: [{' | '.join(_d_fl_parts)}]" if _d_fl_parts else f"  Overall: {_d_ov} | Flagi: BRAK")
        else:
            L.append(f"  Overall: {_d_ov} | Flagi: {', '.join(fl) if fl else 'BRAK'}")
        return '\n'.join(L)

    @staticmethod
    def _lactate_report(ct):
        """E11 Lactate report section."""
        r11 = ct.get('_e11_raw', {})
        if not r11 or r11.get('status') != 'OK':
            return ''
        L = []
        L.append("3d. Analiza Laktatowa (E11 v2.0):")
        n = r11.get('n_points', 0)
        bl = r11.get('la_baseline', 0)
        pk = r11.get('la_peak', 0)
        L.append(f"  Punkty pomiarowe: {n} | Baseline: {bl:.2f} mmol/L | Peak: {pk:.2f} mmol/L")
        if r11.get('lt1_method'):
            hr1 = f"HR {round(r11['lt1_hr_bpm'])}" if r11.get('lt1_hr_bpm') else ''
            sp1 = f" Speed {round(r11['lt1_speed_kmh'],1)}" if r11.get('lt1_speed_kmh') else ''
            la1 = f" La={round(r11['lt1_la_mmol'],2)}" if r11.get('lt1_la_mmol') else ''
            t1 = r11.get('lt1_time_sec')
            ts1 = f"t={int(t1//60)}:{int(t1%60):02d}" if t1 else ''
            L.append(f"  LT1: {r11['lt1_method']} | {ts1} | {hr1}{sp1}{la1}")
        if r11.get('lt2_method'):
            hr2 = f"HR {round(r11['lt2_hr_bpm'])}" if r11.get('lt2_hr_bpm') else ''
            sp2 = f" Speed {round(r11['lt2_speed_kmh'],1)}" if r11.get('lt2_speed_kmh') else ''
            la2 = f" La={round(r11['lt2_la_mmol'],2)}" if r11.get('lt2_la_mmol') else ''
            t2 = r11.get('lt2_time_sec')
            ts2 = f"t={int(t2//60)}:{int(t2%60):02d}" if t2 else ''
            L.append(f"  LT2: {r11['lt2_method']} | {ts2} | {hr2}{sp2}{la2}")
        conc = r11.get('concordance', {})
        for key in ['lt1_consensus', 'lt2_consensus']:
            c = conc.get(key, {})
            if c.get('n_methods', 0) > 0:
                L.append(f"  {c.get('label','')}: consensus {c.get('n_methods')} metod, spread {c.get('spread_sec','?')}s, confidence {c.get('confidence','?')}")
        vt = r11.get('vt_comparison', {})
        if vt.get('vt1_vs_lt1_diff_sec') is not None:
            ag = 'ZGODNE' if vt.get('vt1_vs_lt1_agreement') else 'ROZBIEZNE'
            L.append(f"  VT1 vs LT1: diff {vt['vt1_vs_lt1_diff_sec']}s ({ag})")
        if vt.get('vt2_vs_lt2_diff_sec') is not None:
            ag = 'ZGODNE' if vt.get('vt2_vs_lt2_agreement') else 'ROZBIEZNE'
            L.append(f"  VT2 vs LT2: diff {vt['vt2_vs_lt2_diff_sec']}s ({ag})")
        qw = r11.get('quality_warnings', [])
        if qw:
            L.append(f"  Uwagi: {'; '.join(qw)}")
        return '\n'.join(L)

    @staticmethod
    def _e18_crossval_text(ct):
        """Generate E18 cross-validation text section."""
        if ct.get('E18_status') != 'OK':
            return ""

        lines = []
        lines.append("CROSS-WALIDACJA VT↔LT (E18)")

        e18_sum = ct.get('E18_summary', {})
        lines.append(f"  Ocena: {e18_sum.get('overall_class', '').replace('_',' ')} "
                     f"({e18_sum.get('overall_score_pct', 0):.0f}%)")

        # L1
        l1 = ct.get('E18_layer1', {})
        for prefix, label in [('vt1','VT1'), ('vt2','VT2')]:
            d = l1.get(prefix, {})
            if d.get('status') not in ('NO_VT', None):
                lines.append(f"  {d.get('emoji','')} La@{label}: "
                           f"{d.get('la_interpolated_mmol','-')} mmol/L "
                           f"[{d.get('status','')}] (opt: {d.get('optimal_range',[])})")

        # L2
        l2 = ct.get('E18_layer2', {})
        l2s = l2.get('summary', {})
        if l2s.get('domains_tested', 0) > 0:
            lines.append(f"  Domeny: {l2s.get('domains_confirmed',0)}/{l2s.get('domains_tested',0)} potwierdzone")

        # L3
        l3 = ct.get('E18_layer3', {})
        for prefix, label in [('threshold_1','VT1↔LT1'), ('threshold_2','VT2↔LT2')]:
            d = l3.get(prefix, {})
            if d.get('status') not in ('INCOMPLETE', None):
                lines.append(f"  {d.get('emoji','')} {label}: "
                           f"Δt={d.get('delta_time_sec',0):+.0f}s "
                           f"[{d.get('time_concordance','')}]")

        # Interpretation
        interp = e18_sum.get('interpretation', '')
        if interp:
            lines.append(f"  → {interp}")

        return "\n".join(lines)


    @staticmethod
    def _render_charts_html(ct):
        """Generate interactive charts section with buttons and Canvas JS."""
        import json as _json

        _lac_pts = ct.get('_chart_lactate_points', [])
        _lac_poly = ct.get('_chart_lactate_poly', {})
        _nirs_traces = ct.get('_chart_nirs_traces', [])
        _has_lactate = len(_lac_pts) >= 3
        _has_nirs = len(_nirs_traces) > 0
        _has_gas = len(ct.get('_chart_gas_time', [])) > 10
        _has_substrate = len(ct.get('_chart_sub_time', [])) > 10 and max(ct.get('_chart_sub_fat', [0]) or [0]) > 0
        _has_kinetics = len(ct.get('_chart_full_time', [])) > 10
        _has_protocol = len(ct.get('_chart_protocol_steps', [])) > 0

        if not _has_lactate and not _has_nirs and not _has_gas and not _has_substrate and not _has_kinetics and not _has_protocol:
            return ''

        _chart_data = {
            'lactate': {
                'points': [{'time_sec': p.get('time_sec', 0), 'la': p.get('la', p.get('lactate_mmol', 0))} for p in _lac_pts],
                'poly_time': _lac_poly.get('time_sec', []),
                'poly_fit': _lac_poly.get('lactate_fit', []),
                'lt1_time': ct.get('_chart_lt1_time'),
                'lt1_la': ct.get('_chart_lt1_la'),
                'lt2_time': ct.get('_chart_lt2_time'),
                'lt2_la': ct.get('_chart_lt2_la'),
                'lt1_method': ct.get('_chart_lt1_method', ''),
                'lt2_method': ct.get('_chart_lt2_method', ''),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
                'vt1_hr': ct.get('_chart_vt1_hr'),
                'vt2_hr': ct.get('_chart_vt2_hr'),
                'e18_l1': ct.get('_chart_e18_l1', {}),
            },
            'nirs': {
                'traces': [{'time_sec': tr.get('time_sec', []), 'smo2': tr.get('smo2_pct', tr.get('smo2', []))} for tr in _nirs_traces],
                'channels': ct.get('_chart_nirs_channels', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'gas': {
                'time': ct.get('_chart_gas_time', []),
                'vo2': ct.get('_chart_gas_vo2', []),
                'vco2': ct.get('_chart_gas_vco2', []),
                'hr': ct.get('_chart_gas_hr', []),
                'vt1_vo2': ct.get('VT1_VO2_mlmin'),
                'vt2_vo2': ct.get('VT2_VO2_mlmin'),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'substrate': {
                'time': ct.get('_chart_sub_time', []),
                'cho': ct.get('_chart_sub_cho', []),
                'fat': ct.get('_chart_sub_fat', []),
                'hr': ct.get('_chart_sub_hr', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'kinetics': {
                'time': ct.get('_chart_full_time', []),
                'vo2': ct.get('_chart_full_vo2', []),
                'hr': ct.get('_chart_full_hr', []),
                'speed': ct.get('_chart_full_speed', []),
                'power': ct.get('_chart_full_power', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
                't_stop': ct.get('_chart_t_stop'),
                'vo2peak': ct.get('_chart_vo2peak'),
                'protocol_steps': ct.get('_prot_steps', []),
                'protocol_name': ct.get('_prot_name', ''),
            }
        }
        _cj = _json.dumps(_chart_data, default=str)

        h = '<div style="margin:24px 0 12px;border-top:2px solid #e2e8f0;padding-top:18px;">'
        h += '<div style="font-size:15px;font-weight:700;color:#1e293b;margin-bottom:12px;">'
        h += '&#128202; Wykresy interaktywne</div>'


        _sq = chr(39)
        _has_protocol = len(ct.get('_prot_steps', [])) > 0
        if _has_protocol:
            h += '<button onclick="toggleChart(' + _sq + 'proto' + _sq + ')" id="btn_proto" data-color="#6366f1" style="margin:4px 6px;padding:8px 18px;border:2px solid #6366f1;border-radius:8px;background:#fff;color:#6366f1;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Protokół</button>'
        if _has_kinetics:
            h += '<button onclick="toggleChart(' + _sq + 'kinetics' + _sq + ')" id="btn_kinetics" data-color="#059669" style="margin:4px 6px;padding:8px 18px;border:2px solid #059669;border-radius:8px;background:#fff;color:#059669;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ VO2 Kinetics</button>'
        if _has_gas:
            h += '<button onclick="toggleChart(' + _sq + 'vslope' + _sq + ')" id="btn_vslope" data-color="#0891b2" style="margin:4px 6px;padding:8px 18px;border:2px solid #0891b2;border-radius:8px;background:#fff;color:#0891b2;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ V-slope</button>'
        if _has_substrate:
            h += '<button onclick="toggleChart(' + _sq + 'fatchho' + _sq + ')" id="btn_fatchho" data-color="#d97706" style="margin:4px 6px;padding:8px 18px;border:2px solid #d97706;border-radius:8px;background:#fff;color:#d97706;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Fat/CHO</button>'
        if _has_lactate:
            h += '<button onclick="toggleChart(' + _sq + 'lac' + _sq + ')" id="btn_lac" data-color="#dc2626" style="margin:4px 6px;padding:8px 18px;border:2px solid #dc2626;border-radius:8px;background:#fff;color:#dc2626;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Krzywa mleczanowa</button>'
        if _has_nirs:
            h += '<button onclick="toggleChart(' + _sq + 'nirs' + _sq + ')" id="btn_nirs" data-color="#2563eb" style="margin:4px 6px;padding:8px 18px;border:2px solid #2563eb;border-radius:8px;background:#fff;color:#2563eb;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ NIRS SmO2</button>'
        if _has_lactate and _has_nirs:
            h += '<button onclick="toggleChart(' + _sq + 'dual' + _sq + ')" id="btn_dual" data-color="#7c3aed" style="margin:4px 6px;padding:8px 18px;border:2px solid #7c3aed;border-radius:8px;background:#fff;color:#7c3aed;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Laktat + SmO2</button>'
        h += '<div id="chart_container" style="display:none;margin-top:14px;"></div></div>'
        h += '<script>\nconst CD=' + _cj + ';\n' + CHART_JS + '</script>'

        return h

    @staticmethod
    def render_html_report(ct):
        g = ct.get
        def v(key, default='-'):
            val = g(key, default)
            if val in (None, '', '[BRAK]', 'None', 'nan'): return default
            return str(val)
        def esc(s):
            return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        def _n(val, fmt='.1f', default='-'):
            try:
                n = float(val)
                return f'{n:{fmt}}'
            except: return default
        def _fmt_dur(sec):
            try:
                s=int(float(sec)); return f'{s//60}:{s%60:02d}'
            except: return '-'
        def badge(text, color='#64748b', bg=None):
            if not bg:
                bg = color + '18'
            return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{bg};color:{color};">{text}</span>'
        def row_item(label, value, assessment='', comment='', unit=''):
            colors = {
                'EXCELLENT':'#16a34a','WYBITNA':'#16a34a','HIGH':'#16a34a','NORMA':'#16a34a',
                'GOOD':'#16a34a','MAXIMAL':'#16a34a','VERY_GOOD':'#16a34a','RISING':'#16a34a',
                'PASS':'#16a34a','TIGHT':'#16a34a','YES':'#16a34a','TRAINED':'#16a34a',
                'INCREASING':'#16a34a','SUPERIOR':'#16a34a','SUPRAMAXIMAL':'#16a34a',
                'SPORT_NORM':'#2563eb','SPORT':'#2563eb','BF_DOMINANT':'#2563eb','FLAT':'#2563eb',
                'MODERATE':'#d97706','FAIR':'#d97706','BORDERLINE':'#d97706','WIDE':'#d97706',
                'NORMAL':'#16a34a','VC-I':'#16a34a','VC-II':'#d97706','VC-III':'#ea580c','VC-IV':'#dc2626',
                'POOR':'#dc2626','ABNORMAL':'#dc2626','SLOW':'#ea580c','VERY_SLOW':'#dc2626',
                'VERY_WIDE':'#dc2626','LOW':'#dc2626','FAIL':'#dc2626','DECLINING':'#dc2626',
                'RECREATIONAL':'#d97706','BALANCED':'#2563eb','SV_LIMITATION':'#ea580c',
                'N/A':'#94a3b8','TODO':'#94a3b8','NO_DATA':'#94a3b8','INSUFFICIENT_DATA':'#94a3b8',
                'LOW_HIGH_SV':'#ea580c','NO_RCP_DROP':'#d97706',
            }
            _short = {
                'BORDERLINE':'GRANICZNY','INSUFFICIENT':'NIEWYSTAR.','SUPRAMAXIMAL':'SUPRAMAX',
                'LOW_HIGH_SLOPE':'NISKI↑','HIGH_SLOPE':'WYSOKI↑',
                'VERY_SLOW':'B.WOLNY','VERY_GOOD':'B.DOBRY','VERY_WIDE':'B.SZEROKI',
                'BF_DOMINANT':'BF-DOM.','VT_DOMINANT':'VT-DOM.',
                'DECLINING':'SPADEK','CV_DECOUPLING':'DECOUPL.','NORMAL_DECOUPLING':'NORM.DECOUPL.','AFTER_VT2':'PO VT2','BEFORE_VT2':'PRZED VT2',
                'AFTER_VT1':'PO VT1','BEFORE_VT1':'PRZED VT1','NONE':'BRAK',
                'MILD_ABNORMALITY':'ŁAGODNA ANOM.','EXCELLENT':'DOSKONAŁA',
                'RISING':'ROSNĄCY','INCREASING':'ROSNĄCY','NORMAL':'NORMA',
                'HIGH_DESAT_ELITE':'WYS.DESAT.','LATE_BP1':'PÓŹNY BP1',
                'SLOW_RECOVERY':'WOLNA REC.','LARGE_OVERSHOOT':'DUŻY OVERSHOOT',
                'FREQUENT_SIGHS':'CZĘSTE WESTCH.','EARLY_VT_PLATEAU':'WCZESNE PLAT.VT',
                'HIGH_BF_VAR_EX':'WYS.ZMIEN.BF','PARADOXICAL':'PARADOKS.',
                'SPORT_NORM':'SPORT','DISPROPORTIONATE':'DYSPROP.',
                'DYSFUNCTIONAL_BPD':'DYSFUNKC.','TACHYPNEIC_RISE':'TACHYPNEA',
                'NO_RCP_DROP':'BEZ SPADKU','PARADOXICAL_RISE':'PARAD.↑',
                'SV_LIMITATION':'SV-LIMIT','LOW_HIGH_SV':'NIS-WYS.SV',
                'INSUFFICIENT_DATA':'B/D',
            }
            a_str = str(assessment).strip()
            a_disp = _short.get(a_str.upper().replace(' ','_'), a_str)
            ac = colors.get(a_str.upper().replace(' ','_'), '#64748b')
            bdg = f'<span style="color:{ac};font-weight:600;font-size:10px;white-space:nowrap;">{a_disp}</span>' if a_str and a_str != '-' else ''
            vu = f'{value} {unit}'.strip() if unit else str(value)
            cmt = f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">{comment}</div>' if comment else ''
            return (
                f'<div style="padding:5px 0;border-bottom:1px solid #f1f5f9;display:flex;align-items:flex-start;gap:8px;">'
                f'<div style="flex:0 0 130px;font-size:12px;color:#475569;font-weight:500;">{label}</div>'
                f'<div style="flex:1;font-size:13px;font-weight:600;color:#0f172a;">{vu}{cmt}</div>'
                f'<div style="flex:0 0 auto;min-width:70px;text-align:right;">{bdg}</div></div>'
            )

        def gauge_svg(score, label, size=80, is_pct=False, subtitle=''):
            pct=max(0,min(100,float(score) if score else 0))
            if pct>=85: col='#16a34a'
            elif pct>=70: col='#65a30d'
            elif pct>=55: col='#d97706'
            elif pct>=40: col='#ea580c'
            else: col='#dc2626'
            if pct == 0: col='#cbd5e1'
            r=size*0.4;cx=size/2;cy=size*0.5
            circ=2*3.14159*r;dash=circ*pct/100;gap=circ-dash
            disp = f'{int(score)}{"%" if is_pct else ""}' if score else '\u2014'
            h = size + (24 if subtitle else 14)
            sub_txt = f'<text x="{cx}" y="{size+20}" text-anchor="middle" font-size="7" fill="#94a3b8">{subtitle}</text>' if subtitle else ''
            return f'<svg width="{size}" height="{h}" viewBox="0 0 {size} {h}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="6"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="{dash} {gap}" stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/><text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="700" fill="{col}">{disp}</text><text x="{cx}" y="{cy+r+10}" text-anchor="middle" font-size="8" font-weight="600" fill="#475569">{label}</text>{sub_txt}</svg>'

        def panel_card(title, icon, rows_html):
            return f'''<div style="background:white;border-radius:10px;border:1px solid #e2e8f0;padding:14px;flex:1;min-width:300px;">
  <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;padding-bottom:6px;border-bottom:2px solid #e2e8f0;">{icon} {title}</div>
  {rows_html}
</div>'''

        def section_title(text, num=''):
            pre = f'{num}. ' if num else ''
            return f'<div class="section-title">{pre}{text}</div>'

        def obs_bullet(text, severity='info'):
            colors = {'critical':'#dc2626','warning':'#d97706','sport':'#2563eb','info':'#16a34a','positive':'#16a34a'}
            dots = {'critical':'🔴','warning':'🟡','sport':'🔵','info':'🟢','positive':'🟢'}
            col = colors.get(severity, '#16a34a')
            dot = dots.get(severity, '🟢')
            return f'<div style="font-size:12px;color:#334155;padding:3px 0;">{dot} {text}</div>'

        # =====================================================================
        # GATHER ALL DATA
        # =====================================================================
        _raw_name = str(v('athlete_name','')).strip()
        name = _raw_name if _raw_name and _raw_name not in ('AUTO','Nieznany Zawodnik','-','') else (str(v('athlete_id','')).strip() or 'Sportowiec')
        age = v('age_y'); sex_pl = 'M' if v('sex')=='male' else 'K'
        weight = v('body_mass_kg'); height = v('height_cm')
        protocol = v('protocol_name')
        modality = v('modality','run')
        test_date = v('test_date','')

        # E00
        e00 = g('_e00_raw',{})
        t_stop = e00.get('t_stop',0)
        dur_str = _fmt_dur(t_stop)

        # E01/E15 — Peak values
        vo2_rel = g('VO2max_ml_kg_min')
        vo2_abs = g('VO2max_L_min') or g('VO2max_abs_Lmin')
        e15 = g('_e15_raw',{})
        vo2_class = e15.get('vo2_class_pop','?')
        vo2_pctile = e15.get('vo2_percentile_approx',50)
        vo2_det = e15.get('vo2_determination','VO2peak')
        vo2_pct_pred = e15.get('vo2_pct_predicted')
        vo2_class_sport = e15.get('vo2_class_sport_desc','')
        rer_peak = g('RER_peak')
        rer_class = e15.get('rer_class','?')
        rer_desc = e15.get('rer_desc','')
        hr_peak = g('HR_peak')
        hr_pred = e15.get('hr_predicted_tanaka')
        hr_pct_pred = e15.get('hr_pct_predicted') or e15.get('pct_pred_hr') or (float(hr_peak)/float(hr_pred)*100 if hr_peak and hr_pred else None)
        ve_peak = g('VE_peak')
        plateau = e15.get('plateau_detected', False)
        plateau_note = e15.get('vo2_method_note','')
        hrr1_class_e15 = e15.get('hrr_1min_class','')
        hrr1_desc_e15 = e15.get('hrr_1min_desc','')

        # E02 — Thresholds
        e02 = g('_e02_raw', {})
        if not e02 or not isinstance(e02, dict):
            import copy
            e02 = {}
        vt1_hr = g('VT1_HR_bpm'); vt1_time = g('VT1_Time_s'); vt1_pct = g('VT1_pct_VO2peak')
        vt2_hr = g('VT2_HR_bpm'); vt2_time = g('VT2_Time_s'); vt2_pct = g('VT2_pct_VO2peak')
        vt1_conf = g('VT1_confidence',0); vt2_conf = g('VT2_confidence',0)
        vt1_hr_pct_max = e15.get('vt1_pct_hrmax')
        vt2_hr_pct_max = e15.get('vt2_pct_hrmax')
        vt1_speed = g('VT1_Speed'); vt2_speed = g('VT2_Speed')
        vt1_rer_val = e02.get('vt1_rer'); vt2_rer_val = e02.get('vt2_rer')
        vt1_vo2_abs = e02.get('vt1_vo2_mlmin'); vt2_vo2_abs = e02.get('vt2_vo2_mlmin')

        # E03 — VE/VCO2
        e03 = g('_e03_raw',{})
        ve_vco2_slope = g('VE_VCO2_slope') or e03.get('slope_to_vt2')
        vc_class = e03.get('ventilatory_class','?')
        vc_desc = e03.get('vc_description','')
        ve_vco2_nadir = e03.get('ve_vco2_nadir')
        ve_vco2_at_vt1 = e03.get('ve_vco2_at_vt1')
        ve_vco2_full = e03.get('slope_full')
        ve_vco2_to_vt1 = e03.get('slope_to_vt1')
        slope_pct_pred = e03.get('pct_predicted')

        # E04 — OUES
        e04 = g('_e04_raw',{})
        oues100 = e04.get('oues100'); oues90 = e04.get('oues90'); oues75 = e04.get('oues75')
        oues_vt1 = e04.get('oues_to_vt1'); oues_vt2 = e04.get('oues_to_vt2')
        oues_kg = e04.get('oues_per_kg')
        oues_pct = e04.get('oues_pct_hollenberg')
        oues_stab = e04.get('oues_submax_stability')

        # E05 — O2 Pulse
        e05 = g('_e05_raw',{})
        o2p_peak = e05.get('o2pulse_peak')
        o2p_vt1 = e05.get('o2pulse_at_vt1'); o2p_vt2 = e05.get('o2pulse_at_vt2')
        o2p_traj = e05.get('trajectory','?')
        o2p_pct = e05.get('pct_predicted_friend')
        sv_est = e05.get('estimated_sv_peak_ml')
        o2p_desc = e05.get('trajectory_desc','')

        # E06 — Economy / Efficiency
        e06 = g('_e06_raw',{})
        re_vt1 = e06.get('re_at_vt1'); re_vt2 = e06.get('re_at_vt2')
        re_class = e06.get('re_classification','')
        gain_below_vt1 = e06.get('gain_below_vt1'); gain_vt1 = e06.get('gain_at_vt1'); gain_vt2 = e06.get('gain_at_vt2')
        gain_unit = e06.get('gain_unit','')
        gain_z = e06.get('gain_z_score')
        lin_break = e06.get('linearity_break_time_s')
        delta_eff = e06.get('delta_efficiency_pct')

        # E07 — Breathing
        e07 = g('_e07_raw',{})
        bp_pattern = e07.get('breathing_pattern','?')
        bp_strategy = e07.get('strategy','?')
        bf_peak = e07.get('bf_peak'); bf_vt1 = e07.get('bf_at_vt1'); bf_vt2 = e07.get('bf_at_vt2')
        vt_peak = e07.get('vt_peak_L'); vt_vt1 = e07.get('vt_at_vt1_L'); vt_vt2 = e07.get('vt_at_vt2_L')
        ve_from_bf = e07.get('ve_from_bf_pct')
        rsbi = e07.get('rsbi_peak')
        vt_plat_t = e07.get('vt_plateau_time_s'); vt_plat_pct = e07.get('vt_plateau_pct_exercise')
        e07_flags = e07.get('flags',[])

        # E08 — HRR
        e08 = g('_e08_raw',{})
        hrr1 = e08.get('hrr_1min'); hrr3 = e08.get('hrr_3min')
        hrr1_class = e08.get('interpretation_1min','?')
        hrr3_class = e08.get('interpretation_3min','?')

        # E10 — Substrate
        e10 = g('_e10_raw',{})
        fatmax = e10.get('mfo_gmin'); fatmax_hr = e10.get('fatmax_hr')
        fatmax_pct_vo2 = e10.get('fatmax_pct_vo2peak')
        fatmax_pct_hr = e10.get('fatmax_pct_hrmax')
        fatmax_zone_lo = e10.get('fatmax_zone_hr_low'); fatmax_zone_hi = e10.get('fatmax_zone_hr_high')
        cop_hr = e10.get('cop_hr'); cop_pct_vo2 = e10.get('cop_pct_vo2peak'); cop_rer = e10.get('cop_rer')
        fat_vt1_pct = e10.get('fat_pct_at_vt1'); cho_vt1_pct = e10.get('cho_pct_at_vt1')
        fat_vt2_pct = e10.get('fat_pct_at_vt2'); cho_vt2_pct = e10.get('cho_pct_at_vt2')
        fat_vt1_g = e10.get('fat_gmin_at_vt1'); cho_vt1_g = e10.get('cho_gmin_at_vt1')
        total_kcal = e10.get('total_kcal'); total_fat = e10.get('total_fat_kcal'); total_cho = e10.get('total_cho_kcal')
        zone_sub = e10.get('zone_substrate',{})

        # E11 — Lactate
        e11 = g('_e11_raw',{})
        la_peak = e11.get('la_peak'); la_base = e11.get('la_baseline')
        lt1_hr = e11.get('lt1_hr_bpm'); lt1_time = e11.get('lt1_time_sec')
        lt2_hr = e11.get('lt2_hr_bpm'); lt2_time = e11.get('lt2_time_sec')
        lt1_method = e11.get('lt1_method',''); lt2_method = e11.get('lt2_method','')

        # E12 — NIRS
        e12 = g('_e12_raw',{})
        smo2_rest = e12.get('smo2_rest'); smo2_min = e12.get('smo2_min')
        smo2_vt1 = e12.get('smo2_at_vt1'); smo2_vt2 = e12.get('smo2_at_vt2')
        smo2_peak_val = e12.get('smo2_at_peak')
        desat_pct = e12.get('desat_total_pct'); desat_abs = e12.get('desat_total_abs')
        hrt_s = e12.get('hrt_s'); reox_rate = e12.get('reox_rate')
        overshoot = e12.get('overshoot_abs')
        smo2_recov = e12.get('smo2_recovery_peak')
        bp1_t = e12.get('bp1_time_s'); bp1_smo2 = e12.get('bp1_smo2')
        bp2_t = e12.get('bp2_time_s'); bp2_smo2 = e12.get('bp2_smo2')
        bp1_vs_vt1 = e12.get('bp1_vs_vt1_s'); bp2_vs_vt2 = e12.get('bp2_vs_vt2_s')
        e12_flags = e12.get('flags',[])
        e12_quality = e12.get('signal_quality','')

        # E13 — Cardiovascular coupling
        e13 = g('_e13_raw',{})
        ci = e13.get('chronotropic_index'); ci_class = e13.get('ci_class','?')
        hr_vo2_slope = e13.get('hr_vo2_slope'); slope_class = e13.get('slope_class','?')
        bp_time_e13 = e13.get('breakpoint_time_sec'); bp_pct_e13 = e13.get('breakpoint_pct_exercise')
        bp_vs_vt2_e13 = e13.get('bp_vs_vt2','')
        o2p_traj_e13 = e13.get('o2pulse_trajectory','')
        o2p_clinical = e13.get('o2pulse_clinical','')
        cv_coupling = e13.get('overall_cv_coupling','')

        # E14 — Recovery kinetics
        e14 = g('_e14_raw',{})
        t_half_vo2 = e14.get('t_half_vo2_s')
        tau_vo2 = e14.get('tau_s'); tau_r2 = e14.get('tau_r2')
        mrt = e14.get('mrt_s')
        rec_class = e14.get('recovery_class','?')
        rec_desc = e14.get('recovery_desc','')
        vo2dr = e14.get('vo2dr_s')
        dvo2_60 = e14.get('dvo2_60s_mlkgmin')
        t_half_vco2 = e14.get('t_half_vco2_s'); t_half_ve = e14.get('t_half_ve_s')
        pct_rec_60 = e14.get('pct_recovered_60s'); pct_rec_120 = e14.get('pct_recovered_120s')

        # E16 — Zones
        e16 = g('_e16_raw',{})
        zones_data = e16.get('zones',{})
        three_zone = e16.get('three_zone',{})
        aer_reserve = e16.get('aerobic_reserve_pct')
        thr_gap = e16.get('threshold_gap_bpm')
        thr_gap_pct = e16.get('threshold_gap_pct_hrmax')

        # E17 — Gas exchange
        e17 = g('_e17_raw',{})
        vdvt_rest = e17.get('vdvt_rest'); vdvt_peak = e17.get('vdvt_at_peak')
        vdvt_pattern = e17.get('vdvt_pattern','?')
        petco2_rest = e17.get('petco2_rest'); petco2_vt1 = e17.get('petco2_at_vt1')
        petco2_peak = e17.get('petco2_at_peak'); petco2_pattern = e17.get('petco2_pattern','?')
        peto2_nadir = e17.get('peto2_nadir')

        # E19 — Validation
        e19 = g('_e19_raw',{})
        val_score = e19.get('validity_score',0); val_grade = e19.get('validity_grade','?')
        con_score = e19.get('concordance_score',0); con_grade = e19.get('concordance_grade','?')
        ta = e19.get('temporal_alignment',{})
        ta_vt1 = ta.get('VT1',{}); ta_vt2 = ta.get('VT2',{})

        # Sport context
        _is_sport = False
        try:
            _bv = float(ve_vco2_slope) if ve_vco2_slope else 99
            _bvo2 = float(vo2_rel) if vo2_rel else 0
            _is_sport = _bvo2 >= 40 and _bv < 32
        except: pass

        # =====================================================================
        # BUILD HTML
        # =====================================================================

        h = f'''<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPET Report — {esc(name)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f8fafc;color:#0f172a;line-height:1.5;}}
.wrap{{max-width:920px;margin:0 auto;padding:16px;}}
.section{{margin-bottom:14px;}}
.card{{background:white;border-radius:10px;border:1px solid #e2e8f0;padding:16px;margin-bottom:10px;}}
.section-title{{font-size:14px;font-weight:700;color:#0f172a;padding-bottom:6px;margin-bottom:10px;border-bottom:2px solid #e2e8f0;}}
.flex-row{{display:flex;gap:12px;flex-wrap:wrap;}}
.sub-header{{margin:8px 0 4px;font-size:11px;font-weight:700;color:#475569;border-top:1px solid #e2e8f0;padding-top:6px;text-transform:uppercase;letter-spacing:0.5px;}}
table.ztable{{width:100%;border-collapse:collapse;font-size:11px;}}
table.ztable th{{padding:5px;text-align:left;background:#f8fafc;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;}}
table.ztable td{{padding:4px 5px;border-bottom:1px solid #f1f5f9;}}
@media print{{body{{background:white;}} .wrap{{max-width:100%;padding:8px;}} .card{{break-inside:avoid;}}}}
</style></head><body><div class="wrap">'''

        # ═══════════════════════════════════════════════════════════════
        # 0. HEADER
        # ═══════════════════════════════════════════════════════════════
        _logo_pro = "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iR8OTUllfeDVGX1dZS1JFUyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDkyOS41MiA3MTIuNDIiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIHN0cm9rZS13aWR0aDogMS41cHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMSwgLmNscy0yLCAuY2xzLTMsIC5jbHMtNCwgLmNscy01IHsKICAgICAgICBmaWxsOiBub25lOwogICAgICB9CgogICAgICAuY2xzLTEsIC5jbHMtMiwgLmNscy0zLCAuY2xzLTUgewogICAgICAgIHN0cm9rZS1taXRlcmxpbWl0OiAxMDsKICAgICAgfQoKICAgICAgLmNscy0xLCAuY2xzLTUgewogICAgICAgIHN0cm9rZTogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBzdHJva2U6IHJnYmEoMjU1LDI1NSwyNTUsMC44KTsKICAgICAgfQoKICAgICAgLmNscy0yLCAuY2xzLTMsIC5jbHMtNSB7CiAgICAgICAgc3Ryb2tlLXdpZHRoOiAzcHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMyB7CiAgICAgICAgc3Ryb2tlOiByZ2JhKDI1NSwyNTUsMjU1LDAuNSk7CiAgICAgIH0KCiAgICAgIC5jbHMtNiB7CiAgICAgICAgZmlsbDogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy03IHsKICAgICAgICBjbGlwLXBhdGg6IHVybCgjY2xpcHBhdGgtMSk7CiAgICAgIH0KCiAgICAgIC5jbHMtOCB7CiAgICAgICAgY2xpcC1wYXRoOiB1cmwoI2NsaXBwYXRoKTsKICAgICAgfQogICAgPC9zdHlsZT4KICAgIDxjbGlwUGF0aCBpZD0iY2xpcHBhdGgiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik03NTEuODYsMzE5LjkzaC0zNS4wN2wtMjYuMzktNzQuMTQtOC45MywxNi4yNi0zNC45Ni03My40NC0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41LDE4LjktNTUuMzMtNjYuNzktNDAuMjYtODIuMDhjNzIuMSw0LjIzLDEzOS4zLDM0LjMyLDE5MC44LDg1LjgyLDQxLjc1LDQxLjc1LDY5LjQzLDkzLjgzLDgwLjY3LDE1MC40N2wuMjEsMS4yNloiLz4KICAgIDwvY2xpcFBhdGg+CiAgICA8Y2xpcFBhdGggaWQ9ImNsaXBwYXRoLTEiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik00MzguNzIsODIuODFsLTEwNS45NCwxNjMuOTUtNDIuMTUtNjEuMDYtMjQuOTIsODEuMjItMzkuMDEsNDguNTctNS44NS0yMy42Ny0xNC41NCwzMS40OC0xLjczLTYuMjgtMjIuMTcsMjkuNTYtMTMuMTIuMDVjNi43Ny02Ny40LDM2LjMtMTI5Ljg5LDg0Ljg0LTE3OC40Myw1MC4wMS01MC4wMiwxMTQuODQtNzkuODUsMTg0LjU5LTg1LjM5WiIvPgogICAgPC9jbGlwUGF0aD4KICA8L2RlZnM+CiAgPCEtLSBiZyByZW1vdmVkIC0tPgogIDxnPgogICAgPGc+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTc3Mi40OSw1MjAuODRoLTcuODd2NjguNjdoNS44M2MyLjIzLDAsNC4yLS44Miw1LjktMi40OCwxLjctMS42NSwyLjYtMy42OSwyLjctNi4xMnYtNTMuNTFjMC0xLjg1LS42My0zLjM4LTEuOS00LjU5LTEuMjYtMS4yMS0yLjgyLTEuODctNC42Ny0xLjk3WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTYiIGQ9Ik03NzEuNzYsNDIzLjE1aC03LjE0djY2LjYzaDcuODdjMS44NSwwLDMuNC0uNjMsNC42Ny0xLjksMS4yNi0xLjI2LDEuOS0yLjc3LDEuOS00LjUydi01Mi45M2MwLTIuMDQtLjcxLTMuNzQtMi4xMS01LjEtMS40MS0xLjM2LTMuMTMtMi4wOS01LjE4LTIuMTlaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTY3Mi42MywzNzYuMkg2OS4zMnYyNTguNGg3ODV2LTI1OC40aC0xODEuNjlaTTE0NS43OCw0MjEuMTFoNy4xNGM0Ljg2LDAsNy4yOSwzLjIxLDcuMjksOS42MnY2My4xM2MwLDIuODItLjczLDUuMS0yLjE5LDYuODUtMS40NiwxLjc1LTMuMTYsMi42Mi01LjEsMi42MmgtNy4xNHYtODIuMjNaTTEwNi40Miw2MjAuNDJoLTIyLjkydi0yMzAuMDVoMjIuOTJ2MjMwLjA1Wk0zNTcuOTMsNTgyLjY2YzAsMi43Mi0uNzEsNC45Ni0yLjExLDYuNzEtMS40MSwxLjc1LTMuMTMsMi42Mi01LjE4LDIuNjJzLTMuNjItLjg4LTUuMDMtMi42MmMtMS40MS0xLjc1LTIuMTEtMy45OC0yLjExLTYuNzF2LTkzLjYxYzAtMy41LjctNi4wNSwyLjExLTcuNjUsMS40MS0xLjYsMy4wOC0yLjQxLDUuMDMtMi40MSwyLjA0LDAsMy43Ny44LDUuMTgsMi40MSwxLjQxLDEuNiwyLjExLDQuMTYsMi4xMSw3LjY1djkzLjYxWk0zNTguNzMsNjIwLjQyYzMuOTMtMi4xOCw3LjY1LTUuNzMsMTEuMTYtMTAuNjRsNC43NCwxMC42NGgtMTUuOVpNNDA2LjQ4LDYyMC40MmgtOS4xOXYtMTY5Ljg2aC0yMi42bC0zLjk0LDkuMTljLTQuNzYtNy43OC0xMC44NC0xMS42Ni0xOC4yMy0xMS42NmgtMTUuNDZjLTkuODIsMC0xNy43NywzLjEzLTIzLjg0LDkuNC02LjA4LDYuMjctOS4xMSwxNC4zMS05LjExLDI0LjEzdjEwNi43M2MwLDEwLjYsMy4xMywxOS4wMyw5LjQsMjUuMywyLjk2LDIuOTYsNi4zMiw1LjIxLDEwLjA3LDYuNzhoLTQ0Ljk5YzMuMzUtMS41MSw2LjQ0LTMuNjcsOS4yNi02LjQ4LDYuMDctNi4wNyw5LjExLTEzLjM5LDkuMTEtMjEuOTR2LTM0LjI2aC0zOS4zN3YyNS41MmMwLDIuNzItLjcxLDQuODYtMi4xMSw2LjQyLTEuNDEsMS41Ni0zLjA5LDIuMzMtNS4wMywyLjMzLTQuODYtLjM5LTcuMjktMy4zLTcuMjktOC43NXYtMzguMDVoNTMuOHYtNjYuNjNjMC04LjQ2LTIuOTctMTUuNjUtOC44OS0yMS41OC01LjkzLTUuOTMtMTMuMTItOC44OS0yMS41OC04Ljg5aC0zMi4yMmMtOC4zNiwwLTE1LjUzLDIuOTctMjEuNTEsOC44OS01Ljk4LDUuOTMtOC45NywxMy4xMi04Ljk3LDIxLjU4djExMy40NGMuMTksOC43NSwzLjI4LDE2LjExLDkuMjYsMjIuMDksMi43NSwyLjc1LDUuNzcsNC44NSw5LjA2LDYuMzRoLTc2LjM1di04Ni4wMmgyMC4yN2M1LjczLDAsMTEuMTUtMS4wMiwxNi4yNi0zLjA2LDUuMS0yLjA0LDkuMjYtNS40NywxMi40Ny0xMC4yOCwzLjIxLTQuODEsNC44MS0xMS4zNSw0LjgxLTE5LjYxdi03Ny4yOGMwLTcuNjgtMS42My0xNC4wNS00Ljg4LTE5LjEtMy4yNi01LjA1LTcuNDktOC44LTEyLjY5LTExLjIzLTQuMTEtMS45Mi04LjQyLTMuMDctMTIuOTMtMy40N2gyMzcuNHYyMzAuMDVaTTI0My4xOCw1MTQuMTR2LTI1LjY2YzAtMi43Mi43LTQuOTgsMi4xMS02Ljc4LDEuNDEtMS44LDMuMTMtMi43LDUuMTgtMi43czMuNjIuOSw1LjAzLDIuN2MxLjQxLDEuOCwyLjExLDQuMDYsMi4xMSw2Ljc4djI1LjY2aC0xNC40NFpNNDQ1Ljg1LDYyMC40MnYtNjAuMDdsLjczLTEuMzEsMjAuOTQsNjEuMzhoLTIxLjY3Wk01MjIuNTIsNjIwLjQyaC05LjgxbC00MC4wNC0xMDIuMjEsMzcuNDctNjcuNjVoLTQ1LjQ5bC0xOC44MSwzOS41MXYtOTkuN2g3Ni42N3YyMzAuMDVaTTYxMS43Niw2MjAuNTdoLTY5Ljk5di0yMzAuMzdoMzkuMzd2MTk5LjMxaDMwLjYydjMxLjA2Wk02ODIuOTEsNjIwLjI4bC00LjUyLTU3LjQ1aDBsLTEwLjA2LTEyNy43Mi0xMC4wNiwxMjcuNzJoMGwtNC41Miw1Ny40NWgtMzkuMDhsMjMuNjItMjMwLjM3aDYwLjA3bDIzLjQ3LDIzMC4zN2gtMzguOTNaTTgxOC40Miw0NzguMjdjMCw2LjktMS4wNywxMi40Mi0zLjIxLDE2LjU1LTIuMTQsNC4xMy01LjA4LDcuMTItOC44Miw4Ljk3LTMuNzQsMS44NS04LjA0LDIuNzctMTIuOSwyLjc3LDQuNTcsMCw4Ljc1Ljk3LDEyLjU0LDIuOTIsMy43OSwxLjk1LDYuOCw1LjEzLDkuMDQsOS41NSwyLjIzLDQuNDIsMy4zNSwxMC4yMywzLjM1LDE3LjQydjUxLjAzYzAsMjEuNDgtMTAuMjYsMzIuNTEtMzAuNzcsMzMuMWgtNjIuNHYtMjI4LjQ4aDY0LjU5YzcuODcsMCwxNC42LDIuOCwyMC4xOSw4LjM4LDUuNTksNS41OSw4LjM4LDEyLjM3LDguMzgsMjAuMzR2NTcuNDVaIi8+CiAgICA8L2c+CiAgICA8cmVjdCBjbGFzcz0iY2xzLTMiIHg9IjY5LjMyIiB5PSIzNzYuMiIgd2lkdGg9Ijc4NSIgaGVpZ2h0PSIyNTguNCIvPgogICAgPGcgY2xhc3M9ImNscy04Ij4KICAgICAgPGc+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzcuMTQiIHgyPSI3ODYuNzIiIHkyPSIzNy4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjQzLjY2IiB4Mj0iNzg2LjcyIiB5Mj0iNDMuNjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI1MC4xNyIgeDI9Ijc4Ni43MiIgeTI9IjUwLjE3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNTYuNjkiIHgyPSI3ODYuNzIiIHkyPSI1Ni42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjYzLjIiIHgyPSI3ODYuNzIiIHkyPSI2My4yIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNjkuNzIiIHgyPSI3ODYuNzIiIHkyPSI2OS43MiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijc2LjIzIiB4Mj0iNzg2LjcyIiB5Mj0iNzYuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI4Mi43NSIgeDI9Ijc4Ni43MiIgeTI9IjgyLjc1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iODkuMjYiIHgyPSI3ODYuNzIiIHkyPSI4OS4yNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijk1Ljc4IiB4Mj0iNzg2LjcyIiB5Mj0iOTUuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDIuMjkiIHgyPSI3ODYuNzIiIHkyPSIxMDIuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDguODEiIHgyPSI3ODYuNzIiIHkyPSIxMDguODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMTUuMzIiIHgyPSI3ODYuNzIiIHkyPSIxMTUuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjEuODQiIHgyPSI3ODYuNzIiIHkyPSIxMjEuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjguMzUiIHgyPSI3ODYuNzIiIHkyPSIxMjguMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMzQuODciIHgyPSI3ODYuNzIiIHkyPSIxMzQuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDEuMzkiIHgyPSI3ODYuNzIiIHkyPSIxNDEuMzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDcuOSIgeDI9Ijc4Ni43MiIgeTI9IjE0Ny45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTU0LjQyIiB4Mj0iNzg2LjcyIiB5Mj0iMTU0LjQyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTYwLjkzIiB4Mj0iNzg2LjcyIiB5Mj0iMTYwLjkzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTY3LjQ1IiB4Mj0iNzg2LjcyIiB5Mj0iMTY3LjQ1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTczLjk2IiB4Mj0iNzg2LjcyIiB5Mj0iMTczLjk2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTgwLjQ4IiB4Mj0iNzg2LjcyIiB5Mj0iMTgwLjQ4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTg2Ljk5IiB4Mj0iNzg2LjcyIiB5Mj0iMTg2Ljk5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTkzLjUxIiB4Mj0iNzg2LjcyIiB5Mj0iMTkzLjUxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjAwLjAyIiB4Mj0iNzg2LjcyIiB5Mj0iMjAwLjAyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjA2LjU0IiB4Mj0iNzg2LjcyIiB5Mj0iMjA2LjU0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjEzLjA1IiB4Mj0iNzg2LjcyIiB5Mj0iMjEzLjA1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjE5LjU3IiB4Mj0iNzg2LjcyIiB5Mj0iMjE5LjU3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjI2LjA4IiB4Mj0iNzg2LjcyIiB5Mj0iMjI2LjA4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjMyLjYiIHgyPSI3ODYuNzIiIHkyPSIyMzIuNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjIzOS4xMSIgeDI9Ijc4Ni43MiIgeTI9IjIzOS4xMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI0NS42MyIgeDI9Ijc4Ni43MiIgeTI9IjI0NS42MyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1Mi4xNCIgeDI9Ijc4Ni43MiIgeTI9IjI1Mi4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1OC42NiIgeDI9Ijc4Ni43MiIgeTI9IjI1OC42NiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI2NS4xNyIgeDI9Ijc4Ni43MiIgeTI9IjI2NS4xNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3MS42OSIgeDI9Ijc4Ni43MiIgeTI9IjI3MS42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3OC4yIiB4Mj0iNzg2LjcyIiB5Mj0iMjc4LjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyODQuNzIiIHgyPSI3ODYuNzIiIHkyPSIyODQuNzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTEuMjMiIHgyPSI3ODYuNzIiIHkyPSIyOTEuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTcuNzUiIHgyPSI3ODYuNzIiIHkyPSIyOTcuNzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMDQuMjYiIHgyPSI3ODYuNzIiIHkyPSIzMDQuMjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTAuNzgiIHgyPSI3ODYuNzIiIHkyPSIzMTAuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTcuMjkiIHgyPSI3ODYuNzIiIHkyPSIzMTcuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMjMuODEiIHgyPSI3ODYuNzIiIHkyPSIzMjMuODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzAuMzIiIHgyPSI3ODYuNzIiIHkyPSIzMzAuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzYuODQiIHgyPSI3ODYuNzIiIHkyPSIzMzYuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDMuMzUiIHgyPSI3ODYuNzIiIHkyPSIzNDMuMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDkuODciIHgyPSI3ODYuNzIiIHkyPSIzNDkuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNTYuMzgiIHgyPSI3ODYuNzIiIHkyPSIzNTYuMzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNjIuOSIgeDI9Ijc4Ni43MiIgeTI9IjM2Mi45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzY5LjQxIiB4Mj0iNzg2LjcyIiB5Mj0iMzY5LjQxIi8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxnIGNsYXNzPSJjbHMtNyI+CiAgICAgIDxnPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjUwLjE5IiB4Mj0iNDc2LjA0IiB5Mj0iNTAuMTkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI1Ni43IiB4Mj0iNDc2LjA0IiB5Mj0iNTYuNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjYzLjIyIiB4Mj0iNDc2LjA0IiB5Mj0iNjMuMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI2OS43NCIgeDI9IjQ3Ni4wNCIgeTI9IjY5Ljc0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iNzYuMjUiIHgyPSI0NzYuMDQiIHkyPSI3Ni4yNSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjgyLjc3IiB4Mj0iNDc2LjA0IiB5Mj0iODIuNzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI4OS4yOCIgeDI9IjQ3Ni4wNCIgeTI9Ijg5LjI4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iOTUuOCIgeDI9IjQ3Ni4wNCIgeTI9Ijk1LjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDIuMzEiIHgyPSI0NzYuMDQiIHkyPSIxMDIuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDguODMiIHgyPSI0NzYuMDQiIHkyPSIxMDguODMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMTUuMzQiIHgyPSI0NzYuMDQiIHkyPSIxMTUuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjEuODYiIHgyPSI0NzYuMDQiIHkyPSIxMjEuODYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjguMzciIHgyPSI0NzYuMDQiIHkyPSIxMjguMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMzQuODkiIHgyPSI0NzYuMDQiIHkyPSIxMzQuODkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxNDEuNCIgeDI9IjQ3Ni4wNCIgeTI9IjE0MS40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTQ3LjkyIiB4Mj0iNDc2LjA0IiB5Mj0iMTQ3LjkyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTU0LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMTU0LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTYwLjk1IiB4Mj0iNDc2LjA0IiB5Mj0iMTYwLjk1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTY3LjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMTY3LjQ2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTczLjk4IiB4Mj0iNDc2LjA0IiB5Mj0iMTczLjk4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTgwLjQ5IiB4Mj0iNDc2LjA0IiB5Mj0iMTgwLjQ5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTg3LjAxIiB4Mj0iNDc2LjA0IiB5Mj0iMTg3LjAxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTkzLjUyIiB4Mj0iNDc2LjA0IiB5Mj0iMTkzLjUyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjAwLjA0IiB4Mj0iNDc2LjA0IiB5Mj0iMjAwLjA0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjA2LjU1IiB4Mj0iNDc2LjA0IiB5Mj0iMjA2LjU1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjEzLjA3IiB4Mj0iNDc2LjA0IiB5Mj0iMjEzLjA3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjE5LjU4IiB4Mj0iNDc2LjA0IiB5Mj0iMjE5LjU4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjI2LjEiIHgyPSI0NzYuMDQiIHkyPSIyMjYuMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzMi42MSIgeDI9IjQ3Ni4wNCIgeTI9IjIzMi42MSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzOS4xMyIgeDI9IjQ3Ni4wNCIgeTI9IjIzOS4xMyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI0NS42NCIgeDI9IjQ3Ni4wNCIgeTI9IjI0NS42NCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1Mi4xNiIgeDI9IjQ3Ni4wNCIgeTI9IjI1Mi4xNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1OC42NyIgeDI9IjQ3Ni4wNCIgeTI9IjI1OC42NyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI2NS4xOSIgeDI9IjQ3Ni4wNCIgeTI9IjI2NS4xOSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI3MS43IiB4Mj0iNDc2LjA0IiB5Mj0iMjcxLjciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyNzguMjIiIHgyPSI0NzYuMDQiIHkyPSIyNzguMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyODQuNzMiIHgyPSI0NzYuMDQiIHkyPSIyODQuNzMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTEuMjUiIHgyPSI0NzYuMDQiIHkyPSIyOTEuMjUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTcuNzYiIHgyPSI0NzYuMDQiIHkyPSIyOTcuNzYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMDQuMjgiIHgyPSI0NzYuMDQiIHkyPSIzMDQuMjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTAuNzkiIHgyPSI0NzYuMDQiIHkyPSIzMTAuNzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTcuMzEiIHgyPSI0NzYuMDQiIHkyPSIzMTcuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMjMuODIiIHgyPSI0NzYuMDQiIHkyPSIzMjMuODIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzAuMzQiIHgyPSI0NzYuMDQiIHkyPSIzMzAuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzYuODUiIHgyPSI0NzYuMDQiIHkyPSIzMzYuODUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDMuMzciIHgyPSI0NzYuMDQiIHkyPSIzNDMuMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDkuODgiIHgyPSI0NzYuMDQiIHkyPSIzNDkuODgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNTYuNCIgeDI9IjQ3Ni4wNCIgeTI9IjM1Ni40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzYyLjkxIiB4Mj0iNDc2LjA0IiB5Mj0iMzYyLjkxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzY5LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMzY5LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzc1Ljk0IiB4Mj0iNDc2LjA0IiB5Mj0iMzc1Ljk0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzgyLjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMzgyLjQ2Ii8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTMiIHBvaW50cz0iMTA2LjQyIDM0Ni44NiAyNDEuMzQgMzQ2LjM4IDI5MC42MyAxODUuNzEgMzMyLjc5IDI0Ni43NyA0NjIuNDEgNDYuMTUgNjI0LjUgMzc2LjYzIDY4My42MiAzMjEuODcgNjk3LjE5IDM0My42OCA4MTguNDIgMzQzLjY4Ii8+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik04MTguNDIsMzE5Ljk0aC0xMDkuMzlsLTYyLjUxLTEzMS4zMi0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41MSwxOC45LTkzLjUyLTExMi45LTU5LjEyLDExNC4zNi03LjI3LTE5LjM4cy01NS44NiwxMDcuMS02OC41OSwxMzIuMjljLTEzLjM4LTIyLjg1LTYwLjMxLTEwNS4xNS02MC4zMS0xMDUuMTVsLTc3LjA1LDk1Ljk0LTUuMzMtMTkuMzgtMjkuMDcsMzguNzdoLTY5LjEiLz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTUiIHBvaW50cz0iMTA2LjQyIDM2NSAxODcuMDQgMzY1IDIyMC44NSAyOTEuODMgMjMyIDMzNi45IDMwMC44MSAyMjYuNDEgMzI2Ljk3IDI5NC4yNSA0NDIuMyAxNDQuNTIgNTE2LjQ0IDI5OS41OCA1NTAuODUgMjM3LjA3IDYzNi4xMyAzNDQuNjUgNjkwLjQgMjQ1LjggNzIwLjkzIDMzMS41NyA4MTguNDIgMzMxLjU3Ii8+CiAgPC9nPgo8L3N2Zz4="
        h += f'''<div style="padding:20px 24px;background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);border-radius:16px;color:white;margin-bottom:14px;position:relative;overflow:hidden;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="flex:1;min-width:0;">
      <div style="font-size:11px;color:#94a3b8;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">Raport z badania wydolnościowego</div>
      <div style="font-size:28px;font-weight:800;margin-bottom:4px;letter-spacing:-0.5px;">{esc(name)}</div>
      <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">Wiek: {age} | {sex_pl} | {weight} kg | {height} cm | {test_date}</div>
      <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">Protok\u00f3\u0142: {esc(v('_protocol_description', protocol))} | Czas: {dur_str} | {"\u0042ie\u017cnia" if modality=="run" else "Cykl"}</div>
      <div style="display:flex;gap:14px;font-size:11px;">
        <a href="https://www.peaklab.com.pl" target="_blank" style="color:#93c5fd;text-decoration:none;display:inline-flex;align-items:center;gap:4px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>www.peaklab.com.pl</a>
        <a href="https://www.instagram.com/peak_lab_" target="_blank" style="color:#93c5fd;text-decoration:none;display:inline-flex;align-items:center;gap:4px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>@peak_lab_</a>
      </div>
    </div>
    <div style="flex-shrink:0;margin-left:16px;">
      <img src="data:image/svg+xml;base64,{_logo_pro}" style="height:90px;max-height:90px;opacity:0.9;" alt="PeakLab">
    </div>
  </div>
</div>'''

        # ═══════════════════════════════════════════════════════════════
        # I. DIAGNOZA I REKOMENDACJA
        # ═══════════════════════════════════════════════════════════════
        try: pctile_val = float(vo2_pctile) if vo2_pctile else 50
        except: pctile_val = 50

        # ── UNIFIED PROFILE SCORING (computed in build_canon_table) ──
        _profile_pro = ct.get('_profile') or compute_profile_scores(ct)  # fallback
        _overall_score = _profile_pro['overall']
        _overall_grade = _profile_pro['grade_letter']
        _hrr_gauge_score = _profile_pro['gauge_scores']['recovery']
        pctile_val = _profile_pro['pctile_val']

        # HRR label for gauge subtitle
        try:
            _hrr_val = float(hrr1) if hrr1 else 0
            _hrr_lbl = f'HRR {_hrr_val:.0f}' if _hrr_val > 0 else 'b/d'
        except: _hrr_lbl = 'b/d'

        _gauge_vo2_sub = _profile_pro.get('sport_class', '')
        _gauge_val_sub = str(val_grade)
        _gauge_hrr_sub = _hrr_lbl
        _gauge_overall_sub = _overall_grade
        h += f'''<div class="section"><div class="card">
  {section_title('Diagnoza i Plan', 'I')}
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div style="display:flex;gap:6px;flex-shrink:0;flex-wrap:wrap;">
      {gauge_svg(min(100, _profile_pro['categories']['vo2max']['score']), 'VO\u2082max', subtitle=_gauge_vo2_sub)}
      {gauge_svg(min(100, max(0, val_score)), 'Ważność', subtitle=_gauge_val_sub)}
      {gauge_svg(min(100, max(0, _hrr_gauge_score)), 'Recovery', subtitle=_gauge_hrr_sub)}
      {gauge_svg(min(100, max(0, _overall_score)), 'Profil', subtitle=_gauge_overall_sub)}</div>
    <div style="flex:1;min-width:260px;">'''

        # VO2 gauge bar
        h += f'''<div style="font-size:11px;font-weight:600;color:#475569;margin-bottom:4px;">VO₂{"max" if str(vo2_det)=="VO2max" else "peak"} — {"Norma sportowa" if _is_sport else "Norma populacyjna"}</div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <div style="font-size:36px;font-weight:700;color:#0f172a;">{_n(vo2_rel)}</div>
        <div style="font-size:14px;color:#64748b;">ml/kg/min</div>
        {badge(vo2_class, '#16a34a' if pctile_val >= 70 else '#d97706')}
      </div>
      <div style="position:relative;height:14px;border-radius:7px;overflow:hidden;display:flex;margin-bottom:2px;">
        <div style="flex:20;background:#dc2626;"></div><div style="flex:20;background:#ea580c;"></div>
        <div style="flex:15;background:#eab308;"></div><div style="flex:15;background:#22c55e;"></div>
        <div style="flex:15;background:#3b82f6;"></div><div style="flex:15;background:#7c3aed;"></div>
        <div style="position:absolute;left:{min(97,max(3,_overall_score))}%;top:-2px;font-size:16px;transform:translateX(-50%);">▼</div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:#94a3b8;margin-bottom:6px;">
        <span>Poor</span><span>Fair</span><span>Good</span><span>Excellent</span><span>Superior</span>
      </div>
      <div style="font-size:11px;color:#475569;">
        Percentyl: ~{_n(vo2_pctile,'.0f','?')} | {_n(vo2_pct_pred,'.0f','?')}% predicted | Sportowy: <b>{esc(vo2_class_sport)}</b>
      </div>'''

        # Maximality criteria
        h += '<div class="sub-header">Kryteria wysiłku maksymalnego</div>'
        try:
            rer_ok = rer_peak and str(rer_peak) not in ('-','[BRAK]','None','') and float(rer_peak) >= 1.10
        except (ValueError, TypeError):
            rer_ok = False
        try:
            hr_ok = hr_pct_pred and str(hr_pct_pred) not in ('-','[BRAK]','None','') and float(hr_pct_pred) >= 85
        except (ValueError, TypeError):
            hr_ok = False
        h += f'<div style="font-size:11px;color:#334155;line-height:1.8;">'
        h += f'{"✅" if rer_ok else "❌"} RER {_n(rer_peak,".2f")} {"≥" if rer_ok else "<"}1.10 &nbsp; '
        h += f'{"✅" if hr_ok else "❌"} HR {_n(hr_pct_pred,".0f")}% pred ({_n(hr_peak,".0f")}/{_n(hr_pred,".0f")}) &nbsp; '
        h += f'{"✅" if plateau else "❌"} Plateau VO₂ {"wykryte" if plateau else "brak"}'
        h += '</div>'

        # Auto observations
        h += '<div class="sub-header">Obserwacje kluczowe</div><div style="font-size:12px;line-height:1.8;">'

        # Generate observations from data — with velocity context
        try:
            _vt1p = float(vt1_pct) if vt1_pct else 0
            _vt2p = float(vt2_pct) if vt2_pct else 0
            _gap = float(thr_gap) if thr_gap else 0
            _vo2v = float(vo2_rel) if vo2_rel else 0
            _slp = float(ve_vco2_slope) if ve_vco2_slope else 0
            _pc = ct.get('_performance_context', {})
            _v_vt1 = _pc.get('v_vt1_kmh')
            _v_vt2 = _pc.get('v_vt2_kmh')
            _mas_pct_vt1 = _pc.get('vt1_pct_vref')
            _mas_pct_vt2 = _pc.get('vt2_pct_vref')
            _mas_src = _pc.get('v_ref_source', '')
            _lvl = _pc.get('level_by_speed', '')
            _lvl_pct = _pc.get('level_by_pct_vo2', '')
            _lvl_match = _pc.get('levels_match', True)

            # ── VT1 with speed context (uses unified profile flags) ──
            if _vt1p > 0:
                _vt1_t = f'VT1 przy {_vt1p:.0f}% VO\u2082max'
                if _v_vt1:
                    _vt1_t += f' / {_v_vt1:.1f} km/h'
                if _mas_pct_vt1 and 'MAS_external' in _mas_src:
                    _vt1_t += f' ({_mas_pct_vt1:.0f}% MAS)'

                # Check ceiling trap from unified scoring
                if _profile_pro['flags']['vt1_ceiling_trap']:
                    _sport_ctx_pro = _profile_pro.get('sport_class', '')
                    _sport_lbl_pro = {'ELITE':'elitarny','SUB_ELITE':'subelitarny','COMPETITIVE':'zawodniczy','TRAINED':'wytrenowany','RECREATIONAL':'rekreacyjny','UNTRAINED':'początkujący'}.get(_sport_ctx_pro, f'~{_profile_pro["pctile_val"]:.0f} percentyl')
                    _vt1_t += f' \u2014 wysoki %VO\u2082max ale niski pu\u0142ap absolutny (poziom {_sport_lbl_pro}). Baza "pozornie dobra".'
                    h += obs_bullet(_vt1_t, 'warning')
                elif _v_vt1 and _mas_pct_vt1:
                    if _vt1p >= 65 and _mas_pct_vt1 >= 55:
                        _vt1_t += ' \u2014 <b>bardzo dobra baza tlenowa</b>.'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 65 and _mas_pct_vt1 < 45:
                        _vt1_t += ' \u2014 wysoki %VO\u2082max ale niska pr\u0119dko\u015b\u0107 absolutna.'
                        h += obs_bullet(_vt1_t, 'warning')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa.'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej (wi\u0119cej Z2).'
                        h += obs_bullet(_vt1_t, 'info')
                elif _v_vt1:
                    if _vt1p >= 65:
                        _vt1_t += ' \u2014 <b>dobra baza tlenowa</b>.'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa.'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej.'
                        h += obs_bullet(_vt1_t, 'info')
                else:
                    if _vt1p >= 65:
                        _vt1_t += ' \u2014 dobra baza tlenowa (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej.'
                        h += obs_bullet(_vt1_t, 'info')

            # ── VT2 with speed context ──
            if _vt2p > 0:
                _vt2_t = f'VT2 przy {_vt2p:.0f}% VO\u2082max'
                if _v_vt2:
                    _vt2_t += f' / {_v_vt2:.1f} km/h'
                if _mas_pct_vt2 and 'MAS_external' in _mas_src:
                    _vt2_t += f' ({_mas_pct_vt2:.0f}% MAS)'

                if _lvl:
                    _vt2_t += f' \u2014 poziom <b>{_lvl}</b>'
                    if not _lvl_match and _lvl_pct:
                        _vt2_t += f' (uwaga: %VO\u2082max sugeruje {_lvl_pct})'

                    if _vt2p >= 88 and _lvl in ('Sedentary', 'Recreational'):
                        _vt2_t += '. Wysoki pr\u00f3g wzgl\u0119dny ale niska pr\u0119dko\u015b\u0107 \u2192 niski VO\u2082max.'
                        h += obs_bullet(_vt2_t, 'warning')
                    elif _vt2p >= 88 and _lvl == 'Trained':
                        _vt2_t += '. Dalszy post\u0119p przez interwa\u0142y VO\u2082max.'
                        h += obs_bullet(_vt2_t, 'info')
                    elif _vt2p >= 88:
                        _vt2_t += '. Doskona\u0142y pr\u00f3g potwierdzony pr\u0119dko\u015bci\u0105.'
                        h += obs_bullet(_vt2_t, 'positive')
                    elif _lvl in ('Well-trained', 'Elite'):
                        h += obs_bullet(_vt2_t + '.', 'positive')
                    else:
                        _vt2_t += '. Potencja\u0142 do rozwoju przez trening tempo/SST.'
                        h += obs_bullet(_vt2_t, 'info')
                else:
                    if _vt2p >= 85:
                        _vt2_t += f' \u2014 <b>{"bardzo wysoki" if _vt2p>=90 else "wysoki"} pr\u00f3g</b> (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt2_t, 'positive')
                    else:
                        _vt2_t += ' \u2014 przestrze\u0144 do poprawy treningu progowego.'
                        h += obs_bullet(_vt2_t, 'info')

            # ── Gap ──
            if _gap >= 25:
                h += obs_bullet(f'Gap VT2-VT1: {_gap:.0f} bpm \u2014 <b>du\u017ca przestrze\u0144</b> do treningu tempo/threshold.', 'positive')
            elif _gap >= 15:
                h += obs_bullet(f'Gap VT2-VT1: {_gap:.0f} bpm \u2014 umiarkowana przestrze\u0144 tempo.', 'info')

            # ── VE/VCO2 slope ──
            if _slp < 25 and _slp > 0:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 <b>doskona\u0142a efektywno\u015b\u0107 wentylacyjna</b>.', 'positive')
            elif _slp < 30:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 dobra efektywno\u015b\u0107 wentylacyjna.', 'positive')
            elif _slp < 34:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 umiarkowana efektywno\u015b\u0107 wentylacyjna.', 'warning')

            # ── Breathing flags ──
            if _is_sport and e07_flags:
                flags_str = ', '.join(e07_flags) if isinstance(e07_flags, list) else str(e07_flags)
                h += obs_bullet(f'Flagi oddechowe [{flags_str}] \u2014 przy VO\u2082max {_vo2v:.0f} ml/kg/min reinterpretowane jako <b>adaptacje sportowe</b>.', 'sport')

            # ── FATmax ──
            if fatmax:
                h += obs_bullet(f'FATmax {float(fatmax):.2f} g/min przy HR {_n(fatmax_hr,".0f")} ({_n(fatmax_pct_vo2,".0f")}% VO\u2082) \u2014 strefa HR {_n(fatmax_zone_lo,".0f")}-{_n(fatmax_zone_hi,".0f")} bpm.', 'info')
        except: pass

        h += '</div>'

        # Training recommendation — uses profile scoring + E20 sport-specific session
        h += '<div class="sub-header">Rekomendacja treningowa</div>'
        try:
            _profile_pro = ct.get('_profile') or {}
            _limiter_pro = _profile_pro.get('limiter', {})
            _limiter_key_pro = _profile_pro.get('limiter_key', '')
            _super_pro = _profile_pro.get('superpower', {})

            # E20 GameChanger — same logic as LITE
            _lim_key_map_pro = {
                'vo2max': 'HIGH_THRESHOLDS_LOW_CEILING',
                'vt2': 'HIGH_BASE_LOW_THRESHOLD',
                'vt1': 'LOW_BASE',
                'economy': 'ECONOMY_LIMITER',
                'ventilation': 'VENTILATORY_LIMITER',
                'cardiac': 'CARDIAC_LIMITER',
                'recovery': 'RECOVERY_LIMITER',
                'substrate': 'SUBSTRATE_LIMITER',
                'breathing': 'VENTILATORY_LIMITER',
            }
            _gc_pro_session = ''
            try:
                from e20_training_decision import PhysioSnapshot as _PS, TrainingProfile as _TP, _scale_session as _ss, SPORT_CLASS_RANK as _SCR
                _e20r = {}
                for _eid in ['E00','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','E16','E17','E19']:
                    _e20r[_eid] = ct.get('_' + _eid.lower() + '_raw', {})
                _e20r['_performance_context'] = ct.get('_performance_context', {})
                _e20s = _PS.from_results(_e20r, None)
                _e20s.zones = ct.get('_e16_zones', {})
                _e20m = ct.get('_prot_modality', 'run')
                _sc_r = _SCR.get(_e20s.sport_class, 1)
                _lt = _lim_key_map_pro.get(_limiter_key_pro, 'HIGH_BASE_LOW_THRESHOLD')
                _gc_pro_session = _ss('KEY_1', _e20s, _sc_r, _lt, modality=_e20m)
            except Exception:
                _gc_pro_session = _limiter_pro.get('tip', '')

            # Display: Priority ranking (compact) + GameChanger
            _prio_pro = _profile_pro.get('priority_ranking', [])
            if _prio_pro:
                _top3 = _prio_pro[:3]
                _atut = _prio_pro[-1] if len(_prio_pro) > 3 else None
                h += '<div style="padding:10px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;margin-bottom:6px;font-size:11px;">'
                h += '<b>Ranking priorytetów:</b><br>'
                for i, p in enumerate(_top3):
                    _pc = '#dc2626' if p['score']<55 else ('#d97706' if p['score']<70 else ('#0d9488' if p['score']<85 else '#059669'))
                    h += f'<span style="color:{_pc};font-weight:700;">#{i+1}</span> {p["icon"]} {p["label"]} <span style="color:{_pc};font-weight:700;">{p["score"]:.0f}</span> · '
                if _atut:
                    h += f'<br><span style="color:#059669;font-weight:700;">ATUT:</span> {_atut["icon"]} {_atut["label"]} <span style="color:#059669;font-weight:700;">{_atut["score"]:.0f}</span>'
                h += '</div>'
            if _gc_pro_session:
                # Manual override from config takes priority
                _gc_display = ct.get('_gc_manual') or _gc_pro_session
                _gc_is_manual = bool(ct.get('_gc_manual'))
                h += f'<div style="padding:10px;background:linear-gradient(135deg,#fefce8,#fef9c3);border-radius:8px;border-left:3px solid #eab308;font-size:12px;">'
                h += f'<b>🎯 GAME CHANGER — trening tygodnia</b>{"  <span style=\"font-size:10px;color:#a16207;\">(✏️ ręcznie)</span>" if _gc_is_manual else ""}<br>'
                h += f'<span style="color:#334155;">{_gc_display}</span></div>'
        except: pass

        h += '</div></div></div>'

        # ═══════════════════════════════════════════════════════════════
        # II. PROGI + STREFY
        # ═══════════════════════════════════════════════════════════════

        # VT cards
        def vt_card(label, pct_vo2, hr, time_s, speed, rer_v, vo2_abs_v, hr_pct_mx, conf, pct_mas=None, col='#16a34a'):
            pct_val = _n(pct_vo2, '.0f', '?') if pct_vo2 else '?'
            try:
                bar_w = min(100, max(0, float(pct_vo2))) if pct_vo2 and str(pct_vo2) not in ('-','[BRAK]','None','') else 0
            except (ValueError, TypeError):
                bar_w = 0
            try:
                _conf_f = float(conf) if conf and str(conf) not in ('-','[BRAK]','None','') else 0
            except (ValueError, TypeError):
                _conf_f = 0
            conf_lbl = 'Potwierdzone' if _conf_f>=0.8 else ('Umiarkowane' if _conf_f>=0.5 else 'Niepewne')
            conf_col = '#16a34a' if _conf_f>=0.8 else ('#d97706' if _conf_f>=0.5 else '#dc2626')
            is_vt2 = 'VT2' in label
            return f'''<div style="flex:1;min-width:200px;padding:12px;border:1px solid #e2e8f0;border-radius:8px;background:white;">
  <div style="font-size:12px;font-weight:700;color:#475569;">{label}</div>
  <div style="font-size:28px;font-weight:700;color:#0f172a;">{pct_val}% <span style="font-size:13px;font-weight:400;color:#64748b;">VO₂max</span></div>
  <div style="height:6px;background:#e5e7eb;border-radius:3px;margin:6px 0;"><div style="height:6px;background:{'#3b82f6' if is_vt2 else '#22c55e'};border-radius:3px;width:{bar_w}%;"></div></div>
  <div style="font-size:11px;color:#64748b;">HR {_n(hr,'.0f','-')} bpm ({_n(hr_pct_mx,'.0f','?')}% HRmax) | {_fmt_dur(time_s)}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:2px;">VO₂ {_n(vo2_abs_v,'.0f','-')} ml/min | RER {_n(rer_v,'.2f','-')}{f' | {_n(speed,".1f")} km/h' + (f' ({pct_mas:.0f}% MAS)' if pct_mas else '') if speed else ''}</div>
  <div style="margin-top:4px;">{badge(conf_lbl, conf_col)}</div>
</div>'''


        # ── Performance Context for VT cards ────────────────────
        _pc = ct.get('_performance_context', {})
        _vt1_pct_mas = _pc.get('vt1_pct_vref') if _pc.get('executed') else None
        _vt2_pct_mas = _pc.get('vt2_pct_vref') if _pc.get('executed') else None

        h += f'''<div class="section"><div class="card">
  {section_title('Progi i Strefy treningowe', 'II')}
  <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
    {vt_card('VT1 — Próg tlenowy', vt1_pct, vt1_hr, vt1_time, vt1_speed, vt1_rer_val, vt1_vo2_abs, vt1_hr_pct_max, vt1_conf, _vt1_pct_mas)}
    {vt_card('VT2 — Próg beztlenowy', vt2_pct, vt2_hr, vt2_time, vt2_speed, vt2_rer_val, vt2_vo2_abs, vt2_hr_pct_max, vt2_conf, _vt2_pct_mas)}
  </div>
  '''

        # ── Compact speed classification ─────────────────────
        if _pc.get('executed') and _pc.get('v_vt2_kmh'):
            _lvl_s = _pc.get('level_by_speed', '')
            _lvl_p = _pc.get('level_by_pct_vo2', '')
            _mas_ext = _pc.get('mas_external_m_s')
            _mas_kmh_v = _pc.get('mas_external_kmh')
            _mss_v = _pc.get('mss_external_m_s')
            _asr_v = _pc.get('asr_kmh')
            _pc_flags = _pc.get('flags', [])

            _ctx_parts = []
            if _lvl_s:
                _ctx_parts.append(f'Klasyfikacja: <b>{_lvl_s}</b>')
            if _lvl_s != _lvl_p and _lvl_p:
                _ctx_parts.append(f'<span style="color:#d97706;">(%VO\u2082max \u2192 {_lvl_p})</span>')
            if _mas_ext:
                _ctx_parts.append(f'MAS: <b>{_mas_ext:.2f} m/s</b> ({_mas_kmh_v:.1f} km/h)')
                if _mss_v:
                    _ctx_parts.append(f'MSS: {_mss_v:.1f} m/s')
                    if _asr_v:
                        _ctx_parts.append(f'ASR: {_asr_v:.1f} km/h')

            if _ctx_parts:
                h += '<div style="margin-top:8px;padding:8px 12px;background:#f0f9ff;border-radius:6px;font-size:11px;color:#475569;border-left:3px solid #3b82f6;">' + '  |  '.join(_ctx_parts) + '</div>'

            for _fl in _pc_flags:
                h += f'<div style="margin-top:3px;padding:3px 12px;font-size:10px;color:#d97706;">\u26a0\ufe0f {_fl}</div>'

                # Zones table
        zone_colors = ['#22c55e','#84cc16','#eab308','#f97316','#ef4444']
        zone_names_pl = ['Z1 Regeneracja','Z2 Baza tlenowa','Z3 Tempo','Z4 Próg','Z5 VO₂max']

        h += '<table class="ztable"><tr><th>Strefa</th><th>HR (bpm)</th><th>%HRmax</th><th>Tempo</th><th>RPE</th><th>Opis</th></tr>'
        for i, (zname, zcol) in enumerate(zip(zone_names_pl, zone_colors)):
            zkey = f'z{i+1}'
            zd = zones_data.get(zkey, {})
            hr_lo = zd.get('hr_low','?'); hr_hi = zd.get('hr_high','?')
            phr_lo = zd.get('pct_hrmax_low',''); phr_hi = zd.get('pct_hrmax_high','')
            spd_lo = zd.get('speed_low',''); spd_hi = zd.get('speed_high','')
            rpe = zd.get('rpe','')
            desc = zd.get('description_pl','')
            h += f'<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{zcol};margin-right:4px;vertical-align:middle;"></span>{zname}</td>'
            h += f'<td style="font-weight:600;">{hr_lo}-{hr_hi}</td>'
            h += f'<td>{_n(phr_lo,".0f","")}-{_n(phr_hi,".0f","")}%</td>'
            h += f'<td>{_n(spd_lo,".1f","")}-{_n(spd_hi,".1f","")} km/h</td>'
            h += f'<td>{rpe}</td><td style="font-size:10px;color:#64748b;">{desc}</td></tr>'
        h += '</table>'

        # 3-zone model
        if three_zone:
            h += '<div style="margin-top:6px;font-size:10px;color:#64748b;">Model 3-strefowy: '
            for zk, zv in three_zone.items():
                h += f'{zv.get("label","")} ({zv.get("hr_low","")}-{zv.get("hr_high","")}) | '
            h += '</div>'

        extras = []
        if aer_reserve: extras.append(f'Rezerwa tlenowa: {_n(aer_reserve,".0f")}%')
        if thr_gap: extras.append(f'Gap VT2-VT1: {_n(thr_gap,".0f")} bpm ({_n(thr_gap_pct,".0f")}% HRmax)')
        if extras:
            h += f'<div style="margin-top:6px;font-size:11px;color:#475569;font-weight:600;">{"  |  ".join(extras)}</div>'

        h += '</div></div>'

        # ═══════════════════════════════════════════════════════════════
        # III. PROFIL FIZJOLOGICZNY — 4 PANELE
        # ═══════════════════════════════════════════════════════════════
        h += f'<div class="section">{section_title("Profil fizjologiczny", "III")}'

        # --- PANEL A: WYDOLNOŚĆ ---
        pa = ''
        pa += row_item('VO₂max', f'{_n(vo2_rel)} ml/kg/min', vo2_class, f'{_n(vo2_abs,".2f")} L/min | {_n(vo2_pct_pred,".0f")}% pred. | ~{_n(vo2_pctile,".0f")} percentyl')
        pa += '<div class="sub-header">OUES</div>'
        pa += row_item('OUES 100%', f'{_n(oues100,".0f")}', f'{_n(oues_pct,".0f")}% pred.', f'OUES/kg: {_n(oues_kg,".1f")}')
        pa += row_item('OUES 90/75%', f'{_n(oues90,".0f")} / {_n(oues75,".0f")}', f'stab. {_n(oues_stab,".3f")}' if oues_stab else '', '')
        pa += row_item('OUES @VT1/@VT2', f'{_n(oues_vt1,".0f")} / {_n(oues_vt2,".0f")}', '', '')
        pa += '<div class="sub-header">VE/VCO₂</div>'
        pa += row_item('Slope (→VT2)', _n(ve_vco2_slope), vc_class, f'{vc_desc} | {_n(slope_pct_pred,".0f")}% pred.')
        pa += row_item('Slope full/→VT1', f'{_n(ve_vco2_full)} / {_n(ve_vco2_to_vt1)}', '', '')
        pa += row_item('Nadir / @VT1', f'{_n(ve_vco2_nadir)} / {_n(ve_vco2_at_vt1)}', '', '')
        pa += row_item('RER peak', _n(rer_peak,'.3f'), rer_class, f'{"Plateau VO₂ ✓" if plateau else ""} | {rer_desc}')
        pa += row_item('VE peak', f'{_n(ve_peak,".1f")} L/min', '', '')

        # --- PANEL B: SERCE & RECOVERY ---
        pb = ''
        pb += row_item('O₂-pulse peak', f'{_n(o2p_peak)} ml/beat', o2p_traj, f'{_n(o2p_pct,".0f")}% pred. | Est. SV ~{_n(sv_est,".0f")} ml')
        pb += row_item('O₂-pulse @VT1/VT2', f'{_n(o2p_vt1)} / {_n(o2p_vt2)} ml/beat', '', o2p_desc)
        pb += row_item('CI', _n(ci,'.2f'), ci_class, 'Kompetencja chronotropowa')
        pb += row_item('HR-VO₂ slope', f'{_n(hr_vo2_slope)} bpm/L', 'SPORT' if slope_class in ('VERY_LOW','LOW_HIGH_SV') and _is_sport else slope_class, f'R²={_n(e13.get("hr_vo2_r2"),".3f")}')
        pb += row_item('Breakpoint HR-VO₂', f'{_fmt_dur(bp_time_e13)} ({_n(bp_pct_e13,".0f")}%)', bp_vs_vt2_e13, f'CV coupling: {"łagodna anomalia" if cv_coupling=="MILD_ABNORMALITY" else ("doskonały" if cv_coupling=="EXCELLENT" else cv_coupling)}')
        pb += row_item('HRmax', f'{_n(hr_peak,".0f")} bpm', '', f'{_n(hr_pct_pred,".0f")}% pred. ({_n(hr_pred,".0f")})')

        pb += '<div class="sub-header">RECOVERY</div>'

        # HRR 1min — with sport-aware description
        _hrr1_comment = hrr1_desc_e15 or ''
        if hrr1 and float(hrr1) > 0:
            _hv = float(hrr1)
            if _hv >= 40: _hrr1_comment = f'Bardzo dobra (typowa dla sportowców)'
            elif _hv >= 25: _hrr1_comment = f'Dobra — prawidłowa odpowiedź autonomiczna'
            elif _hv >= 12: _hrr1_comment = f'Poniżej normy sportowej — rozważ periodyzację'
            else: _hrr1_comment = f'Nieprawidłowa (<12 bpm) — wymaga oceny klinicznej'
        pb += row_item('HRR 1min', f'{_n(hrr1,".0f")} bpm', (str(hrr1_class or hrr1_class_e15)).upper(), _hrr1_comment)

        # HRR 3min — add classification
        _hrr3_class = str(hrr3_class).upper() if hrr3_class and str(hrr3_class) != '?' else ''
        _hrr3_comment = ''
        if hrr3 and float(hrr3) > 0:
            _h3 = float(hrr3)
            if _h3 >= 50: _hrr3_class = _hrr3_class or 'GOOD'; _hrr3_comment = 'Prawidłowa 3-min recovery'
            elif _h3 >= 30: _hrr3_class = _hrr3_class or 'NORMAL'
            else: _hrr3_class = _hrr3_class or 'SLOW'; _hrr3_comment = 'Spowolniona recovery autonomiczna'
        pb += row_item('HRR 3min', f'{_n(hrr3,".0f")} bpm', _hrr3_class, _hrr3_comment)

        # T½ VO₂ — improved display with conditional tau
        _thalf_desc = ''
        _thalf_class = rec_class if rec_class and rec_class != '?' else ''
        if t_half_vo2:
            _tv = float(t_half_vo2)
            if _tv < 60: _thalf_class = _thalf_class or 'EXCELLENT'
            elif _tv < 90: _thalf_class = _thalf_class or 'GOOD'
            elif _tv < 120: _thalf_class = _thalf_class or 'NORMAL'
            elif _tv < 180: _thalf_class = _thalf_class or 'SLOW'
            else: _thalf_class = _thalf_class or 'VERY_SLOW'
        if tau_vo2 and tau_r2 and float(tau_r2) >= 0.80:
            _thalf_desc = f'τ={_n(tau_vo2,".0f")}s (R²={_n(tau_r2,".2f")})'
        elif tau_vo2 and tau_r2:
            _thalf_desc = f'τ={_n(tau_vo2,".0f")}s (R²={_n(tau_r2,".2f")} — niski fit)'
        elif rec_desc:
            _thalf_desc = rec_desc
        pb += row_item('T½ VO₂', f'{_n(t_half_vo2,".0f")}s', _thalf_class, _thalf_desc)

        # T½ VCO₂ / VE — add ratio interpretation
        _tvco2_desc = ''
        if t_half_vco2 and t_half_vo2 and float(t_half_vo2) > 0:
            _ratio = float(t_half_vco2) / float(t_half_vo2)
            if _ratio > 1.5: _tvco2_desc = f'VCO₂/VO₂ ratio: {_ratio:.2f} — wysoka akumulacja laktatu'
            elif _ratio > 1.1: _tvco2_desc = f'VCO₂/VO₂ ratio: {_ratio:.2f} — norma'
            else: _tvco2_desc = f'VCO₂/VO₂ ratio: {_ratio:.2f}'
        pb += row_item('T½ VCO₂ / VE', f'{_n(t_half_vco2,".0f")}s / {_n(t_half_ve,".0f")}s', '', _tvco2_desc)

        # Recovery 60s/120s — add norms context
        _rec_pct_class = ''
        _rec_pct_desc = ''
        if pct_rec_60 is not None:
            _p60 = float(pct_rec_60)
            if _p60 >= 40: _rec_pct_class = 'GOOD'
            elif _p60 >= 25: _rec_pct_class = 'NORMAL'
            else: _rec_pct_class = 'SLOW'
        if dvo2_60:
            _rec_pct_desc = f'ΔVO₂ 60s: {_n(dvo2_60,".1f")} ml/kg/min'
        pb += row_item('Recovery 60s/120s', f'{_n(pct_rec_60,".0f")}% / {_n(pct_rec_120,".0f")}%', _rec_pct_class, _rec_pct_desc)

        # --- PANEL C: ODDYCHANIE ---
        pc = ''
        bp_assess = 'SPORT' if bp_pattern in ('DYSFUNCTIONAL_BPD',) and _is_sport else bp_pattern
        pc += row_item('Wzorzec oddech.', bp_strategy, bp_assess, f'{_n(ve_from_bf,".0f")}% VE z BF | VT peak {_n(vt_peak,".1f")} L')
        pc += row_item('BF rest/VT1/VT2/peak', f'{_n(e07.get("bf_rest"),".0f")} / {_n(bf_vt1,".0f")} / {_n(bf_vt2,".0f")} / {_n(bf_peak,".0f")} /min', '', f'RSBI {_n(rsbi,".1f")}' if rsbi else '')
        pc += row_item('VT rest/VT1/VT2/peak', f'{_n(e07.get("vt_rest_L"),".2f")} / {_n(vt_vt1,".2f")} / {_n(vt_vt2,".2f")} / {_n(vt_peak,".2f")} L', '', '')
        if vt_plat_t:
            pc += row_item('Plateau VT', f't={_fmt_dur(vt_plat_t)} ({_n(vt_plat_pct,".0f")}%)', e07.get('vt_plateau_vs_vt2',''), '')

        vdvt_assess = 'SPORT' if vdvt_pattern in ('PARADOXICAL_RISE',) and _is_sport else vdvt_pattern
        pc += row_item('VD/VT', f'{_n(vdvt_rest,".2f")}→{_n(vdvt_peak,".2f")}', vdvt_assess, '')
        pc += row_item('PetCO₂', f'rest {_n(petco2_rest,".0f")} → VT1 {_n(petco2_vt1,".0f")} → peak {_n(petco2_peak,".0f")}', petco2_pattern, '')
        if peto2_nadir:
            pc += row_item('PetO₂ nadir', f'{_n(peto2_nadir,".0f")} mmHg', '', '')

        br_pct = g('BR_pct')
        pc += row_item('BR', f'{_n(br_pct,".0f")}%' if br_pct else '-', '', 'Brak MVV' if not br_pct else '')

        if e07_flags:
            flags_str = ', '.join(e07_flags) if isinstance(e07_flags, list) else str(e07_flags)
            pc += f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Flagi: {flags_str}</div>'

        # --- PANEL D: PALIWO ---
        pd_html = ''
        pd_html += '<div class="sub-header">FATmax</div>'
        pd_html += row_item('MFO', f'{_n(fatmax,".2f")} g/min ({_n(e10.get("mfo_mgkg_min"),".1f")} mg/kg/min)' if fatmax else '-', '', f'HR {_n(fatmax_hr,".0f")} ({_n(fatmax_pct_hr,".0f")}% HRmax) | {_n(fatmax_pct_vo2,".0f")}% VO₂peak' if fatmax_hr else '')
        pd_html += row_item('Strefa FATmax', f'HR {_n(fatmax_zone_lo,".0f")}-{_n(fatmax_zone_hi,".0f")} bpm' if fatmax_zone_lo else '-', '', '')
        pd_html += '<div class="sub-header">Crossover & Substrat</div>'
        pd_html += row_item('Crossover', f'HR {_n(cop_hr,".0f")}' if cop_hr else '-', '', f'{_n(cop_pct_vo2,".0f")}% VO₂ | RER {_n(cop_rer,".3f")}' if cop_hr else '')
        pd_html += row_item('Substrat @VT1', f'FAT {_n(fat_vt1_pct,".0f")}% ({_n(fat_vt1_g,".2f")} g/min) / CHO {_n(cho_vt1_pct,".0f")}% ({_n(cho_vt1_g,".2f")} g/min)' if fat_vt1_pct else '-', '', '')
        pd_html += row_item('Substrat @VT2', f'FAT {_n(fat_vt2_pct,".0f")}% / CHO {_n(cho_vt2_pct,".0f")}%' if fat_vt2_pct else '-', '', '')
        if total_kcal:
            pd_html += row_item('Kalorie total', f'{_n(total_kcal,".0f")} kcal', '', f'FAT {_n(total_fat,".0f")} + CHO {_n(total_cho,".0f")} kcal')

        # Zone substrate table
        if zone_sub:
            pd_html += '<div class="sub-header">Substrat per strefa</div>'
            pd_html += '<table class="ztable"><tr><th>Strefa</th><th>FAT g/h</th><th>CHO g/h</th><th>FAT%</th><th>kcal/h</th></tr>'
            for zk in ['z2','z3','z4','z5']:
                zs = zone_sub.get(zk,{})
                if zs:
                    note = f' *' if zs.get('note') else ''
                    pd_html += f'<tr><td>{zk.upper()}</td><td>{_n(zs.get("fat_gh"),".0f","-")}</td><td>{_n(zs.get("cho_gh"),".0f","-")}</td><td>{_n(zs.get("fat_pct"),".0f","-")}%</td><td>{_n(zs.get("kcal_h"),".0f","-")}{note}</td></tr>'
            pd_html += '</table>'

        # Lactate
        pd_html += '<div class="sub-header">Laktat</div>'
        if la_peak:
            pd_html += row_item('La peak', f'{_n(la_peak)} mmol/L', '', f'Baseline {_n(la_base)}')
            pd_html += row_item('LT1', f'HR {_n(lt1_hr,".0f")} | {_fmt_dur(lt1_time)}' if lt1_hr else '-', '', lt1_method)
            pd_html += row_item('LT2', f'HR {_n(lt2_hr,".0f")} | {_fmt_dur(lt2_time)}' if lt2_hr else '-', '', lt2_method)
        else:
            pd_html += row_item('Laktat', 'Brak danych', 'NO_DATA', '')

        h += f'''<div class="flex-row">
  {panel_card('🫁 Wydolność i Efektywność', '', pa)}
  {panel_card('❤️ Serce i Recovery', '', pb)}
</div>
<div class="flex-row" style="margin-top:12px;">
  {panel_card('💨 Oddychanie', '', pc)}
  {panel_card('⛽ Paliwo i Metabolizm', '', pd_html)}
</div>'''

        # --- EFFICIENCY / ECONOMY (E06) — test_device determines RE vs GAIN ---
        if e06.get('status') == 'OK':
            pe = ''
            _test_dev = ct.get('test_device', 'treadmill')
            _show_re = _test_dev == 'treadmill'
            if _show_re:
                # Running: show Running Economy as primary
                pe += '<div class="sub-header">Ekonomia biegu (Running Economy)</div>'
                pe += row_item('RE @VT1', f'{_n(re_vt1,".1f")} ml/kg/km', re_class, f'VO₂ {_n(e06.get("vo2_at_vt1"),".1f")} ml/kg/min @ {_n(e06.get("load_at_vt1"),".1f")} km/h')
                pe += row_item('RE @VT2', f'{_n(re_vt2,".1f")} ml/kg/km', '', f'VO₂ {_n(e06.get("vo2_at_vt2"),".1f")} ml/kg/min @ {_n(e06.get("load_at_vt2"),".1f")} km/h')
                pe += row_item('RE ogólna', f'{_n(e06.get("running_economy_mlkgkm"),".1f")} ml/kg/km', '', '')
                # Gain as secondary info
                pe += '<div style="font-size:10px;color:#94a3b8;margin-top:4px;">Gain <VT1: {0} {1} (R²={2}) | z={3}</div>'.format(
                    _n(gain_below_vt1,".2f"), gain_unit, _n(e06.get("gain_below_vt1_r2"),".3f"), _n(gain_z,".2f"))
            else:
                # Bike/other: show Gain and Delta Efficiency as primary
                pe += '<div class="sub-header">Efektywność (Gain / Delta Efficiency)</div>'
                pe += row_item('Gain <VT1', f'{_n(gain_below_vt1,".2f")} {gain_unit}', f'z={_n(gain_z,".2f")}', f'R²={_n(e06.get("gain_below_vt1_r2"),".3f")}')
                pe += row_item('Gain @VT1', f'{_n(gain_vt1,".2f")} {gain_unit}', '', '')
                pe += row_item('Gain @VT2', f'{_n(gain_vt2,".2f")} {gain_unit}', '', '')
                if delta_eff:
                    pe += row_item('Delta Efficiency', f'{_n(delta_eff,".1f")}%', '', '')
            if lin_break:
                pe += row_item('Linearity break', _fmt_dur(lin_break), '', f'@ {_n(e06.get("linearity_break_load"),".1f")} {"km/h" if _show_re else "W"}')
            h += f'<div style="margin-top:12px;">{panel_card("🏃 Efektywność" if _show_re else "⚡ Efektywność", "", pe)}</div>'

        h += '</div>'  # end section III

        # ═══════════════════════════════════════════════════════════════
        # IV. WALIDACJA KRZYŻOWA
        # ═══════════════════════════════════════════════════════════════
        h += f'<div class="section"><div class="card">{section_title("Walidacja krzyżowa", "IV")}'

        # Concordance checks
        cc = e19.get('concordance_checks',[])
        if cc:
            checks_html = ''
            for item in cc:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    name_c, status, note = item[0], item[1], item[2]
                    pts = item[3] if len(item) > 3 else 0
                else: continue
                col = '#16a34a' if status in ('EXCELLENT','GOOD','PASS','TIGHT') else ('#d97706' if status in ('MODERATE','FAIR','WIDE') else '#dc2626')
                icon = '✓' if col == '#16a34a' else ('~' if col == '#d97706' else '✗')
                checks_html += f'<span style="display:inline-block;margin:2px 4px;padding:3px 8px;border-radius:4px;font-size:11px;background:{col}14;color:{col};font-weight:600;">{icon} {name_c}: {status}</span>'
            h += f'<div style="margin-bottom:8px;">{checks_html}</div>'

        # Temporal alignment
        for bp_name, bp_data in ta.items():
            if not bp_data: continue
            srcs = bp_data.get('sources',{})
            sp = bp_data.get('spread_sec','?')
            conf = bp_data.get('confidence','?')
            outlier = bp_data.get('outlier','')
            tc = '#16a34a' if conf=='HIGH' else ('#d97706' if conf=='MODERATE' else '#dc2626')
            src_parts = [f'{k} {_fmt_dur(sv)}' for k,sv in srcs.items()]
            h += f'<div style="font-size:11px;padding:3px 0;"><b>{bp_name}</b>: {" | ".join(src_parts)} — <span style="color:{tc};font-weight:600;">spread {_n(sp,".0f")}s ({conf})</span>'
            if outlier:
                h += f' <span style="color:#dc2626;font-size:10px;">(outlier: {outlier})</span>'
            h += '</div>'

        # Validity criteria
        vc19 = e19.get('validity_criteria',{})
        if vc19:
            vc_chips = ''
            for k, val_tuple in vc19.items():
                if isinstance(val_tuple, (list, tuple)) and len(val_tuple) >= 3:
                    s, p, d = val_tuple[0], val_tuple[1], val_tuple[2]
                else: continue
                col = '#16a34a' if s in ('EXCELLENT','YES','OPTIMAL','GOOD','BOTH') else ('#d97706' if s in ('FAIR','VT1_ONLY','ACCEPTABLE') else ('#dc2626' if s in ('NO','POOR','FAIL','INSUFFICIENT') else '#94a3b8'))
                vc_chips += f'<span style="display:inline-block;margin:1px 3px;padding:2px 6px;border-radius:3px;font-size:10px;background:{col}14;color:{col};font-weight:600;">{k}: {s} +{p}</span>'
            h += f'<div style="margin-top:6px;"><span style="font-size:10px;color:#64748b;">Ważność {val_score}/100 ({val_grade}): </span>{vc_chips}</div>'

        h += '</div></div>'

        # ═══════════════════════════════════════════════════════════════
        # V. NIRS / SmO2
        # ═══════════════════════════════════════════════════════════════
        if e12.get('status') == 'OK' and smo2_min is not None:
            h += f'<div class="section"><div class="card">{section_title("NIRS / SmO₂", "V")}'
            h += row_item('SmO₂ rest→min→recovery', f'{_n(smo2_rest,".0f")}→{_n(smo2_min,".0f")}→{_n(smo2_recov,".0f")}%', e12_quality, f'Desaturacja: {_n(desat_abs,".0f")}% abs ({_n(desat_pct,".0f")}% rel.)')
            h += row_item('SmO₂ @VT1 / @VT2', f'{_n(smo2_vt1,".0f")}% / {_n(smo2_vt2,".0f")}%', '', f'@peak: {_n(smo2_peak_val,".0f")}%')
            h += row_item('Half Recovery Time', f'{_n(hrt_s,".1f")}s', '', f'Reox rate: {_n(reox_rate,".3f")} %/s | Overshoot: +{_n(overshoot,".0f")}%')

            if bp1_t:
                bp1_ctx = f'{_n(bp1_vs_vt1,".0f")}s po VT1' if bp1_vs_vt1 and float(bp1_vs_vt1 or 0)>0 else f'{_n(abs(float(bp1_vs_vt1 or 0)),".0f")}s przed VT1' if bp1_vs_vt1 else ''
                h += row_item('NIRS BP1', f'{_fmt_dur(bp1_t)}', '', f'SmO₂: {_n(bp1_smo2,".0f")}% | {bp1_ctx}')
            if bp2_t:
                bp2_ctx = f'{_n(bp2_vs_vt2,".0f")}s po VT2' if bp2_vs_vt2 and float(bp2_vs_vt2 or 0)>0 else f'{_n(abs(float(bp2_vs_vt2 or 0)),".0f")}s przed VT2' if bp2_vs_vt2 else ''
                h += row_item('NIRS BP2', f'{_fmt_dur(bp2_t)}', '', f'SmO₂: {_n(bp2_smo2,".0f")}% | {bp2_ctx}')

            if e12_flags:
                flags_str = ', '.join(e12_flags) if isinstance(e12_flags, list) else str(e12_flags)
                h += f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Flagi: {flags_str}</div>'
            h += '</div></div>'

        # ═══════════════════════════════════════════════════════════════
        # VI. PORÓWNANIE SPORTOWE
        # ═══════════════════════════════════════════════════════════════
        try:
            _vo2v = float(vo2_rel) if vo2_rel else 0
            h += f'<div class="section"><div class="card">{section_title("Porównanie sportowe", "VI")}'
            # Use real E15 sport class boundaries — sex and modality aware
            _sex_key = v('sex', 'male')
            _mod_key = modality if modality in ('run', 'bike') else 'default'
            _sport_labels = {
                'ELITE': 'Elita', 'SUB_ELITE': 'Sub-elite', 'COMPETITIVE': 'Zawodnik',
                'TRAINED': 'Wytrenowany', 'RECREATIONAL': 'Amator', 'UNTRAINED': 'Początkujący'
            }
            try:
                from engines import Engine_E15_Normalization as E15_VO2max_Norms
                _tbl = E15_VO2max_Norms.VO2MAX_SPORT_FEMALE if _sex_key == 'female' else E15_VO2max_Norms.VO2MAX_SPORT_MALE
                _ranges = _tbl.get(_mod_key, _tbl.get('default', []))
            except:
                # Fallback if engines not importable
                if _sex_key == 'female':
                    _ranges = [(0,35,'UNTRAINED'),(35,45,'RECREATIONAL'),(45,53,'TRAINED'),(53,58,'COMPETITIVE'),(58,65,'SUB_ELITE'),(65,999,'ELITE')]
                else:
                    _ranges = [(0,40,'UNTRAINED'),(40,50,'RECREATIONAL'),(50,58,'TRAINED'),(58,65,'COMPETITIVE'),(65,72,'SUB_ELITE'),(72,999,'ELITE')]

            # Build sport_refs in descending order (elite first)
            sport_refs = []
            for lo, hi, cls in reversed(_ranges):
                lbl = _sport_labels.get(cls, cls)
                hi_disp = hi if hi < 900 else f'{lo}+'
                sport_refs.append((lbl, lo, min(hi, lo + 15)))

            h += '<div style="display:flex;gap:4px;align-items:flex-end;height:120px;margin-bottom:6px;">'
            for lbl, lo, hi in sport_refs:
                mid = (lo + hi) / 2
                bar_h = mid * 1.2
                is_you = lo <= _vo2v < hi or (hi >= 900 and _vo2v >= lo)
                h += f'<div style="flex:1;text-align:center;">'
                h += f'<div style="font-size:8px;color:#64748b;margin-bottom:2px;">{lbl}</div>'
                h += f'<div style="height:{bar_h}px;background:{"#3b82f6" if is_you else "#e2e8f0"};border-radius:4px 4px 0 0;display:flex;align-items:flex-end;justify-content:center;">'
                hi_lbl = f'{hi}+' if hi >= 900 else str(hi)
                h += f'<span style="font-size:9px;color:{"white" if is_you else "#94a3b8"};padding:2px;">{lo}-{hi_lbl}</span></div></div>'
            h += f'<div style="flex:1;text-align:center;">'
            h += f'<div style="font-size:9px;font-weight:700;color:#3b82f6;">👉 Ty</div>'
            h += f'<div style="font-size:18px;font-weight:700;color:#0f172a;">{_n(vo2_rel)}</div>'
            h += '</div></div>'
            h += f'<div style="font-size:10px;color:#94a3b8;text-align:center;">Normy sportowe: {"♀ kobiece" if _sex_key=="female" else "♂ męskie"} | {"bieg" if _mod_key=="run" else ("rower" if _mod_key=="bike" else "ogólne")}</div>'
            h += '</div></div>'
        except: pass

        # ═══════════════════════════════════════════════════════════════
        # VII. WYKRESY
        # ═══════════════════════════════════════════════════════════════
        proto_tbl = v('_protocol_table_html','')
        if proto_tbl and proto_tbl.strip():
            h += '<div class="section"><div class="card">'
            h += '<div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="var b=this.parentElement.querySelector(\'.proto-body\');b.style.display=b.style.display==\'none\'?\'block\':\'none\';">'
            h += f'{section_title("Protokół testu")}'
            h += '<span style="font-size:18px;color:#94a3b8;">▸</span></div>'
            h += '<div class="proto-body" style="display:none;">' + proto_tbl + '</div>'
            h += '</div></div>'

        charts_html = ReportAdapter._render_charts_html(ct)
        h += f'<div class="section">{section_title("Wykresy", "VII")}{charts_html}</div>'

        # Footer
        h += '<div style="padding:14px;font-size:10px;color:#9ca3af;text-align:center;">CPET Report v4.0 | Pełna analiza fizjologiczna | Automatycznie generowany</div>'
        h += '</div></body></html>'

        return h

    @staticmethod
    def render_lite_html_report(ct):
        """
        LITE report — for athletes/clients.
        Simplified, visual, actionable. No raw diagnostic data.
        Uses same canon_table (ct) as PRO report.
        """
        import numpy as np
        g = ct.get
        def v(key, default='-'):
            val = g(key, default)
            if val in (None, '', '[BRAK]', 'None', 'nan'): return default
            return str(val)
        def esc(s):
            return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        def _n(val, fmt='.1f', default='-'):
            try:
                n = float(val)
                return f'{n:{fmt}}'
            except: return default
        def _fmt_dur(sec):
            try:
                s=int(float(sec)); return f'{s//60}:{s%60:02d}'
            except: return '-'
        def badge(text, color='#64748b', bg=None):
            if not bg:
                bg = color + '18'
            return f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;background:{bg};color:{color};">{text}</span>'

        def gauge_svg(score, label, size=90, subtitle=''):
            pct=max(0,min(100,float(score) if score else 0))
            if pct>=85: col='#16a34a'
            elif pct>=70: col='#65a30d'
            elif pct>=55: col='#d97706'
            elif pct>=40: col='#ea580c'
            else: col='#dc2626'
            if pct == 0: col='#cbd5e1'
            r=size*0.38;cx=size/2;cy=size*0.48
            circ=2*3.14159*r;dash=circ*pct/100;gap=circ-dash
            disp = f'{int(score)}' if score else '\u2014'
            h_svg = size + (24 if subtitle else 14)
            sub_txt = f'<text x="{cx}" y="{size+18}" text-anchor="middle" font-size="8" fill="#94a3b8">{subtitle}</text>' if subtitle else ''
            return f'<svg width="{size}" height="{h_svg}" viewBox="0 0 {size} {h_svg}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="7"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="7" stroke-dasharray="{dash} {gap}" stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/><text x="{cx}" y="{cy+2}" text-anchor="middle" dominant-baseline="middle" font-size="18" font-weight="700" fill="{col}">{disp}</text><text x="{cx}" y="{cy+r+12}" text-anchor="middle" font-size="9" font-weight="600" fill="#475569">{label}</text>{sub_txt}</svg>'

        # =====================================================================
        # GATHER DATA
        # =====================================================================
        _raw_name = str(v('athlete_name','')).strip()
        name = _raw_name if _raw_name and _raw_name not in ('AUTO','Nieznany Zawodnik','-','') else (str(v('athlete_id','')).strip() or 'Sportowiec')
        age = v('age_y'); sex_pl = 'M' if v('sex')=='male' else 'K'
        weight = v('body_mass_kg'); height = v('height_cm')
        modality = v('modality','run')
        test_date = v('test_date','')
        protocol = v('protocol_name')
        _td = ct.get('test_device', 'treadmill')
        _device_pl = {'treadmill': 'Bieżnia', 'bike_erg': 'Rower', 'rowing_erg': 'Ergometr wiośl.'}.get(_td, 'Bieżnia' if modality == 'run' else 'Rower')
        _sport_map = {'crossfit': 'CrossFit', 'hyrox': 'HYROX', 'triathlon': 'Triathlon', 'mma': 'MMA', 'swimming': 'Pływanie', 'soccer': 'Piłka nożna', 'xc_ski': 'Biegi narc.', 'rowing': 'Wioślarstwo'}
        _sport_tag = f' • {_sport_map[modality]}' if modality in _sport_map else ''

        e00 = g('_e00_raw',{})
        t_stop = e00.get('t_stop',0)
        dur_str = _fmt_dur(t_stop)

        vo2_rel = g('VO2max_ml_kg_min')
        vo2_abs = g('VO2max_L_min') or g('VO2max_abs_Lmin')
        e15 = g('_e15_raw',{})
        vo2_class = e15.get('vo2_class_pop','?')
        vo2_pctile = e15.get('vo2_percentile_approx',50)
        vo2_det = e15.get('vo2_determination','VO2peak')
        vo2_class_sport = e15.get('vo2_class_sport_desc','')
        vo2_pct_pred = g('VO2_pct_predicted')
        hr_peak = g('HR_peak')
        rer_peak = g('RER_peak')

        vt1_hr = g('VT1_HR_bpm'); vt1_time = g('VT1_Time_s'); vt1_pct = g('VT1_pct_VO2peak')
        vt2_hr = g('VT2_HR_bpm'); vt2_time = g('VT2_Time_s'); vt2_pct = g('VT2_pct_VO2peak')
        vt1_speed = g('VT1_Speed'); vt2_speed = g('VT2_Speed')

        e16 = g('_e16_raw',{})
        zones_data = e16.get('zones',{})
        thr_gap = e16.get('threshold_gap_bpm')
        aer_reserve = e16.get('aerobic_reserve_pct')

        e10 = g('_e10_raw',{})
        fatmax = e10.get('mfo_gmin'); fatmax_hr = e10.get('fatmax_hr')
        fatmax_pct_vo2 = e10.get('fatmax_pct_vo2peak')
        fatmax_zone_lo = e10.get('fatmax_zone_hr_low'); fatmax_zone_hi = e10.get('fatmax_zone_hr_high')
        cop_pct_vo2 = g('COP_pct_vo2peak') or e10.get('cop_pct_vo2peak')
        cop_hr = e10.get('cop_hr')
        zone_sub = e10.get('zone_substrate', {})

        e08 = g('_e08_raw',{})
        hrr1 = e08.get('hrr_1min')
        hrr3 = e08.get('hrr_3min')

        e03 = g('_e03_raw',{})
        ve_vco2_slope = g('VE_VCO2_slope') or e03.get('slope_to_vt2')
        ve_vco2_nadir = g('VE_VCO2_nadir')

        e06 = g('_e06_raw',{})
        gain_z = e06.get('gain_z_score')
        gain_full = g('GAIN_full')
        re_mlkgkm = g('RE_mlkgkm')

        e05 = g('_e05_raw',{})
        o2p_pct = e05.get('pct_predicted_friend')
        o2p_trajectory = g('O2pulse_trajectory')
        o2p_ff = g('O2pulse_ff')

        e07 = g('_e07_raw',{})
        bf_peak = e07.get('bf_peak') if e07 else None
        breathing_strategy = e07.get('strategy','') if e07 else ''
        breathing_flags = e07.get('flags',[]) if e07 else []

        _pc = ct.get('_performance_context', {})

        try: pctile_val = float(vo2_pctile) if vo2_pctile else 50
        except: pctile_val = 50

        # =====================================================================
        # UNIFIED PROFILE SCORING (computed in build_canon_table)
        # =====================================================================
        def _sf(val, default=None):
            try: return float(val) if val not in (None, '', '-', '[BRAK]', 'None', 'nan') else default
            except: return default

        _profile = ct.get('_profile') or compute_profile_scores(ct)  # fallback if not pre-computed
        _cat = _profile['categories']
        _overall = _profile['overall']
        _grade = _profile['grade']
        _grade_col = _profile['grade_color']
        _limiter_key = _profile['limiter_key']
        _super_key = _profile['super_key']
        _limiter = _profile['limiter']
        _superpower = _profile['superpower']
        pctile_val = _profile['pctile_val']
        _hrr_score = _profile['gauge_scores']['recovery']
        _vt2_score = _profile['gauge_scores']['vt2']
        _vent_score = _profile['gauge_scores']['ventilation']
        _econ_score = _profile['gauge_scores']['economy']

        # =====================================================================
        # BUILD HTML
        # =====================================================================

        h = f'''<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPET Report LITE — {esc(name)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f8fafc;color:#0f172a;line-height:1.6;}}
.wrap{{max-width:920px;margin:0 auto;padding:20px;}}
.card{{background:white;border-radius:14px;border:1px solid #e2e8f0;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);}}
.section-icon{{font-size:22px;margin-right:8px;vertical-align:middle;}}
.section-title{{font-size:16px;font-weight:700;color:#0f172a;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #e2e8f0;}}
@media print{{body{{background:white;}} .wrap{{max-width:100%;padding:10px;}} .card{{break-inside:avoid;box-shadow:none;}}}}
</style></head><body><div class="wrap">'''

        # ─── HEADER ───
        _logo_b64 = "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iR8OTUllfeDVGX1dZS1JFUyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDkyOS41MiA3MTIuNDIiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIHN0cm9rZS13aWR0aDogMS41cHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMSwgLmNscy0yLCAuY2xzLTMsIC5jbHMtNCwgLmNscy01IHsKICAgICAgICBmaWxsOiBub25lOwogICAgICB9CgogICAgICAuY2xzLTEsIC5jbHMtMiwgLmNscy0zLCAuY2xzLTUgewogICAgICAgIHN0cm9rZS1taXRlcmxpbWl0OiAxMDsKICAgICAgfQoKICAgICAgLmNscy0xLCAuY2xzLTUgewogICAgICAgIHN0cm9rZTogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBzdHJva2U6IHJnYmEoMjU1LDI1NSwyNTUsMC44KTsKICAgICAgfQoKICAgICAgLmNscy0yLCAuY2xzLTMsIC5jbHMtNSB7CiAgICAgICAgc3Ryb2tlLXdpZHRoOiAzcHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMyB7CiAgICAgICAgc3Ryb2tlOiByZ2JhKDI1NSwyNTUsMjU1LDAuNSk7CiAgICAgIH0KCiAgICAgIC5jbHMtNiB7CiAgICAgICAgZmlsbDogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy03IHsKICAgICAgICBjbGlwLXBhdGg6IHVybCgjY2xpcHBhdGgtMSk7CiAgICAgIH0KCiAgICAgIC5jbHMtOCB7CiAgICAgICAgY2xpcC1wYXRoOiB1cmwoI2NsaXBwYXRoKTsKICAgICAgfQogICAgPC9zdHlsZT4KICAgIDxjbGlwUGF0aCBpZD0iY2xpcHBhdGgiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik03NTEuODYsMzE5LjkzaC0zNS4wN2wtMjYuMzktNzQuMTQtOC45MywxNi4yNi0zNC45Ni03My40NC0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41LDE4LjktNTUuMzMtNjYuNzktNDAuMjYtODIuMDhjNzIuMSw0LjIzLDEzOS4zLDM0LjMyLDE5MC44LDg1LjgyLDQxLjc1LDQxLjc1LDY5LjQzLDkzLjgzLDgwLjY3LDE1MC40N2wuMjEsMS4yNloiLz4KICAgIDwvY2xpcFBhdGg+CiAgICA8Y2xpcFBhdGggaWQ9ImNsaXBwYXRoLTEiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik00MzguNzIsODIuODFsLTEwNS45NCwxNjMuOTUtNDIuMTUtNjEuMDYtMjQuOTIsODEuMjItMzkuMDEsNDguNTctNS44NS0yMy42Ny0xNC41NCwzMS40OC0xLjczLTYuMjgtMjIuMTcsMjkuNTYtMTMuMTIuMDVjNi43Ny02Ny40LDM2LjMtMTI5Ljg5LDg0Ljg0LTE3OC40Myw1MC4wMS01MC4wMiwxMTQuODQtNzkuODUsMTg0LjU5LTg1LjM5WiIvPgogICAgPC9jbGlwUGF0aD4KICA8L2RlZnM+CiAgPCEtLSBiZyByZW1vdmVkIC0tPgogIDxnPgogICAgPGc+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTc3Mi40OSw1MjAuODRoLTcuODd2NjguNjdoNS44M2MyLjIzLDAsNC4yLS44Miw1LjktMi40OCwxLjctMS42NSwyLjYtMy42OSwyLjctNi4xMnYtNTMuNTFjMC0xLjg1LS42My0zLjM4LTEuOS00LjU5LTEuMjYtMS4yMS0yLjgyLTEuODctNC42Ny0xLjk3WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTYiIGQ9Ik03NzEuNzYsNDIzLjE1aC03LjE0djY2LjYzaDcuODdjMS44NSwwLDMuNC0uNjMsNC42Ny0xLjksMS4yNi0xLjI2LDEuOS0yLjc3LDEuOS00LjUydi01Mi45M2MwLTIuMDQtLjcxLTMuNzQtMi4xMS01LjEtMS40MS0xLjM2LTMuMTMtMi4wOS01LjE4LTIuMTlaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTY3Mi42MywzNzYuMkg2OS4zMnYyNTguNGg3ODV2LTI1OC40aC0xODEuNjlaTTE0NS43OCw0MjEuMTFoNy4xNGM0Ljg2LDAsNy4yOSwzLjIxLDcuMjksOS42MnY2My4xM2MwLDIuODItLjczLDUuMS0yLjE5LDYuODUtMS40NiwxLjc1LTMuMTYsMi42Mi01LjEsMi42MmgtNy4xNHYtODIuMjNaTTEwNi40Miw2MjAuNDJoLTIyLjkydi0yMzAuMDVoMjIuOTJ2MjMwLjA1Wk0zNTcuOTMsNTgyLjY2YzAsMi43Mi0uNzEsNC45Ni0yLjExLDYuNzEtMS40MSwxLjc1LTMuMTMsMi42Mi01LjE4LDIuNjJzLTMuNjItLjg4LTUuMDMtMi42MmMtMS40MS0xLjc1LTIuMTEtMy45OC0yLjExLTYuNzF2LTkzLjYxYzAtMy41LjctNi4wNSwyLjExLTcuNjUsMS40MS0xLjYsMy4wOC0yLjQxLDUuMDMtMi40MSwyLjA0LDAsMy43Ny44LDUuMTgsMi40MSwxLjQxLDEuNiwyLjExLDQuMTYsMi4xMSw3LjY1djkzLjYxWk0zNTguNzMsNjIwLjQyYzMuOTMtMi4xOCw3LjY1LTUuNzMsMTEuMTYtMTAuNjRsNC43NCwxMC42NGgtMTUuOVpNNDA2LjQ4LDYyMC40MmgtOS4xOXYtMTY5Ljg2aC0yMi42bC0zLjk0LDkuMTljLTQuNzYtNy43OC0xMC44NC0xMS42Ni0xOC4yMy0xMS42NmgtMTUuNDZjLTkuODIsMC0xNy43NywzLjEzLTIzLjg0LDkuNC02LjA4LDYuMjctOS4xMSwxNC4zMS05LjExLDI0LjEzdjEwNi43M2MwLDEwLjYsMy4xMywxOS4wMyw5LjQsMjUuMywyLjk2LDIuOTYsNi4zMiw1LjIxLDEwLjA3LDYuNzhoLTQ0Ljk5YzMuMzUtMS41MSw2LjQ0LTMuNjcsOS4yNi02LjQ4LDYuMDctNi4wNyw5LjExLTEzLjM5LDkuMTEtMjEuOTR2LTM0LjI2aC0zOS4zN3YyNS41MmMwLDIuNzItLjcxLDQuODYtMi4xMSw2LjQyLTEuNDEsMS41Ni0zLjA5LDIuMzMtNS4wMywyLjMzLTQuODYtLjM5LTcuMjktMy4zLTcuMjktOC43NXYtMzguMDVoNTMuOHYtNjYuNjNjMC04LjQ2LTIuOTctMTUuNjUtOC44OS0yMS41OC01LjkzLTUuOTMtMTMuMTItOC44OS0yMS41OC04Ljg5aC0zMi4yMmMtOC4zNiwwLTE1LjUzLDIuOTctMjEuNTEsOC44OS01Ljk4LDUuOTMtOC45NywxMy4xMi04Ljk3LDIxLjU4djExMy40NGMuMTksOC43NSwzLjI4LDE2LjExLDkuMjYsMjIuMDksMi43NSwyLjc1LDUuNzcsNC44NSw5LjA2LDYuMzRoLTc2LjM1di04Ni4wMmgyMC4yN2M1LjczLDAsMTEuMTUtMS4wMiwxNi4yNi0zLjA2LDUuMS0yLjA0LDkuMjYtNS40NywxMi40Ny0xMC4yOCwzLjIxLTQuODEsNC44MS0xMS4zNSw0LjgxLTE5LjYxdi03Ny4yOGMwLTcuNjgtMS42My0xNC4wNS00Ljg4LTE5LjEtMy4yNi01LjA1LTcuNDktOC44LTEyLjY5LTExLjIzLTQuMTEtMS45Mi04LjQyLTMuMDctMTIuOTMtMy40N2gyMzcuNHYyMzAuMDVaTTI0My4xOCw1MTQuMTR2LTI1LjY2YzAtMi43Mi43LTQuOTgsMi4xMS02Ljc4LDEuNDEtMS44LDMuMTMtMi43LDUuMTgtMi43czMuNjIuOSw1LjAzLDIuN2MxLjQxLDEuOCwyLjExLDQuMDYsMi4xMSw2Ljc4djI1LjY2aC0xNC40NFpNNDQ1Ljg1LDYyMC40MnYtNjAuMDdsLjczLTEuMzEsMjAuOTQsNjEuMzhoLTIxLjY3Wk01MjIuNTIsNjIwLjQyaC05LjgxbC00MC4wNC0xMDIuMjEsMzcuNDctNjcuNjVoLTQ1LjQ5bC0xOC44MSwzOS41MXYtOTkuN2g3Ni42N3YyMzAuMDVaTTYxMS43Niw2MjAuNTdoLTY5Ljk5di0yMzAuMzdoMzkuMzd2MTk5LjMxaDMwLjYydjMxLjA2Wk02ODIuOTEsNjIwLjI4bC00LjUyLTU3LjQ1aDBsLTEwLjA2LTEyNy43Mi0xMC4wNiwxMjcuNzJoMGwtNC41Miw1Ny40NWgtMzkuMDhsMjMuNjItMjMwLjM3aDYwLjA3bDIzLjQ3LDIzMC4zN2gtMzguOTNaTTgxOC40Miw0NzguMjdjMCw2LjktMS4wNywxMi40Mi0zLjIxLDE2LjU1LTIuMTQsNC4xMy01LjA4LDcuMTItOC44Miw4Ljk3LTMuNzQsMS44NS04LjA0LDIuNzctMTIuOSwyLjc3LDQuNTcsMCw4Ljc1Ljk3LDEyLjU0LDIuOTIsMy43OSwxLjk1LDYuOCw1LjEzLDkuMDQsOS41NSwyLjIzLDQuNDIsMy4zNSwxMC4yMywzLjM1LDE3LjQydjUxLjAzYzAsMjEuNDgtMTAuMjYsMzIuNTEtMzAuNzcsMzMuMWgtNjIuNHYtMjI4LjQ4aDY0LjU5YzcuODcsMCwxNC42LDIuOCwyMC4xOSw4LjM4LDUuNTksNS41OSw4LjM4LDEyLjM3LDguMzgsMjAuMzR2NTcuNDVaIi8+CiAgICA8L2c+CiAgICA8cmVjdCBjbGFzcz0iY2xzLTMiIHg9IjY5LjMyIiB5PSIzNzYuMiIgd2lkdGg9Ijc4NSIgaGVpZ2h0PSIyNTguNCIvPgogICAgPGcgY2xhc3M9ImNscy04Ij4KICAgICAgPGc+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzcuMTQiIHgyPSI3ODYuNzIiIHkyPSIzNy4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjQzLjY2IiB4Mj0iNzg2LjcyIiB5Mj0iNDMuNjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI1MC4xNyIgeDI9Ijc4Ni43MiIgeTI9IjUwLjE3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNTYuNjkiIHgyPSI3ODYuNzIiIHkyPSI1Ni42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjYzLjIiIHgyPSI3ODYuNzIiIHkyPSI2My4yIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNjkuNzIiIHgyPSI3ODYuNzIiIHkyPSI2OS43MiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijc2LjIzIiB4Mj0iNzg2LjcyIiB5Mj0iNzYuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI4Mi43NSIgeDI9Ijc4Ni43MiIgeTI9IjgyLjc1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iODkuMjYiIHgyPSI3ODYuNzIiIHkyPSI4OS4yNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijk1Ljc4IiB4Mj0iNzg2LjcyIiB5Mj0iOTUuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDIuMjkiIHgyPSI3ODYuNzIiIHkyPSIxMDIuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDguODEiIHgyPSI3ODYuNzIiIHkyPSIxMDguODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMTUuMzIiIHgyPSI3ODYuNzIiIHkyPSIxMTUuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjEuODQiIHgyPSI3ODYuNzIiIHkyPSIxMjEuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjguMzUiIHgyPSI3ODYuNzIiIHkyPSIxMjguMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMzQuODciIHgyPSI3ODYuNzIiIHkyPSIxMzQuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDEuMzkiIHgyPSI3ODYuNzIiIHkyPSIxNDEuMzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDcuOSIgeDI9Ijc4Ni43MiIgeTI9IjE0Ny45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTU0LjQyIiB4Mj0iNzg2LjcyIiB5Mj0iMTU0LjQyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTYwLjkzIiB4Mj0iNzg2LjcyIiB5Mj0iMTYwLjkzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTY3LjQ1IiB4Mj0iNzg2LjcyIiB5Mj0iMTY3LjQ1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTczLjk2IiB4Mj0iNzg2LjcyIiB5Mj0iMTczLjk2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTgwLjQ4IiB4Mj0iNzg2LjcyIiB5Mj0iMTgwLjQ4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTg2Ljk5IiB4Mj0iNzg2LjcyIiB5Mj0iMTg2Ljk5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTkzLjUxIiB4Mj0iNzg2LjcyIiB5Mj0iMTkzLjUxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjAwLjAyIiB4Mj0iNzg2LjcyIiB5Mj0iMjAwLjAyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjA2LjU0IiB4Mj0iNzg2LjcyIiB5Mj0iMjA2LjU0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjEzLjA1IiB4Mj0iNzg2LjcyIiB5Mj0iMjEzLjA1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjE5LjU3IiB4Mj0iNzg2LjcyIiB5Mj0iMjE5LjU3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjI2LjA4IiB4Mj0iNzg2LjcyIiB5Mj0iMjI2LjA4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjMyLjYiIHgyPSI3ODYuNzIiIHkyPSIyMzIuNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjIzOS4xMSIgeDI9Ijc4Ni43MiIgeTI9IjIzOS4xMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI0NS42MyIgeDI9Ijc4Ni43MiIgeTI9IjI0NS42MyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1Mi4xNCIgeDI9Ijc4Ni43MiIgeTI9IjI1Mi4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1OC42NiIgeDI9Ijc4Ni43MiIgeTI9IjI1OC42NiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI2NS4xNyIgeDI9Ijc4Ni43MiIgeTI9IjI2NS4xNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3MS42OSIgeDI9Ijc4Ni43MiIgeTI9IjI3MS42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3OC4yIiB4Mj0iNzg2LjcyIiB5Mj0iMjc4LjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyODQuNzIiIHgyPSI3ODYuNzIiIHkyPSIyODQuNzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTEuMjMiIHgyPSI3ODYuNzIiIHkyPSIyOTEuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTcuNzUiIHgyPSI3ODYuNzIiIHkyPSIyOTcuNzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMDQuMjYiIHgyPSI3ODYuNzIiIHkyPSIzMDQuMjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTAuNzgiIHgyPSI3ODYuNzIiIHkyPSIzMTAuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTcuMjkiIHgyPSI3ODYuNzIiIHkyPSIzMTcuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMjMuODEiIHgyPSI3ODYuNzIiIHkyPSIzMjMuODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzAuMzIiIHgyPSI3ODYuNzIiIHkyPSIzMzAuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzYuODQiIHgyPSI3ODYuNzIiIHkyPSIzMzYuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDMuMzUiIHgyPSI3ODYuNzIiIHkyPSIzNDMuMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDkuODciIHgyPSI3ODYuNzIiIHkyPSIzNDkuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNTYuMzgiIHgyPSI3ODYuNzIiIHkyPSIzNTYuMzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNjIuOSIgeDI9Ijc4Ni43MiIgeTI9IjM2Mi45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzY5LjQxIiB4Mj0iNzg2LjcyIiB5Mj0iMzY5LjQxIi8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxnIGNsYXNzPSJjbHMtNyI+CiAgICAgIDxnPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjUwLjE5IiB4Mj0iNDc2LjA0IiB5Mj0iNTAuMTkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI1Ni43IiB4Mj0iNDc2LjA0IiB5Mj0iNTYuNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjYzLjIyIiB4Mj0iNDc2LjA0IiB5Mj0iNjMuMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI2OS43NCIgeDI9IjQ3Ni4wNCIgeTI9IjY5Ljc0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iNzYuMjUiIHgyPSI0NzYuMDQiIHkyPSI3Ni4yNSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjgyLjc3IiB4Mj0iNDc2LjA0IiB5Mj0iODIuNzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI4OS4yOCIgeDI9IjQ3Ni4wNCIgeTI9Ijg5LjI4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iOTUuOCIgeDI9IjQ3Ni4wNCIgeTI9Ijk1LjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDIuMzEiIHgyPSI0NzYuMDQiIHkyPSIxMDIuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDguODMiIHgyPSI0NzYuMDQiIHkyPSIxMDguODMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMTUuMzQiIHgyPSI0NzYuMDQiIHkyPSIxMTUuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjEuODYiIHgyPSI0NzYuMDQiIHkyPSIxMjEuODYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjguMzciIHgyPSI0NzYuMDQiIHkyPSIxMjguMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMzQuODkiIHgyPSI0NzYuMDQiIHkyPSIxMzQuODkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxNDEuNCIgeDI9IjQ3Ni4wNCIgeTI9IjE0MS40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTQ3LjkyIiB4Mj0iNDc2LjA0IiB5Mj0iMTQ3LjkyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTU0LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMTU0LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTYwLjk1IiB4Mj0iNDc2LjA0IiB5Mj0iMTYwLjk1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTY3LjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMTY3LjQ2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTczLjk4IiB4Mj0iNDc2LjA0IiB5Mj0iMTczLjk4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTgwLjQ5IiB4Mj0iNDc2LjA0IiB5Mj0iMTgwLjQ5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTg3LjAxIiB4Mj0iNDc2LjA0IiB5Mj0iMTg3LjAxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTkzLjUyIiB4Mj0iNDc2LjA0IiB5Mj0iMTkzLjUyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjAwLjA0IiB4Mj0iNDc2LjA0IiB5Mj0iMjAwLjA0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjA2LjU1IiB4Mj0iNDc2LjA0IiB5Mj0iMjA2LjU1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjEzLjA3IiB4Mj0iNDc2LjA0IiB5Mj0iMjEzLjA3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjE5LjU4IiB4Mj0iNDc2LjA0IiB5Mj0iMjE5LjU4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjI2LjEiIHgyPSI0NzYuMDQiIHkyPSIyMjYuMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzMi42MSIgeDI9IjQ3Ni4wNCIgeTI9IjIzMi42MSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzOS4xMyIgeDI9IjQ3Ni4wNCIgeTI9IjIzOS4xMyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI0NS42NCIgeDI9IjQ3Ni4wNCIgeTI9IjI0NS42NCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1Mi4xNiIgeDI9IjQ3Ni4wNCIgeTI9IjI1Mi4xNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1OC42NyIgeDI9IjQ3Ni4wNCIgeTI9IjI1OC42NyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI2NS4xOSIgeDI9IjQ3Ni4wNCIgeTI9IjI2NS4xOSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI3MS43IiB4Mj0iNDc2LjA0IiB5Mj0iMjcxLjciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyNzguMjIiIHgyPSI0NzYuMDQiIHkyPSIyNzguMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyODQuNzMiIHgyPSI0NzYuMDQiIHkyPSIyODQuNzMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTEuMjUiIHgyPSI0NzYuMDQiIHkyPSIyOTEuMjUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTcuNzYiIHgyPSI0NzYuMDQiIHkyPSIyOTcuNzYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMDQuMjgiIHgyPSI0NzYuMDQiIHkyPSIzMDQuMjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTAuNzkiIHgyPSI0NzYuMDQiIHkyPSIzMTAuNzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTcuMzEiIHgyPSI0NzYuMDQiIHkyPSIzMTcuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMjMuODIiIHgyPSI0NzYuMDQiIHkyPSIzMjMuODIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzAuMzQiIHgyPSI0NzYuMDQiIHkyPSIzMzAuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzYuODUiIHgyPSI0NzYuMDQiIHkyPSIzMzYuODUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDMuMzciIHgyPSI0NzYuMDQiIHkyPSIzNDMuMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDkuODgiIHgyPSI0NzYuMDQiIHkyPSIzNDkuODgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNTYuNCIgeDI9IjQ3Ni4wNCIgeTI9IjM1Ni40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzYyLjkxIiB4Mj0iNDc2LjA0IiB5Mj0iMzYyLjkxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzY5LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMzY5LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzc1Ljk0IiB4Mj0iNDc2LjA0IiB5Mj0iMzc1Ljk0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzgyLjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMzgyLjQ2Ii8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTMiIHBvaW50cz0iMTA2LjQyIDM0Ni44NiAyNDEuMzQgMzQ2LjM4IDI5MC42MyAxODUuNzEgMzMyLjc5IDI0Ni43NyA0NjIuNDEgNDYuMTUgNjI0LjUgMzc2LjYzIDY4My42MiAzMjEuODcgNjk3LjE5IDM0My42OCA4MTguNDIgMzQzLjY4Ii8+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik04MTguNDIsMzE5Ljk0aC0xMDkuMzlsLTYyLjUxLTEzMS4zMi0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41MSwxOC45LTkzLjUyLTExMi45LTU5LjEyLDExNC4zNi03LjI3LTE5LjM4cy01NS44NiwxMDcuMS02OC41OSwxMzIuMjljLTEzLjM4LTIyLjg1LTYwLjMxLTEwNS4xNS02MC4zMS0xMDUuMTVsLTc3LjA1LDk1Ljk0LTUuMzMtMTkuMzgtMjkuMDcsMzguNzdoLTY5LjEiLz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTUiIHBvaW50cz0iMTA2LjQyIDM2NSAxODcuMDQgMzY1IDIyMC44NSAyOTEuODMgMjMyIDMzNi45IDMwMC44MSAyMjYuNDEgMzI2Ljk3IDI5NC4yNSA0NDIuMyAxNDQuNTIgNTE2LjQ0IDI5OS41OCA1NTAuODUgMjM3LjA3IDYzNi4xMyAzNDQuNjUgNjkwLjQgMjQ1LjggNzIwLjkzIDMzMS41NyA4MTguNDIgMzMxLjU3Ii8+CiAgPC9nPgo8L3N2Zz4="
        h += f'''<div style="padding:20px 24px;background:linear-gradient(135deg,#1e293b 0%,#334155 50%,#475569 100%);border-radius:16px;color:white;margin-bottom:18px;position:relative;overflow:hidden;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:11px;color:#94a3b8;letter-spacing:0.5px;text-transform:uppercase;">Raport z badania wydolnościowego</span>
      </div>
      <div style="font-size:28px;font-weight:800;margin-bottom:4px;letter-spacing:-0.5px;">{esc(name)}</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#cbd5e1;margin-bottom:8px;">
        <span>\U0001f4c5 {test_date}</span>
        <span>\U0001f3c3 {_device_pl}{_sport_tag}</span>
        <span>\u23f1\ufe0f Czas: {dur_str}</span>
        <span>\u2764\ufe0f HR max: {_n(hr_peak,".0f","-")} bpm</span>
      </div>
      <div style="display:flex;gap:14px;font-size:11px;">
        <a href="https://www.peaklab.com.pl" target="_blank" style="color:#93c5fd;text-decoration:none;display:inline-flex;align-items:center;gap:4px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>www.peaklab.com.pl</a>
        <a href="https://www.instagram.com/peak_lab_" target="_blank" style="color:#93c5fd;text-decoration:none;display:inline-flex;align-items:center;gap:4px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>@peak_lab_</a>
      </div>
    </div>
    <div style="flex-shrink:0;margin-left:16px;">
      <img src="data:image/svg+xml;base64,{_logo_b64}" style="height:90px;max-height:90px;opacity:0.9;" alt="PeakLab">
    </div>
  </div>
</div>'''

        # ─── 1. TWÓJ WYNIK ───
        h += f'''<div class="card">
  <div class="section-title"><span class="section-icon">🏆</span>Twój wynik</div>
  <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-size:52px;font-weight:800;color:#0f172a;line-height:1;">{_n(vo2_rel)}</div>
      <div style="font-size:14px;color:#64748b;margin-top:2px;">ml/kg/min</div>
      <div style="font-size:12px;color:#94a3b8;">VO\u2082{"max" if str(vo2_det)=="VO2max" else "peak"}</div>
    </div>
    <div style="flex:1;min-width:200px;">
      <div style="margin-bottom:8px;">
        <span style="font-size:18px;font-weight:700;color:{_grade_col};">{_grade}</span>
        <span style="font-size:12px;color:#94a3b8;margin-left:8px;">Wynik ogólny: {_overall:.0f}/100</span>
      </div>
      <div style="position:relative;height:18px;border-radius:9px;overflow:hidden;display:flex;margin-bottom:4px;">
        <div style="flex:20;background:#dc2626;"></div><div style="flex:20;background:#ea580c;"></div>
        <div style="flex:15;background:#eab308;"></div><div style="flex:15;background:#22c55e;"></div>
        <div style="flex:15;background:#3b82f6;"></div><div style="flex:15;background:#7c3aed;"></div>
        <div style="position:absolute;left:{min(97,max(3,_overall))}%;top:-3px;font-size:20px;transform:translateX(-50%);">\u25bc</div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;">
        <span>S\u0142aby</span><span>Przeci\u0119tny</span><span>Dobry</span><span>Bardzo dobry</span><span>Elitarny</span>
      </div>
      <div style="font-size:12px;color:#475569;margin-top:6px;">
        Kategoria: <b>{esc(str(vo2_class))}</b> (~{_n(vo2_pctile,".0f","?")} percentyl populacyjny){f' | Sportowo: <b>{esc(vo2_class_sport)}</b>' if vo2_class_sport else ''}
      </div>{f"""
      <div style="margin-top:8px;padding:8px 12px;background:#f0f9ff;border-radius:8px;border-left:3px solid #3b82f6;font-size:10px;color:#475569;line-height:1.6;">
        <b>📊 Populacja:</b> Twój VO₂max jest wyższy niż u {_n(vo2_pctile,'.0f','?')}% osób w Twoim wieku (= {esc(str(vo2_class))}).<br>
        <b>🏋️ Sport:</b> W porównaniu do aktywnych zawodników {esc(vo2_class_sport.split('(')[-1].rstrip(')')) if vo2_class_sport and '(' in vo2_class_sport else ''} to poziom {esc(vo2_class_sport.split('(')[0].strip().lower()) if vo2_class_sport else ''}.<br>
        <b>🏆 Wynik ogólny ({_overall:.0f}/100)</b> łączy VO₂max + progi + ekonomię + regenerację — to nie sam VO₂max.
      </div>""" if vo2_class_sport else ''}
    </div>
  </div>
</div>'''

        # ─── 2. PROFIL WYDOLNOŚCI — kompaktowa tabela priorytetów ───
        _sport_demands = _profile.get('sport_demands', {})

        # Sport label for header
        _sp_name = (ct.get('sport','') or ct.get('_sport','') or ct.get('_cfg_sport','') or ct.get('_prot_modality','') or ct.get('modality','')).upper()
        _sp_icon_map = {'RUN':'\U0001f3c3','HYROX':'\U0001f3cb\ufe0f','BIKE':'\U0001f6b4','TRIATHLON':'\U0001f3ca','CROSSFIT':'\U0001f4aa','MMA':'\U0001f94a','SOCCER':'\u26bd','SWIMMING':'\U0001f3ca','ROWING':'\U0001f6a3','XC_SKI':'\u26f7\ufe0f'}
        _sp_icon = _sp_icon_map.get(_sp_name, '\U0001f3af')
        _sp_lbl_map = {'RUN':'biegu','HYROX':'HYROX','BIKE':'kolarstwa','TRIATHLON':'triathlonu','CROSSFIT':'CrossFit','MMA':'MMA','SOCCER':'pi\u0142ki no\u017cnej','SWIMMING':'p\u0142ywania','ROWING':'wio\u015blarstwa','XC_SKI':'narciarstwa biegowego'}
        _sp_lbl = _sp_lbl_map.get(_sp_name, _sp_name)

        _prio = _profile.get('priority_ranking', [])

        h += f'''<div class="card">
  <div class="section-title"><span class="section-icon">\U0001f4ca</span>Profil wydolno\u015bci <span style="font-size:11px;font-weight:400;color:#94a3b8;margin-left:6px;">{_sp_icon} {_sp_lbl}</span></div>'''

        if _prio:
            def _pc(sc):
                if sc >= 85: return '#059669'
                if sc >= 70: return '#0d9488'
                if sc >= 55: return '#d97706'
                return '#dc2626'

            def _pb(idx, sc):
                if idx < 3 and sc < 75:
                    if idx == 0: return '\U0001f534 PRIORYTET #1', '#dc2626'
                    if idx == 1: return '\U0001f534 PRIORYTET #2', '#ea580c'
                    return '\U0001f7e0 PRIORYTET #3', '#ea580c'
                if sc >= 90: return '\U0001f7e2 ATUT', '#059669'
                if sc < 55: return '\U0001f7e0 DO POPRAWY', '#d97706'
                return None, None

            # Table header
            h += '''<div style="border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;margin-top:8px;">
  <div style="display:flex;padding:5px 10px;background:#f1f5f9;font-size:8px;font-weight:600;color:#94a3b8;gap:4px;align-items:center;">
    <span style="width:22px;">#</span><span style="width:22px;"></span><span style="flex:1;">Kategoria</span><span style="width:80px;">Wynik</span><span style="width:28px;"></span><span style="width:90px;"></span>
  </div>'''

            for i, p in enumerate(_prio):
                sc = p['score']
                c = _pc(sc)
                bt, bc = _pb(i, sc)
                stars_str = '\u2605' * p['stars'] + '\u2606' * (3 - p['stars'])
                is_p = i < 3 and sc < 75
                is_a = sc >= 90
                row_bg = '#fef2f208' if is_p else ('#ecfdf508' if is_a else ('#fff' if i % 2 == 0 else '#fafbfc'))

                h += f'<div style="display:flex;padding:6px 10px;align-items:center;gap:4px;background:{row_bg};border-top:1px solid #f1f5f9;">'
                h += f'<span style="width:22px;font-size:10px;font-weight:700;color:{c};">{i+1}</span>'
                h += f'<span style="width:22px;font-size:13px;">{p["icon"]}</span>'
                h += f'<div style="flex:1;"><span style="font-size:11px;font-weight:600;color:#1e293b;">{p["label"]}</span>'
                h += f' <span style="font-size:8px;color:{"#d97706" if p["stars"]>=2 else "#94a3b8"};">{stars_str}</span></div>'
                h += f'<div style="width:80px;height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;">'
                h += f'<div style="width:{max(8,sc)}%;height:100%;background:linear-gradient(90deg,{c}88,{c});border-radius:3px;"></div></div>'
                h += f'<span style="width:28px;text-align:right;font-size:12px;font-weight:800;color:{c};">{sc:.0f}</span>'
                if bt:
                    h += f'<span style="width:90px;font-size:7px;font-weight:700;padding:2px 5px;background:{bc}12;color:{bc};border-radius:3px;border:1px solid {bc}30;text-align:center;">{bt}</span>'
                else:
                    h += '<span style="width:90px;"></span>'
                h += '</div>'

            h += '</div>'

            # Footer legend
            h += '''<div style="margin-top:8px;padding:6px 10px;background:#f8fafc;border-radius:8px;font-size:9px;color:#64748b;line-height:1.5;">
  <b>Skala 0\u2013100:</b> system PeakLab \u0142\u0105cz\u0105cy norm\u0119 populacyjn\u0105, sportow\u0105 i fizjologiczn\u0105.<br>
  <span style="color:#059669;">\u25cf</span>\u00a0>85 = poziom sportowy \u00a0\u00b7\u00a0 <span style="color:#0d9488;">\u25cf</span>\u00a070\u201385 = dobrze \u00a0\u00b7\u00a0 <span style="color:#d97706;">\u25cf</span>\u00a055\u201370 = przeci\u0119tnie \u00a0\u00b7\u00a0 <span style="color:#dc2626;">\u25cf</span>\u00a0&lt;55 = wymaga pracy<br>
  <b>\u2191</b> Na g\u00f3rze = najwi\u0119kszy potencja\u0142 poprawy dla Twojego sportu \u00a0\u00b7\u00a0 <b>\u2193</b> Na dole = Twoja najmocniejsza strona<br>
  <span style="color:#94a3b8;">\u2605\u2605\u2605 krytyczne \u00b7 \u2605\u2605\u2606 wa\u017cne \u00b7 \u2605\u2606\u2606 pomocnicze \u2014 waga kategorii dla Twojego sportu</span>
</div>'''
        if _limiter and _limiter.get('limiter_text'):
            _lim_key_map = {
                'vo2max': 'HIGH_THRESHOLDS_LOW_CEILING',
                'vt2': 'HIGH_BASE_LOW_THRESHOLD',
                'vt1': 'LOW_BASE',
                'economy': 'ECONOMY_LIMITER',
                'ventilation': 'VENTILATORY_LIMITER',
                'cardiac': 'CARDIAC_LIMITER',
                'recovery': 'RECOVERY_LIMITER',
                'substrate': 'SUBSTRATE_LIMITER',
                'breathing': 'VENTILATORY_LIMITER',
            }
            try:
                from e20_training_decision import PhysioSnapshot, TrainingProfile, score_limiters, _scale_session, SPORT_CLASS_RANK
                _e20_results = {}
                for _eid in ['E00','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','E16','E17','E19']:
                    _e20_results[_eid] = ct.get('_' + _eid.lower() + '_raw', {})
                _e20_results['_performance_context'] = ct.get('_performance_context', {})
                _e20_snap = PhysioSnapshot.from_results(_e20_results, None)
                _e20_snap.zones = zones_data
                _e20_mod = ct.get('_prot_modality', 'run')
                _e20_profile = TrainingProfile(modality=_e20_mod)
                _sc_rank = SPORT_CLASS_RANK.get(_e20_snap.sport_class, 1)
                _lim_type = _lim_key_map.get(_limiter_key, 'HIGH_BASE_LOW_THRESHOLD')
                # Use KEY_1 (primary session) for GameChanger
                _gc_session = _scale_session('KEY_1', _e20_snap, _sc_rank, _lim_type, modality=_e20_mod)
            except Exception:
                _gc_session = _limiter.get('tip', '')

            if _gc_session:
                # Manual override from config takes priority
                _gc_display_lite = ct.get('_gc_manual') or _gc_session
                _gc_is_manual_lite = bool(ct.get('_gc_manual'))
                h += f'''<div style="margin-top:14px;padding:14px 16px;background:linear-gradient(135deg,#fefce8,#fef9c3);border-radius:12px;border-left:4px solid #eab308;">
  <div style="font-size:13px;font-weight:700;color:#a16207;margin-bottom:6px;">🎯 GAME CHANGER — trening tygodnia{"  <span style='font-size:10px;'>(✏️ ręcznie)</span>" if _gc_is_manual_lite else ""}</div>
  <div style="font-size:12px;color:#334155;line-height:1.6;">{_gc_display_lite}</div>
  <div style="margin-top:6px;font-size:10px;color:#a16207;">Priorytet #1: {_limiter.get('label','')}</div>
</div>'''

        # Standalone manual GC when no auto limiter was detected
        if not (_limiter and _limiter.get('limiter_text')) and ct.get('_gc_manual'):
            h += f'''<div style="margin-top:14px;padding:14px 16px;background:linear-gradient(135deg,#fefce8,#fef9c3);border-radius:12px;border-left:4px solid #eab308;">
  <div style="font-size:13px;font-weight:700;color:#a16207;margin-bottom:6px;">🎯 GAME CHANGER — trening tygodnia  <span style='font-size:10px;'>(✏️ ręcznie)</span></div>
  <div style="font-size:12px;color:#334155;line-height:1.6;">{ct['_gc_manual']}</div>
</div>'''

        h += '</div>'

        # ─── 3. STREFY TRENINGOWE ───
        zone_colors = ['#94a3b8','#3b82f6','#22c55e','#f97316','#ef4444']
        zone_names = ['Z1 Regeneracja','Z2 Baza tlenowa','Z3 Tempo','Z4 Pr\u00f3g','Z5 VO\u2082max']
        zone_feelings = [
            'Luz, rozmowa bez problemu',
            'Komfortowy wysi\u0142ek, rozmowa OK',
            'Czujesz wysi\u0142ek, kr\u00f3tkie zdania',
            'Ci\u0119\u017cko, pojedyncze s\u0142owa',
            'Maksymalny wysi\u0142ek, bez rozmowy'
        ]
        zone_uses = [
            'Rozgrzewka, sch\u0142adzanie, dzie\u0144 regeneracyjny',
            'D\u0142ugie biegi, baza aerobowa \u2014 tu sp\u0119dzasz 70-80% czasu',
            'Trening tempa, biegi progowe',
            'Interwa\u0142y progowe, wy\u015bcigi 10km-HM',
            'Interwa\u0142y VO\u2082max, szybkie powt\u00f3rzenia'
        ]

        h += '<div class="card">'
        h += '<div class="section-title"><span class="section-icon">\U0001f49a</span>Twoje strefy treningowe</div>'

        # VT1/VT2 cards
        h += '<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">'
        if vt1_hr:
            _vt1_pwr = g('VT1_Power')
            _is_power_dev = g('test_device','') in ('bike_erg','rowing_erg')
            if _is_power_dev and _vt1_pwr:
                _vt1_spd = f' | {_n(_vt1_pwr,".0f")} W'
            else:
                _vt1_spd = f' | {_n(vt1_speed,".1f")} km/h' if vt1_speed else ''
            h += f'<div style="flex:1;min-width:180px;padding:12px;background:#eff6ff;border-radius:10px;border-left:4px solid #3b82f6;"><div style="font-size:11px;color:#475569;font-weight:600;">1. pr\u00f3g wentylacyjny (VT1)</div><div style="font-size:22px;font-weight:700;color:#2563eb;">\u2764\ufe0f {_n(vt1_hr,".0f")} bpm</div><div style="font-size:11px;color:#64748b;">{_n(vt1_pct,".0f")}% VO\u2082max{_vt1_spd}</div><div style="font-size:10px;color:#94a3b8;margin-top:4px;">Poni\u017cej tego t\u0119tna \u2014 spalasz g\u0142\u00f3wnie t\u0142uszcze</div></div>'
        if vt2_hr:
            _vt2_pwr = g('VT2_Power')
            if _is_power_dev and _vt2_pwr:
                _vt2_spd = f' | {_n(_vt2_pwr,".0f")} W'
            else:
                _vt2_spd = f' | {_n(vt2_speed,".1f")} km/h' if vt2_speed else ''
            h += f'<div style="flex:1;min-width:180px;padding:12px;background:#fef2f2;border-radius:10px;border-left:4px solid #ef4444;"><div style="font-size:11px;color:#475569;font-weight:600;">2. pr\u00f3g wentylacyjny (VT2)</div><div style="font-size:22px;font-weight:700;color:#dc2626;">\u2764\ufe0f {_n(vt2_hr,".0f")} bpm</div><div style="font-size:11px;color:#64748b;">{_n(vt2_pct,".0f")}% VO\u2082max{_vt2_spd}</div><div style="font-size:10px;color:#94a3b8;margin-top:4px;">Powy\u017cej tego \u2014 organizm nie nad\u0105\u017ca z usuwaniem mleczanu</div></div>'
        h += '</div>'

        for i, (zname, zcol, zfeel, zuse) in enumerate(zip(zone_names, zone_colors, zone_feelings, zone_uses)):
            zkey = f'z{i+1}'
            zd = zones_data.get(zkey, {})
            hr_lo = zd.get('hr_low','?'); hr_hi = zd.get('hr_high','?')
            spd_lo = zd.get('speed_low',''); spd_hi = zd.get('speed_high','')
            pwr_lo = zd.get('power_low',''); pwr_hi = zd.get('power_high','')
            if _is_power_dev and pwr_lo:
                spd_str = f' | {_n(pwr_lo,".0f","")}-{_n(pwr_hi,".0f","")} W'
            elif spd_lo:
                spd_str = f' | {_n(spd_lo,".1f","")}-{_n(spd_hi,".1f","")} km/h'
            else:
                spd_str = ''

            h += f'''<div style="display:flex;align-items:center;gap:12px;padding:10px;margin-bottom:6px;border-radius:10px;background:{'#fafafa' if i%2==0 else 'white'};">
  <div style="width:14px;height:14px;border-radius:50%;background:{zcol};flex-shrink:0;"></div>
  <div style="flex:1;">
    <div style="font-size:13px;font-weight:600;color:#0f172a;">{zname}</div>
    <div style="font-size:11px;color:#64748b;">{zfeel}</div>
    <div style="font-size:10px;color:#94a3b8;">{zuse}</div>
  </div>
  <div style="text-align:right;flex-shrink:0;">
    <div style="font-size:16px;font-weight:700;color:#0f172a;">{hr_lo}-{hr_hi}</div>
    <div style="font-size:10px;color:#94a3b8;">bpm{spd_str}</div>
  </div>
</div>'''

        h += '</div>'

        # ─── 4. METABOLIZM LIPIDÓW ───
        if fatmax or zone_sub:
            try:
                _fatmax_gh = float(fatmax) * 60 if fatmax else 0
            except:
                _fatmax_gh = 0

            # FATmax + Crossover summary row
            _cop_hr_s = f' | HR {int(cop_hr)} bpm' if cop_hr else ''
            _cop_txt = f'{_n(cop_pct_vo2,".0f")}% VO\u2082max{_cop_hr_s}' if cop_pct_vo2 else '\u2014'

            h += f'''<div class="card">
  <div class="section-title"><span class="section-icon">\u26fd</span>Metabolizm lipid\u00f3w</div>
  <div style="margin-bottom:10px;padding:8px 12px;background:#f0fdf4;border-radius:8px;border-left:3px solid #22c55e;font-size:10px;color:#475569;line-height:1.6;">
    <b>\U0001f4d6 Jak czyta\u0107:</b> <b>Crossover</b> = punkt, w kt\u00f3rym spalanie CHO przekracza FAT \u00b7 <b>FATmax</b> = intensywno\u015b\u0107 z najwy\u017cszym spalaniem t\u0142uszczu w g/h \u00b7 <b>VT1</b> = 1. pr\u00f3g wentylacyjny. Kolejno\u015b\u0107 Crossover &lt; VT1 &lt; FATmax jest fizjologicznie prawid\u0142owa.
  </div>
  <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:160px;padding:12px;background:#fefce8;border-radius:10px;border-left:4px solid #eab308;text-align:center;">
      <div style="font-size:11px;color:#475569;font-weight:600;">FATmax</div>
      <div style="font-size:28px;font-weight:700;color:#ca8a04;">{_fatmax_gh:.0f} <span style="font-size:13px;font-weight:500;">g/h</span></div>
      <div style="font-size:10px;color:#64748b;">HR {_n(fatmax_hr,".0f")} bpm ({_n(fatmax_pct_vo2,".0f")}% VO\u2082max)</div>
      <div style="font-size:10px;color:#94a3b8;margin-top:2px;">Strefa: {_n(fatmax_zone_lo,".0f")}\u2013{_n(fatmax_zone_hi,".0f")} bpm</div>
    </div>
    <div style="flex:1;min-width:160px;padding:12px;background:#eff6ff;border-radius:10px;border-left:4px solid #3b82f6;text-align:center;">
      <div style="font-size:11px;color:#475569;font-weight:600;">Crossover (CHO=FAT)</div>
      <div style="font-size:28px;font-weight:700;color:#2563eb;">{_cop_txt}</div>
      <div style="font-size:10px;color:#64748b;">Powy\u017cej tego \u2014 dominuje glikogen</div>
    </div>
  </div>'''

            # Zone substrate table
            if zone_sub:
                _zone_names = {'z1': 'Z1 Regeneracja', 'z2': 'Z2 Baza tlenowa', 'z3': 'Z3 Tempo', 'z4': 'Z4 Pr\u00f3g', 'z5': 'Z5 VO\u2082max'}
                _zone_colors = {'z1': '#94a3b8', 'z2': '#3b82f6', 'z3': '#22c55e', 'z4': '#f97316', 'z5': '#ef4444'}

                h += '''<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:4px;">
  <thead>
    <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px 6px;text-align:left;color:#475569;font-weight:600;">Strefa</th>
      <th style="padding:8px 6px;text-align:center;color:#eab308;font-weight:600;">\U0001f7e1 FAT g/h</th>
      <th style="padding:8px 6px;text-align:center;color:#3b82f6;font-weight:600;">\U0001f535 CHO g/h</th>
      <th style="padding:8px 6px;text-align:center;color:#475569;font-weight:600;">FAT %</th>
      <th style="padding:8px 6px;text-align:center;color:#ef4444;font-weight:600;">\U0001f525 kcal/h</th>
    </tr>
  </thead>
  <tbody>'''

                for zk in ['z2', 'z3', 'z4', 'z5']:
                    zs = zone_sub.get(zk, {})
                    if zs:
                        _zn = _zone_names.get(zk, zk.upper())
                        _zc = _zone_colors.get(zk, '#475569')
                        _fat_gh = _n(zs.get('fat_gh'), '.0f', '\u2014')
                        _cho_gh = _n(zs.get('cho_gh'), '.0f', '\u2014')
                        _fat_pct = _n(zs.get('fat_pct'), '.0f', '\u2014')
                        _kcal = _n(zs.get('kcal_h'), '.0f', '\u2014')
                        _fat_pct_v = _sf(zs.get('fat_pct'), 0)
                        # Visual bar for FAT%
                        _bar_w = max(0, min(100, _fat_pct_v))
                        _bar_col = '#16a34a' if _fat_pct_v >= 50 else ('#eab308' if _fat_pct_v >= 25 else '#ef4444')
                        h += f'''<tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:8px 6px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{_zc};margin-right:6px;"></span><b>{_zn}</b></td>
      <td style="padding:8px 6px;text-align:center;color:#eab308;font-weight:600;">{_fat_gh}</td>
      <td style="padding:8px 6px;text-align:center;color:#3b82f6;font-weight:600;">{_cho_gh}</td>
      <td style="padding:8px 6px;text-align:center;">
        <div style="display:flex;align-items:center;gap:4px;justify-content:center;">
          <div style="width:40px;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;"><div style="width:{_bar_w}%;height:100%;background:{_bar_col};border-radius:3px;"></div></div>
          <span style="font-size:11px;color:#475569;">{_fat_pct}%</span>
        </div>
      </td>
      <td style="padding:8px 6px;text-align:center;font-weight:600;color:#334155;">{_kcal}</td>
    </tr>'''

                h += '</tbody></table>'

            # Practical tip
            try:
                _cho_z3 = float(zone_sub.get('z3', {}).get('cho_gh', 40)) if zone_sub else 40
                _cho_rec = max(30, int(_cho_z3 * 0.5))
            except:
                _cho_rec = 30
            h += f'''<div style="margin-top:12px;padding:10px;background:#f0fdf4;border-radius:8px;font-size:11px;color:#334155;line-height:1.6;">
    \U0001f4a1 <b>Praktycznie:</b> W Z2 spalasz wi\u0119cej t\u0142uszcz\u00f3w, w Z4-Z5 zu\u017cywasz g\u0142\u00f3wnie glikogen.
    Na d\u0142ugich biegach (>60 min) uzupe\u0142niaj <b>{_cho_rec}\u201360 g CHO/h</b> aby unikn\u0105\u0107 "\u015bciany". Dostosuj do indywidualnej tolerancji \u017co\u0142\u0105dkowej.
  </div>
</div>'''

        # ─── 4b. EKONOMIA ───
        _econ_data = _profile.get('categories', {}).get('economy', {})
        _econ_score = _econ_data.get('score', 0)
        _econ_lbl = _econ_data.get('label', 'Ekonomia ruchu')
        _econ_interp = _profile.get('interpretations', {}).get('economy', '')

        if ct.get('test_device', 'treadmill') == 'treadmill' and re_mlkgkm:
            _re_display = f'{_n(re_mlkgkm, ".0f", "—")}'
            _re_unit = 'ml O\u2082/kg/km'
            _re_v_f = float(re_mlkgkm) if re_mlkgkm else 999
            _re_rating = 'Elitarna' if _re_v_f < 180 else ('Dobra' if _re_v_f < 200 else ('Przeciętna' if _re_v_f < 215 else 'Do poprawy'))
            _re_color = '#16a34a' if _re_v_f < 200 else ('#eab308' if _re_v_f < 215 else '#ef4444')
            _gain_info = f'<div style="font-size:11px;color:#64748b;margin-top:4px;">GAIN (sprawność mechaniczna): {_n(gain_z, "+.1f", "—")}</div>' if gain_z else ''
            h += f'''<div class="card">
  <div class="section-title"><span class="section-icon">\u2699\ufe0f</span>{esc(_econ_lbl)}</div>
  <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
    <div style="text-align:center;padding:12px 20px;background:#f8fafc;border-radius:10px;border-left:4px solid {_re_color};">
      <div style="font-size:11px;color:#475569;font-weight:600;">Running Economy</div>
      <div style="font-size:28px;font-weight:700;color:{_re_color};">{_re_display} <span style="font-size:12px;font-weight:500;">{_re_unit}</span></div>
      <div style="font-size:11px;color:#64748b;">{_re_rating}</div>
      {_gain_info}
    </div>
    <div style="flex:1;min-width:200px;font-size:13px;color:#334155;line-height:1.7;">
      {_econ_interp or f"RE {_re_display} ml/kg/km — im niższe, tym lepsza ekonomia."}
      <div style="margin-top:8px;font-size:11px;color:#64748b;">
        <b>Normy RE (bieg):</b> &lt;180 = elitarna, 180-200 = dobra, 200-215 = przeciętna, &gt;215 = do poprawy
      </div>
    </div>
  </div>
</div>'''
        elif gain_z is not None:
            _gz_v = float(gain_z) if gain_z else 0
            _gz_color = '#16a34a' if _gz_v > 0.5 else ('#eab308' if _gz_v > -0.5 else '#ef4444')
            _gz_rating = 'Ponadprzeciętna' if _gz_v > 0.5 else ('W normie' if _gz_v > -0.5 else 'Do poprawy')
            h += f'''<div class="card">
  <div class="section-title"><span class="section-icon">\u2699\ufe0f</span>{esc(_econ_lbl)}</div>
  <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
    <div style="text-align:center;padding:12px 20px;background:#f8fafc;border-radius:10px;border-left:4px solid {_gz_color};">
      <div style="font-size:11px;color:#475569;font-weight:600;">GAIN (sprawność mechaniczna)</div>
      <div style="font-size:28px;font-weight:700;color:{_gz_color};">{_gz_v:+.1f}</div>
      <div style="font-size:11px;color:#64748b;">{_gz_rating}</div>
    </div>
    <div style="flex:1;min-width:200px;font-size:13px;color:#334155;line-height:1.7;">
      {_econ_interp or "GAIN (sprawność mechaniczna) porównuje Twoją efektywność z normą populacyjną."}
    </div>
  </div>
</div>'''

        # ─── 5. CO TO ZNACZY ───
        h += '<div class="card">'
        h += '<div class="section-title"><span class="section-icon">\U0001f4a1</span>Co oznaczaj\u0105 Twoje wyniki</div>'
        h += '<div style="font-size:13px;color:#334155;line-height:1.8;">'

        # Unified interpretations from _profile (single source of truth)
        _interp_texts = _profile.get('interpretations', {})
        _interp_order = ['vo2max', 'thresholds', 'vt1', 'recovery', 'economy', 'ventilation', 'cardiac', 'substrate']
        _has_any = False
        for _ik in _interp_order:
            _itxt = _interp_texts.get(_ik)
            if _itxt:
                h += f'<p style="margin-bottom:8px;">{_itxt}</p>'
                _has_any = True
        if not _has_any:
            h += '<p>Brak wystarczaj\u0105cych danych do pe\u0142nej interpretacji.</p>'

        h += '</div></div>'

        # ─── 5a. JAK CZYTAĆ TEN RAPORT (legenda, zwijana) ───
        h += '''<div class="card" style="padding:0;">
  <details style="cursor:pointer;">
    <summary style="padding:12px 16px;font-size:13px;font-weight:600;color:#475569;list-style:none;display:flex;align-items:center;gap:8px;">
      <span style="font-size:14px;">\U0001f4d6</span> Jak czyta\u0107 ten raport
      <span style="margin-left:auto;font-size:10px;color:#94a3b8;">\u25bc rozwi\u0144</span>
    </summary>
    <div style="padding:0 16px 14px;font-size:12px;color:#475569;line-height:1.8;border-top:1px solid #e2e8f0;">
      <div style="margin-top:10px;">
        <b style="font-size:13px;">Skr\u00f3ty u\u017cyte w raporcie:</b>
        <table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:11px;">
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:4px 8px;font-weight:600;width:100px;color:#1e293b;">VO\u2082max</td>
            <td style="padding:4px 8px;color:#64748b;">Maksymalne poch\u0142anianie tlenu (ml/kg/min). Im wy\u017csze, tym lepsza wydolno\u015b\u0107 tlenowa.</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;background:#fafbfc;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">VT1 / VT2</td>
            <td style="padding:4px 8px;color:#64748b;">Progi wentylacyjne. VT1 = przej\u015bcie z metabolizmu czysto tlenowego. VT2 = pr\u00f3g kwasowy (lactate threshold).</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">HRR\u2081</td>
            <td style="padding:4px 8px;color:#64748b;">Heart Rate Recovery \u2014 spadek t\u0119tna (bpm) w 1. minucie po zako\u0144czeniu wysi\u0142ku. >25 = dobrze, >40 = bardzo dobrze.</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;background:#fafbfc;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">RER</td>
            <td style="padding:4px 8px;color:#64748b;">Respiratory Exchange Ratio (VCO\u2082/VO\u2082). RER=1.0 \u2192 100% glikogen. RER >1.10 \u2192 pe\u0142ne wyczerpanie (test miarodajny).</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">RE</td>
            <td style="padding:4px 8px;color:#64748b;">Running Economy \u2014 zu\u017cycie O\u2082 na km biegu (ml O\u2082/kg/km). Im ni\u017csze, tym lepiej. <180 = elitarna.</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;background:#fafbfc;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">O\u2082 pulse</td>
            <td style="padding:4px 8px;color:#64748b;">Obj\u0119to\u015b\u0107 tlenu na jedno uderzenie serca (ml/beat). Miara si\u0142y i wydajno\u015bci serca.</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">FATmax</td>
            <td style="padding:4px 8px;color:#64748b;">Intensywno\u015b\u0107 z najwy\u017cszym bezwzgl\u0119dnym spalaniem t\u0142uszczu (g/h).</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;background:#fafbfc;">
            <td style="padding:4px 8px;font-weight:600;color:#1e293b;">Crossover</td>
            <td style="padding:4px 8px;color:#64748b;">Punkt, w kt\u00f3rym spalanie w\u0119glowodan\u00f3w (CHO) przekracza spalanie t\u0142uszcz\u00f3w (FAT).</td>
          </tr>
        </table>
      </div>
      <div style="margin-top:12px;">
        <b style="font-size:13px;">Scoring 0\u2013100:</b>
        <div style="margin-top:4px;font-size:11px;color:#64748b;line-height:1.7;">
          Wewn\u0119trzny system PeakLab \u0142\u0105cz\u0105cy normy populacyjne (wiek + p\u0142e\u0107), sportowe (Twoja dyscyplina) i fizjologiczne.<br>
          <span style="color:#059669;">\u25cf</span>\u00a0<b>>85</b> = poziom sportowy \u00a0\u00b7\u00a0
          <span style="color:#0d9488;">\u25cf</span>\u00a0<b>70\u201385</b> = dobrze \u00a0\u00b7\u00a0
          <span style="color:#d97706;">\u25cf</span>\u00a0<b>55\u201370</b> = przeci\u0119tnie \u00a0\u00b7\u00a0
          <span style="color:#dc2626;">\u25cf</span>\u00a0&lt;55 = wymaga pracy
        </div>
      </div>
      <div style="margin-top:12px;">
        <b style="font-size:13px;">Dwie skale por\u00f3wnawcze:</b>
        <div style="margin-top:4px;font-size:11px;color:#64748b;line-height:1.7;">
          <b>Percentyl populacyjny</b> \u2014 Tw\u00f3j wynik na tle og\u00f3lnej populacji (m\u0119\u017cczy\u017ani/kobiety w Twoim wieku).<br>
          <b>Klasa sportowa</b> \u2014 Tw\u00f3j wynik na tle aktywnych sportowc\u00f3w w Twojej dyscyplinie. Te dwie skale mog\u0105 si\u0119 istotnie r\u00f3\u017cni\u0107.
        </div>
      </div>
    </div>
  </details>
</div>'''

        # ─── 5b. PROTOKÓŁ (zwijany) ───
        _proto_desc = v('_protocol_description', '')
        if not _proto_desc:
            _proto_desc = f'{_device_pl}, {protocol}'
        h += f'''<div class="card" style="padding:0;">
  <details style="cursor:pointer;">
    <summary style="padding:12px 16px;font-size:13px;font-weight:600;color:#475569;list-style:none;display:flex;align-items:center;gap:8px;">
      <span style="font-size:14px;">\U0001f4cb</span> Protokół badania
      <span style="margin-left:auto;font-size:10px;color:#94a3b8;">▼ rozwiń</span>
    </summary>
    <div style="padding:0 16px 14px;font-size:12px;color:#475569;line-height:1.7;border-top:1px solid #e2e8f0;">
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;">
        <div><b>Protokół:</b> {esc(_proto_desc)}</div>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">
        <span><b>Modalność:</b> {_device_pl}</span>
        <span><b>Czas wysiłku:</b> {dur_str}</span>
        <span><b>HR max:</b> {_n(hr_peak, ".0f", "—")} bpm</span>
        <span><b>RER peak:</b> {_n(rer_peak, ".2f", "—")} <span style="color:#94a3b8;font-size:10px;">(VCO₂/VO₂; >1.10 = pełne wyczerpanie {chr(10004) if rer_peak and float(rer_peak) > 1.10 else ""})</span></span>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">
        <span><b>VO\u2082max:</b> {_n(vo2_rel)} ml/kg/min ({_n(vo2_abs)} L/min)</span>
        <span><b>Percentyl pop.:</b> ~{_n(vo2_pctile, ".0f", "?")}%</span>
        <span><b>Klasa sportowa:</b> {esc(vo2_class_sport)}</span>
      </div>
    </div>
  </details>
</div>'''

        # ─── 6. WYKRESY ───
        try:
            charts_html = ReportAdapter._render_charts_html(ct)
        except Exception:
            charts_html = ''
        if charts_html:
            h += f'<div class="card"><div class="section-title"><span class="section-icon">\U0001f4c8</span>Wykresy</div>{charts_html}</div>'

        # ─── FOOTER ───
        h += '<div style="padding:16px;font-size:10px;color:#9ca3af;text-align:center;">CPET Report LITE v1.0 | Wygenerowano automatycznie | Szczeg\u00f3\u0142owy raport PRO dost\u0119pny na \u017cyczenie</div>'
        h += '</div></body></html>'

        return h

    @staticmethod
    def _gasex_report(ct):
        r17 = ct.get('_e17_raw', {})
        if not r17 or r17.get('status') in (None, 'NO_SIGNAL', 'NO_TIME'):
            return ""
        lines = ["", "  3f. Wymiana gazowa / Gas Exchange (E17 v2.0):"]
        sigs = r17.get('available_signals', [])
        gep = r17.get('gas_exchange_pattern', '-')
        # Sport context for E17 text
        _gx_vo2 = ct.get('_interp_vo2_mlkgmin')
        _gx_vs = None
        try:
            _gx_vsr = ct.get('VE_VCO2_slope')
            if _gx_vsr and str(_gx_vsr) not in ('-','[BRAK]','None',''): _gx_vs = float(_gx_vsr)
        except: pass
        _gx_sport = (_gx_vo2 is not None and _gx_vo2 >= 45 and _gx_vs is not None and _gx_vs < 30)
        _gx_vdp = r17.get('vdvt_at_peak')
        if _gx_sport and gep == 'ABNORMAL' and _gx_vdp is not None and _gx_vdp < 0.20:
            _gx_serious = [f for f in r17.get('flags',[]) if f in ('SPO2_SEVERE','VDVT_PARADOXICAL','PETCO2_NO_RISE','HYPERCAPNIA_PEAK')]
            if all(f == 'VDVT_PARADOXICAL' for f in _gx_serious):
                gep = 'SPORT_NORM (reinterpretacja: VD/VT <0.20 u sportowca)'
        lines.append(f"      Dostepne sygnaly: {', '.join(sigs)}")
        lines.append(f"      Ogolna ocena: [{gep}]")
        if 'PetCO2' in sigs:
            lines.append(f"      PetCO2: rest [{r17.get('petco2_rest','-')}] | @VT1 [{r17.get('petco2_at_vt1','-')}] | @VT2 [{r17.get('petco2_at_vt2','-')}] | peak [{r17.get('petco2_at_peak','-')}] mmHg")
            lines.append(f"        rise rest->VT1: [{r17.get('petco2_rise_to_vt1','-')}] | drop VT2->peak: [{r17.get('petco2_drop_vt2_to_peak','-')}]")
            lines.append(f"        Wzorzec: [{r17.get('petco2_pattern','-')}] — {r17.get('petco2_clinical','')}")
        if 'PetO2' in sigs:
            lines.append(f"      PetO2: rest [{r17.get('peto2_rest','-')}] | nadir [{r17.get('peto2_nadir','-')}] | @VT1 [{r17.get('peto2_at_vt1','-')}] | peak [{r17.get('peto2_at_peak','-')}] mmHg | [{r17.get('peto2_pattern','-')}]")
        if 'VD/VT' in sigs:
            _vdp_txt = r17.get('vdvt_pattern','-')
            _vdc_txt = r17.get('vdvt_clinical','')
            if _gx_sport and _gx_vdp is not None and _gx_vdp < 0.20 and _vdp_txt == 'PARADOXICAL_RISE':
                _vdp_txt = 'SPORT_NORM'
                _vdc_txt = f'VD/VT wzrost {r17.get("vdvt_rest","-")}→{_gx_vdp:.3f} — wartości bezwzględne <0.20 = NORMA SPORTOWA (elitarna efektywność wentylacyjna)'
            lines.append(f"      VD/VT: rest [{r17.get('vdvt_rest','-')}] | @VT1 [{r17.get('vdvt_at_vt1','-')}] | peak [{r17.get('vdvt_at_peak','-')}] | delta [{r17.get('vdvt_delta','-')}]")
            lines.append(f"        Wzorzec: [{_vdp_txt}] — {_vdc_txt}")
        if 'SpO2' in sigs:
            lines.append(f"      SpO2: rest [{r17.get('spo2_rest','-')}] | min [{r17.get('spo2_min','-')}] | drop [{r17.get('spo2_drop','-')}]% | [{r17.get('spo2_clinical','-')}]")
        if 'PaCO2_est' in sigs:
            lines.append(f"      PaCO2(est): rest [{r17.get('paco2_rest','-')}] | peak [{r17.get('paco2_peak','-')}] mmHg")
        if 'PAO2_est' in sigs:
            lines.append(f"      PAO2(est): rest [{r17.get('pao2_rest','-')}] | peak [{r17.get('pao2_peak','-')}] mmHg")
        fl = r17.get('flags', [])
        if fl:
            if _gx_sport and _gx_vdp is not None and _gx_vdp < 0.20:
                _gx_sf = {'VDVT_PARADOXICAL','VDVT_NO_DROP','VDVT_HIGH_PEAK'}
                _gx_c = [f for f in fl if f not in _gx_sf]
                _gx_s = [f for f in fl if f in _gx_sf]
                parts = []
                if _gx_s: parts.append(f'sport (VD/VT<0.20): {", ".join(_gx_s)}')
                if _gx_c: parts.append(f'{", ".join(_gx_c)}')
                lines.append(f"      Flagi: [{' | '.join(parts)}]")
            else:
                lines.append(f"      Flagi: {', '.join(fl)}")
        return "\n".join(lines)

    @staticmethod
    def _validity_concordance_report(ct):
        """E19 Test Validity + Concordance report section."""
        r19 = ct.get('_e19_raw', {})
        if not r19 or r19.get('status') != 'OK':
            return ""
        L = ["", "  3g. Wiarygodność testu i spójność fizjologiczna (E19 v1.0):"]
        vs = r19.get('validity_score', 0)
        vg = r19.get('validity_grade', '?')
        L.append(f"      VALIDITY SCORE: [{vs}/100] Grade: [{vg}]")
        vc = r19.get('validity_criteria', {})
        for k, (status, pts, desc) in vc.items():
            L.append(f"        {k}: [{status}] +{pts} — {desc}")
        vf = r19.get('validity_flags', [])
        if vf:
            L.append(f"        Flagi ważności: {', '.join(vf)}")
        cs = r19.get('concordance_score', 0)
        cg = r19.get('concordance_grade', '?')
        L.append(f"      CONCORDANCE SCORE: [{cs}%] Grade: [{cg}]")
        for (name, status, note, pts) in r19.get('concordance_checks', []):
            L.append(f"        {name}: [{status}] — {note}")
        cf = r19.get('concordance_flags', [])
        if cf:
            L.append(f"        Flagi spójności: {', '.join(cf)}")
        ta = r19.get('temporal_alignment', {})
        if ta:
            for bp, data in ta.items():
                srcs = data.get('sources', {})
                sp = data.get('spread_sec', '?')
                conf = data.get('confidence', '?')
                src_str = ', '.join(f'{k}={v:.0f}s' for k,v in srcs.items())
                L.append(f"        {bp} alignment: {src_str} | spread={sp}s | confidence={conf}")
        return "\n".join(L)

    def render_text_report(ct: 'Dict[str, Any]') -> str:
        """Renderuje pełny raport tekstowy wg szablonu T12."""


        # E06 gain report section — contextual (run vs machine)
        _g = ct.get
        _is_run = _g('GAIN_modality','') in ('run','walk')
        _gain_lines = []
        if _is_run:
            _gain_lines.append("VI. EKONOMIA BIEGU (E06 v2)")
            _gain_lines.append(f"\u2022 Running Economy na VT1: [{_g('RE_at_vt1','[BRAK]')}] ml/kg/km (speed {_g('LOAD_at_vt1','-')} km/h)")
            _gain_lines.append(f"\u2022 Running Economy na VT2: [{_g('RE_at_vt2','[BRAK]')}] ml/kg/km (speed {_g('LOAD_at_vt2','-')} km/h) -> {_g('RE_class','-')}")
            _gain_lines.append(f"  -> Norma: Elite <195 | Well-trained <208 | Recreational <225 | Klasyfikacja przy speed >=8 km/h")
            _gain_lines.append(f"• Gain (dVO2/dSpeed) <VT1: [{_g('GAIN_below_vt1','[BRAK]')}] {_g('GAIN_unit','')} (R2={_g('GAIN_below_vt1_r2','-')})")
            _gain_lines.append(f"  -> Gain pelny zakres: [{_g('GAIN_full','[BRAK]')}] {_g('GAIN_unit','')} | Z-score: {_g('GAIN_z','-')}")
        else:
            _gain_lines.append("VI. GAIN / EFEKTYWNOSC WYSILKU (E06 v2)")
            _gain_lines.append(f"• Modalnosc: [{_g('GAIN_modality','-')}]")
            _gain_lines.append(f"• Gain (dVO2/dWatt) <VT1: [{_g('GAIN_below_vt1','[BRAK]')}] {_g('GAIN_unit','')} (R2={_g('GAIN_below_vt1_r2','-')})")
            _gain_lines.append(f"  -> Gain pelny zakres: [{_g('GAIN_full','[BRAK]')}] {_g('GAIN_unit','')}")
            _gain_lines.append(f"  -> Norma: {_g('GAIN_norm_ref','-')} {_g('GAIN_unit','')} ({_g('GAIN_norm_src','')}) | Z-score: {_g('GAIN_z','-')}")
            _gain_lines.append(f"• Gain na progu VT1: [{_g('GAIN_at_vt1','[BRAK]')}] {_g('GAIN_unit','')} (load {_g('LOAD_at_vt1','-')} W)")
            _gain_lines.append(f"• Gain na progu VT2: [{_g('GAIN_at_vt2','[BRAK]')}] {_g('GAIN_unit','')} (load {_g('LOAD_at_vt2','-')} W)")
            _de = _g('DELTA_EFF')
            _de1 = _g('EFF_at_vt1')
            _de2 = _g('EFF_at_vt2')
            _gain_lines.append(f"• Delta Efficiency: srednia [{_de if _de else '[BRAK]'}]% | VT1 [{_de1 if _de1 else '-'}]% | VT2 [{_de2 if _de2 else '-'}]%")
        _lb = _g('LIN_BREAK_time')
        _gain_lines.append("• Zlamanie liniowosci VO2: " + (f"t={int(_lb)}s" if _lb else "nie wykryto"))
        _gain_section = chr(10).join(_gain_lines)
        report = f"""
[A] ANALYSIS_REPORT
RAPORT CPET/TEST WYDOLNOŚCIOWY
=============================================
NAGŁÓWEK: Profil Zawodnika
Imię / ID: [{ct['athlete_name']} / {ct['athlete_id']}]
Data: [{ct['test_date']}] | Lokalizacja: [{ct['location']}]
Protokół: [{ct['protocol']}]
Parametry: Wiek [{ct['age']}] | Wzrost [{ct['height']}] cm | Masa [{ct['weight']}] kg

I. DIAGNOZA I PLAN (EXECUTIVE SUMMARY)
1. Główny limiter: [DO UZUPEŁNIENIA PRZEZ TRENERA]
2. Ocena poziomu: [DO UZUPEŁNIENIA PRZEZ TRENERA]

II. STREFY TRENINGOWE (MODEL 5-STREFOWY)
KOTWICE METABOLICZNE:
• VT1 (Aerobic Threshold): HR [{ct['VT1_HR']}] bpm | Speed [{ct['VT1_Speed']}] km/h | Power [{ct['VT1_Power']}] W | VO2 [{ct.get('VT1_VO2_mlmin','-')}] ml/min ({ct.get('VT1_pct_VO2max','-')}% {ct.get('VO2_determination','VO2max')})
• VT2 (Anaerobic Threshold): HR [{ct['VT2_HR']}] bpm | Speed [{ct['VT2_Speed']}] km/h | Power [{ct['VT2_Power']}] W | VO2 [{ct.get('VT2_VO2_mlmin','-')}] ml/min ({ct.get('VT2_pct_VO2max','-')}% {ct.get('VO2_determination','VO2max')})

TABELA STREF:
Z1 - Regeneracja (Active Recovery)
• HR: [{ct['Z1_HR_low']} - {ct['Z1_HR_high']}] bpm
Z2 - Baza tlenowa (Endurance)
• HR: [{ct['Z2_HR_low']} - {ct['Z2_HR_high']}] bpm
Z3 - Tempo (Aerobic Power)
• HR: [{ct['Z3_HR_low']} - {ct['Z3_HR_high']}] bpm
Z4 - Próg (Threshold)
• HR: [{ct['Z4_HR_low']} - {ct['Z4_HR_high']}] bpm
Z5 - VO2max (Maximum Power)
• HR: [{ct['Z5_HR_low']} - {ct['Z5_HR_high']}] bpm

III. KLUCZOWE WYNIKI FIZJOLOGICZNE
1. Wydolność tlenowa:
• VO2peak (rel): [{ct['VO2peak_rel']}] ml/kg/min ({ct['VO2_pct_predicted']}% predicted {ct.get('VO2_pred_method','Wasserman')})
• VO2peak (abs): [{ct['VO2peak_abs']}] L/min
• Klasyfikacja: Populacja [{ct['VO2_class_pop']}] | Sport [{ct['VO2_class_sport']}]
• Jakość testu: [{ct['test_quality']}] | RER: [{ct['RER_class']}] | VE/VCO2: [{ct['VE_VCO2_class']}]
  ↳ {ct['VE_VCO2_class_desc']}
2. Układ sercowo-naczyniowy:
• HRmax: [{ct['HRmax']}] bpm
• O2 Pulse peak: [{ct['O2pulse_peak']}] ml/beat  |  at VT1: [{ct['O2pulse_at_vt1']}]  |  at VT2: [{ct['O2pulse_at_vt2']}]
  ↳ Predicted FRIEND: [{ct['O2pulse_pred_friend']}] ({ct['O2pulse_pct_friend']}%) | NOODLE: [{ct['O2pulse_pred_noodle']}] ({ct['O2pulse_pct_noodle']}%)
  ↳ Est. SV peak: [{ct['O2pulse_est_sv']}] ml  |  Trajektoria: [{ct['O2pulse_trajectory']}] — {ct['O2pulse_traj_desc']}
  ↳ FF: [{ct['O2pulse_ff']}] | O2PRR: [{ct['O2pulse_o2prr']}] | Flagi: [{ct['O2pulse_flags']}]
• HRR (Regeneracja): 1min [{ct['HRR_60s']}] bpm | 3min [{ct['HRR_180s']}] bpm
3. Układ oddechowy — Efektywność wentylacji (E03 v2.0):
• VEpeak: [{ct['VEpeak']}] L/min
• VE/VCO2 slope:  do VT2 [{ct['VE_VCO2_slope_vt2']}] | do VT1 [{ct['VE_VCO2_slope_vt1']}] | cały test [{ct['VE_VCO2_slope_full']}]
• VE/VCO2 nadir:  [{ct['VE_VCO2_nadir']}]  |  ratio przy VT1: [{ct['VE_VCO2_at_vt1']}]  |  ratio peak: [{ct['VE_VCO2_peak']}]
• Y-intercept:    [{ct['VE_VCO2_intercept']}] L/min  |  R²: [{ct['VE_VCO2_r2']}]
• Predicted slope: [{ct['VE_VCO2_predicted']}]  |  % predicted: [{ct['VE_VCO2_pct_pred']}]%
• Klasa Arena:    [{ct['VE_VCO2_vent_class']}] — {ct['VE_VCO2_vc_desc']}
• PETCO2:         rest [{ct['PETCO2_rest']}] | VT1 [{ct['PETCO2_vt1']}] | peak [{ct['PETCO2_peak']}] mmHg
• Flagi wentylacyjne: [{ct['VE_VCO2_flags']}]
{ReportAdapter._oues_interpretation(ct)}
{ReportAdapter._breathing_pattern_report(ct)}
4. Inne:
• RER peak: [{ct['RERpeak']}]
{ReportAdapter._nirs_report(ct)}
{ReportAdapter._drift_report(ct)}
{ReportAdapter._gasex_report(ct)}
{ReportAdapter._validity_concordance_report(ct)}
{ReportAdapter._lactate_report(ct)}

{_gain_section}

{ReportAdapter._e18_crossval_text(ct)}

IV. LIMITERY (RANKING) - DO ANALIZY PRZEZ TRENERA
[MIEJSCE NA ANALIZĘ AI]

V. PROFIL METABOLICZNY
• FATmax (MFO): [{ct['FATmax_g']}] g/min ({ct.get('MFO_mgkg_min', '-')} mg/kg/min)
  ↳ HR: [{ct['FATmax_HR']}] bpm ({ct.get('FATmax_pct_hrmax', '-')}% HRmax)
  ↳ Intensywność: {ct.get('FATmax_pct_vo2peak', '-')}% VO2peak
  ↳ Strefa FATmax (≥90% MFO): HR {ct.get('FATmax_zone_hr_low', '-')}-{ct.get('FATmax_zone_hr_high', '-')} bpm
{("• Crossover Point (CHO>FAT): HR ["+str(ct['CHO_Cross_HR'])+"] bpm ("+str(ct.get('COP_pct_vo2peak','-'))+"% VO2peak, RER "+str(ct.get('COP_RER','-'))+")") if ct.get('CHO_Cross_HR') and ct['CHO_Cross_HR'] not in ('[BRAK]', None, 'None') else "• Crossover Point: " + (ct.get('COP_note','') if ct.get('COP_note') else "brak danych")}
• Substrat przy VT1: FAT {ct.get('fat_pct_at_vt1', '-')}% / CHO {ct.get('cho_pct_at_vt1', '-')}%

  ┌──────────────────────────────────────────────────────────┐
  │ SPALANIE SUBSTRATÓW W STREFACH TRENINGOWYCH              │
  ├──────────┬──────────┬──────────┬──────────┬──────────────┤
  │ Strefa   │ FAT g/h  │ CHO g/h  │ FAT%/CHO%│ kcal/h       │
  ├──────────┼──────────┼──────────┼──────────┼──────────────┤
  │ Z2 Baza  │ {str(ct.get('z2_fat_gh', '-')):>8} │ {str(ct.get('z2_cho_gh', '-')):>8} │ {str(ct.get('z2_fat_pct', '-'))+'/'+str(ct.get('z2_cho_pct', '-')):>8} │ {str(ct.get('z2_kcal_h', '-')):>12} │
  │ Z4 Próg  │ {str(ct.get('z4_fat_gh', '-')):>8} │ {str(ct.get('z4_cho_gh', '-')):>8} │ {str(ct.get('z4_fat_pct', '-'))+'/'+str(ct.get('z4_cho_pct', '-')):>8} │ {str(ct.get('z4_kcal_h', '-')):>12} │
  │ Z5 VO2max│ {str(ct.get('z5_fat_gh', '-')):>8} │ {str(ct.get('z5_cho_gh', '-')):>8} │ {str(ct.get('z5_fat_pct', '-'))+'/'+str(ct.get('z5_cho_pct', '-')):>8} │ {str(ct.get('z5_kcal_h', '-')):>12} │
  └──────────┴──────────┴──────────┴──────────┴──────────────┘
  kcal/h wg Weir (1949). W Z4/Z5 (RER≥1) FAT≈0 — dominuje glikoliza.
  CHO g/h = zapotrzebowanie na węglowodany w treningu/wyścigu.

=============================================
KONIEC RAPORTU (v1.1 T12)
"""
        return report

print("✅ Komórka 4: Report Adapter (FULL T12 TEMPLATE RESTORED) załadowana.")

# ═══════════════════════════════════════════════════════════════
# KINETICS REPORT
# ═══════════════════════════════════════════════════════════════

def render_kinetics_report(results: dict, ct: dict, df=None) -> str:
    """Generate standalone HTML kinetics report.
    
    Args:
        results: dict with E00, E01, E02, E14, E21, etc.
        ct: canon table from build_canon_table()
        df: processed DataFrame (for chart data)
    Returns:
        Complete HTML string
    """
    import html as html_mod
    import json
    import numpy as np

    esc = html_mod.escape

    # ── Helper functions ──
    def _n(v, fmt='.1f'):
        if v is None: return '-'
        try: return f'{float(v):{fmt}}'
        except: return str(v)

    def badge(text, color='#16a34a'):
        return f'<span style="display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;background:{color}15;color:{color};border:1px solid {color}40;">{text}</span>'

    def badge_class(cls):
        colors = {
            'ELITE':'#7c3aed','TRAINED':'#16a34a','ACTIVE':'#d97706','SLOW':'#ea580c',
            'MINIMAL':'#16a34a','NORMAL':'#65a30d','HIGH':'#ea580c','VERY_HIGH':'#dc2626',
            'EXCELLENT':'#16a34a','GOOD':'#16a34a','WELL_INTEGRATED':'#16a34a',
            'DELIVERY_LIMITED':'#ea580c','PERIPHERAL_LIMITED':'#dc2626','MIXED_CHECK_DATA':'#d97706',
        }
        col = colors.get(cls, '#64748b')
        labels = {
            'ELITE':'ELITARNA','TRAINED':'WYTRENOWANA','ACTIVE':'AKTYWNA','SLOW':'WOLNA',
            'MINIMAL':'MINIMALNY','NORMAL':'NORMALNY','HIGH':'WYSOKI','VERY_HIGH':'B.WYSOKI',
            'EXCELLENT':'DOSKONAŁA','GOOD':'DOBRA',
            'WELL_INTEGRATED':'ZINTEGROWANY','DELIVERY_LIMITED':'LIMIT DOSTAWY O₂',
            'PERIPHERAL_LIMITED':'LIMIT PERYFERYJNY','MIXED_CHECK_DATA':'MIESZANY',
        }
        return badge(labels.get(cls, cls), col)

    def gauge_svg(score, label, size=80, subtitle=''):
        pct = max(0, min(100, float(score) if score else 0))
        if pct >= 85: col = '#16a34a'
        elif pct >= 70: col = '#65a30d'
        elif pct >= 55: col = '#d97706'
        elif pct >= 40: col = '#ea580c'
        else: col = '#dc2626'
        if pct == 0: col = '#cbd5e1'
        r = size * 0.4; cx = size / 2; cy = size * 0.5
        circ = 2 * 3.14159 * r; dash = circ * pct / 100; gap = circ - dash
        disp = f'{int(score)}' if score else '\u2014'
        h = size + (24 if subtitle else 14)
        sub_txt = f'<text x="{cx}" y="{size+20}" text-anchor="middle" font-size="7" fill="#94a3b8">{subtitle}</text>' if subtitle else ''
        return f'<svg width="{size}" height="{h}" viewBox="0 0 {size} {h}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="6"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="{dash} {gap}" stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/><text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="700" fill="{col}">{disp}</text><text x="{cx}" y="{cy+r+10}" text-anchor="middle" font-size="8" font-weight="600" fill="#475569">{label}</text>{sub_txt}</svg>'

    def row_item(label, value, assessment='', comment=''):
        colors = {
            'EXCELLENT':'#16a34a','NORMA':'#16a34a','GOOD':'#16a34a','TRAINED':'#16a34a',
            'ELITE':'#7c3aed','NORMAL':'#16a34a','MINIMAL':'#16a34a',
            'MODERATE':'#d97706','ACTIVE':'#d97706',
            'SLOW':'#ea580c','HIGH':'#ea580c',
            'VERY_SLOW':'#dc2626','VERY_HIGH':'#dc2626','POOR':'#dc2626',
        }
        a_upper = str(assessment).upper().strip()
        col = colors.get(a_upper, '#64748b')
        a_disp = a_upper if a_upper else ''
        short = {
            'EXCELLENT':'DOSKONAŁA','TRAINED':'WYTREN.','ELITE':'ELITARNA',
            'MINIMAL':'MINIMALNY','VERY_SLOW':'B.WOLNA','VERY_HIGH':'B.WYSOKI',
            'NORMAL':'NORMA','GOOD':'DOBRA','SLOW':'WOLNA','ACTIVE':'AKTYWNA',
        }
        a_label = short.get(a_upper, a_disp)
        a_html = f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;background:{col}12;color:{col};">{a_label}</span>' if a_label else ''
        c_html = f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">{comment}</div>' if comment else ''
        return f'''<div style="display:flex;align-items:flex-start;padding:5px 0;border-bottom:1px solid #f1f5f9;">
  <div style="flex:0 0 140px;font-size:12px;color:#64748b;font-weight:500;">{label}</div>
  <div style="flex:1;font-size:13px;font-weight:600;color:#0f172a;">{value}</div>
  <div style="flex:0 0 auto;text-align:right;">{a_html}</div>
</div>{c_html}'''

    # ── Sport-Phenotype compatibility matrix ──
    MODALITY_LABELS = {
        'run': '🏃 Bieg', 'bike': '🚴 Kolarstwo', 'triathlon': '🏊 Triathlon',
        'rowing': '🚣 Wioślarstwo', 'crossfit': '🏋️ CrossFit',
        'hyrox': '💪 HYROX', 'swimming': '🏊 Pływanie', 'xc_ski': '⛷️ Biegi narciarskie',
        'soccer': '⚽ Piłka nożna', 'mma': '🥊 MMA / Sporty walki',
    }

    # compatibility: 3=ideal, 2=good, 1=partial, 0=mismatch
    SPORT_PHENOTYPE_FIT = {
        'run': {
            'ELITE_AEROBIC': (3, 'Optymalny profil dla biegacza — szybkie τ i niski SC to klucz do ekonomii biegowej'),
            'DIESEL': (3, 'Idealny profil na dystanse >10K — silnik wytrzymałościowy z efektywną ekonomią'),
            'TEMPO_RUNNER': (3, 'Profil tempowca — naturalny dla biegów 5K–HM, potencjał przez trening progowy'),
            'BURST_RECOVER': (1, 'Profil powtarzalnych wysiłków — mniej optymalny dla biegów dystansowych, lepszy dla fartleków'),
            'POWER_ENDURANCE': (1, 'Profil siłowy — wymaga pracy nad τ i SC żeby poprawić ekonomię biegową'),
            'DELIVERY_LIMITED': (2, 'Limitacja centralna — mięśnie OK, poprawa przez trening interwałowy i progowy'),
            'PERIPHERAL_LIMITED': (1, 'Limitacja obwodowa — wymaga budowy bazy aerobowej i ekonomii biegowej'),
        },
        'bike': {
            'ELITE_AEROBIC': (3, 'Zoptymalizowany profil — doskonały dla TT i kolarstwa szosowego'),
            'DIESEL': (3, 'Silnik wytrzymałościowy — idealny na długie etapy i TT'),
            'TEMPO_RUNNER': (2, 'Profil tempowca — solidny w średnich dystansach, optymalizacja SC poprawi threshold power'),
            'BURST_RECOVER': (2, 'Profil repeat-effort — przydatny w kryteriach i wyścigach drogowych z atakami'),
            'POWER_ENDURANCE': (2, 'Profil siłowy — nadaje się do sprintów i krótkich wyścigów, buduj bazę na dystans'),
            'DELIVERY_LIMITED': (2, 'Limitacja centralna — trening SST/FTP poprawi delivery'),
            'PERIPHERAL_LIMITED': (1, 'Limitacja obwodowa — bazowy trening objętościowy Z2 jest priorytetem'),
        },
        'triathlon': {
            'ELITE_AEROBIC': (3, 'Optymalny profil triatlonisty — szybka kinetyka = szybka adaptacja do zmian dyscypliny'),
            'DIESEL': (3, 'Idealny silnik Ironmana — efektywność i wytrzymałość na wielu godzinach wysiłku'),
            'TEMPO_RUNNER': (2, 'Solidny profil na Olympic/HIM — τ do optymalizacji na Ironmana'),
            'BURST_RECOVER': (1, 'Mniej optymalny — triathlon wymaga stabilnego steady-state, nie zmian tempa'),
            'POWER_ENDURANCE': (1, 'Profil siłowy wymaga znacznej pracy nad bazą aerobową dla triathlonu'),
            'DELIVERY_LIMITED': (2, 'Limitacja centralna — poprawa przez block training i swim/bike volume'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga budowy bazy aerobowej — priorytet to objętość Z2 we wszystkich dyscyplinach'),
        },
        'hyrox': {
            'ELITE_AEROBIC': (3, 'Zoptymalizowany — szybka recovery między stacjami to klucz w HYROX'),
            'DIESEL': (3, 'Idealny profil HYROX PRO — efektywna ekonomia na bieżni + recovery na stacjach'),
            'TEMPO_RUNNER': (2, 'Dobry potencjał — optymalizacja τ poprawi recovery między stacjami'),
            'BURST_RECOVER': (3, 'Naturalny profil HYROX — szybka recovery między różnymi zadaniami'),
            'POWER_ENDURANCE': (2, 'Siła OK na stacje, ale poprawa τ i SC poprawi segmenty biegowe'),
            'DELIVERY_LIMITED': (2, 'Trening interwałowy i race-pace poprawi delivery'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga pracy nad bazą — stacje siłowe OK, biegi wymagają poprawy'),
        },
        'crossfit': {
            'ELITE_AEROBIC': (2, 'Dobry motor aerobowy — ale CrossFit wymaga też mocy i recovery'),
            'DIESEL': (2, 'Silna baza aerobowa — kluczowa w długich WODach, rozbudowa power capacity'),
            'TEMPO_RUNNER': (2, 'Solidna baza — buduj repeat-effort capacity przez interwały'),
            'BURST_RECOVER': (3, 'Idealny profil CrossFit — szybka recovery między seriami i rundami'),
            'POWER_ENDURANCE': (3, 'Naturalny profil CF — dominacja Type II + zdolność do powtórzeń'),
            'DELIVERY_LIMITED': (2, 'Poprawa delivery pomoże w dłuższych WODach (Murph, Hero WODs)'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga budowy bazy — dłuższe metcony będą problemem'),
        },
        'soccer': {
            'ELITE_AEROBIC': (2, 'Silny motor — piłka wymaga też explosive recovery'),
            'DIESEL': (2, 'Dobra baza wytrzymałościowa — buduj repeat-sprint ability'),
            'TEMPO_RUNNER': (2, 'Solidny profil — optymalizacja recovery poprawi grę w 2. połowie'),
            'BURST_RECOVER': (3, 'Idealny profil piłkarza — szybka recovery po sprintach i zmianach kierunku'),
            'POWER_ENDURANCE': (2, 'Profil siłowy — dobry na sprinty, buduj bazę na 90 minut'),
            'DELIVERY_LIMITED': (2, 'Trening SSG i interwałowy poprawi wydolność meczową'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga bazowej pracy aerobowej — preseason priority'),
        },
        'mma': {
            'ELITE_AEROBIC': (2, 'Silna baza — MMA wymaga też repeat-effort capacity'),
            'DIESEL': (2, 'Dobra baza na 3-5 rund — buduj explosive recovery'),
            'TEMPO_RUNNER': (2, 'Solidna baza — poprawa recovery kluczowa między rundami'),
            'BURST_RECOVER': (3, 'Idealny profil MMA — szybka recovery po exchange\'ach i grapplingu'),
            'POWER_ENDURANCE': (3, 'Naturalny profil — explosive + repeat capacity = kluczowe w walce'),
            'DELIVERY_LIMITED': (2, 'Poprawa delivery poprawi pacing w dłuższych walkach'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga pracy bazowej — grappling wymaga wydolności'),
        },
        'rowing': {
            'ELITE_AEROBIC': (3, 'Optymalny profil wioślarza — szybka kinetyka i efektywność'),
            'DIESEL': (3, 'Idealny silnik — efektywna ekonomia na 2K i dystanse'),
            'TEMPO_RUNNER': (2, 'Solidna baza — optymalizacja τ poprawi rate-ups'),
            'BURST_RECOVER': (1, 'Wioślarstwo wymaga raczej stabilnego steady-state niż repeat-effort'),
            'POWER_ENDURANCE': (2, 'Siła OK — buduj bazę aerobową na 2K+ dystanse'),
            'DELIVERY_LIMITED': (2, 'Trening UT2/UT1 i progowy poprawi delivery'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga budowy bazy — priorytet to objętość UT2'),
        },
        'swimming': {
            'ELITE_AEROBIC': (3, 'Optymalny profil — szybka kinetyka kluczowa w pływaniu'),
            'DIESEL': (3, 'Silnik wytrzymałościowy — idealny na 400+ i OW'),
            'TEMPO_RUNNER': (2, 'Dobra baza na 200–400m — optymalizacja τ poprawi economy'),
            'BURST_RECOVER': (2, 'Przydatny w IM i relay — szybka recovery między dystansami'),
            'POWER_ENDURANCE': (2, 'Profil sprinterski — dobry na 50–100m, buduj bazę na dystans'),
            'DELIVERY_LIMITED': (2, 'Trening aerobowy + technique poprawi delivery'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga bazowej pracy aerobowej'),
        },
        'xc_ski': {
            'ELITE_AEROBIC': (3, 'Optymalny profil narciarza — szybka kinetyka na zmianach terenu'),
            'DIESEL': (3, 'Idealny silnik na dystanse — efektywność to klucz w biegach narciarskich'),
            'TEMPO_RUNNER': (2, 'Solidna baza — optymalizacja τ poprawi odpowiedź na zmiany terenu'),
            'BURST_RECOVER': (2, 'Przydatny na sprintach i zmianach tempa'),
            'POWER_ENDURANCE': (1, 'Profil siłowy — narty wymagają raczej bazowej wydolności'),
            'DELIVERY_LIMITED': (2, 'Poprawa delivery kluczowa — trening objętościowy priorytetem'),
            'PERIPHERAL_LIMITED': (1, 'Wymaga dużej bazy aerobowej — narty to sport wytrzymałościowy'),
        },
    }

    # ── Extract data ──
    e14 = results.get('E14', {})
    e21 = results.get('E21', {})
    e22 = results.get('E22', {})
    e02 = results.get('E02', {})
    e01 = results.get('E01', {})

    stages = e14.get('stages', [])
    off_ks = e14.get('off_kinetics', [])
    kin_summary = e14.get('kinetics_summary', {})
    
    phenotype = e21.get('phenotype', '?')
    pheno_info = e21.get('phenotype_info', {})
    pheno_conf = e21.get('phenotype_confidence', 0)
    limitation = e21.get('limitation', {})
    fiber = e21.get('fiber_type_proxy', {})
    kin_profile = e21.get('kinetic_profile', {})
    domain_val = e21.get('domain_validation', {})
    priorities = e21.get('training_priorities', [])
    flags = e21.get('flags', [])

    name = ct.get('name', ct.get('athlete_name', ''))
    test_date = ct.get('test_date', ct.get('date', ''))
    age = ct.get('age', '')
    weight = ct.get('weight', '')
    height = ct.get('height', '')
    sport = ct.get('sport', ct.get('modality', 'run'))
    sport_label = MODALITY_LABELS.get(sport, sport)

    vo2max = e01.get('vo2peak_rel_mlkgmin', ct.get('VO2peak_rel'))
    vo2max_abs = e01.get('vo2peak_abs_mlmin', ct.get('VO2peak_abs'))
    vt1_speed = e02.get('vt1_speed_kmh')
    vt2_speed = e02.get('vt2_speed_kmh')
    vt1_pct = kin_profile.get('vt1_pct_vo2max')
    vt2_pct = kin_profile.get('vt2_pct_vo2max')
    heavy_width = kin_profile.get('heavy_zone_width_pct')

    # Logo (inline — same SVG as PRO/LITE reports)
    _logo = "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iR8OTUllfeDVGX1dZS1JFUyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDkyOS41MiA3MTIuNDIiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIHN0cm9rZS13aWR0aDogMS41cHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMSwgLmNscy0yLCAuY2xzLTMsIC5jbHMtNCwgLmNscy01IHsKICAgICAgICBmaWxsOiBub25lOwogICAgICB9CgogICAgICAuY2xzLTEsIC5jbHMtMiwgLmNscy0zLCAuY2xzLTUgewogICAgICAgIHN0cm9rZS1taXRlcmxpbWl0OiAxMDsKICAgICAgfQoKICAgICAgLmNscy0xLCAuY2xzLTUgewogICAgICAgIHN0cm9rZTogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBzdHJva2U6IHJnYmEoMjU1LDI1NSwyNTUsMC44KTsKICAgICAgfQoKICAgICAgLmNscy0yLCAuY2xzLTMsIC5jbHMtNSB7CiAgICAgICAgc3Ryb2tlLXdpZHRoOiAzcHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMyB7CiAgICAgICAgc3Ryb2tlOiByZ2JhKDI1NSwyNTUsMjU1LDAuNSk7CiAgICAgIH0KCiAgICAgIC5jbHMtNiB7CiAgICAgICAgZmlsbDogI2ZmZmZmZjsKICAgICAgfQoKICAgICAgLmNscy03IHsKICAgICAgICBjbGlwLXBhdGg6IHVybCgjY2xpcHBhdGgtMSk7CiAgICAgIH0KCiAgICAgIC5jbHMtOCB7CiAgICAgICAgY2xpcC1wYXRoOiB1cmwoI2NsaXBwYXRoKTsKICAgICAgfQogICAgPC9zdHlsZT4KICAgIDxjbGlwUGF0aCBpZD0iY2xpcHBhdGgiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik03NTEuODYsMzE5LjkzaC0zNS4wN2wtMjYuMzktNzQuMTQtOC45MywxNi4yNi0zNC45Ni03My40NC0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41LDE4LjktNTUuMzMtNjYuNzktNDAuMjYtODIuMDhjNzIuMSw0LjIzLDEzOS4zLDM0LjMyLDE5MC44LDg1LjgyLDQxLjc1LDQxLjc1LDY5LjQzLDkzLjgzLDgwLjY3LDE1MC40N2wuMjEsMS4yNloiLz4KICAgIDwvY2xpcFBhdGg+CiAgICA8Y2xpcFBhdGggaWQ9ImNsaXBwYXRoLTEiPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTQiIGQ9Ik00MzguNzIsODIuODFsLTEwNS45NCwxNjMuOTUtNDIuMTUtNjEuMDYtMjQuOTIsODEuMjItMzkuMDEsNDguNTctNS44NS0yMy42Ny0xNC41NCwzMS40OC0xLjczLTYuMjgtMjIuMTcsMjkuNTYtMTMuMTIuMDVjNi43Ny02Ny40LDM2LjMtMTI5Ljg5LDg0Ljg0LTE3OC40Myw1MC4wMS01MC4wMiwxMTQuODQtNzkuODUsMTg0LjU5LTg1LjM5WiIvPgogICAgPC9jbGlwUGF0aD4KICA8L2RlZnM+CiAgPCEtLSBiZyByZW1vdmVkIC0tPgogIDxnPgogICAgPGc+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTc3Mi40OSw1MjAuODRoLTcuODd2NjguNjdoNS44M2MyLjIzLDAsNC4yLS44Miw1LjktMi40OCwxLjctMS42NSwyLjYtMy42OSwyLjctNi4xMnYtNTMuNTFjMC0xLjg1LS42My0zLjM4LTEuOS00LjU5LTEuMjYtMS4yMS0yLjgyLTEuODctNC42Ny0xLjk3WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTYiIGQ9Ik03NzEuNzYsNDIzLjE1aC03LjE0djY2LjYzaDcuODdjMS44NSwwLDMuNC0uNjMsNC42Ny0xLjksMS4yNi0xLjI2LDEuOS0yLjc3LDEuOS00LjUydi01Mi45M2MwLTIuMDQtLjcxLTMuNzQtMi4xMS01LjEtMS40MS0xLjM2LTMuMTMtMi4wOS01LjE4LTIuMTlaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtNiIgZD0iTTY3Mi42MywzNzYuMkg2OS4zMnYyNTguNGg3ODV2LTI1OC40aC0xODEuNjlaTTE0NS43OCw0MjEuMTFoNy4xNGM0Ljg2LDAsNy4yOSwzLjIxLDcuMjksOS42MnY2My4xM2MwLDIuODItLjczLDUuMS0yLjE5LDYuODUtMS40NiwxLjc1LTMuMTYsMi42Mi01LjEsMi42MmgtNy4xNHYtODIuMjNaTTEwNi40Miw2MjAuNDJoLTIyLjkydi0yMzAuMDVoMjIuOTJ2MjMwLjA1Wk0zNTcuOTMsNTgyLjY2YzAsMi43Mi0uNzEsNC45Ni0yLjExLDYuNzEtMS40MSwxLjc1LTMuMTMsMi42Mi01LjE4LDIuNjJzLTMuNjItLjg4LTUuMDMtMi42MmMtMS40MS0xLjc1LTIuMTEtMy45OC0yLjExLTYuNzF2LTkzLjYxYzAtMy41LjctNi4wNSwyLjExLTcuNjUsMS40MS0xLjYsMy4wOC0yLjQxLDUuMDMtMi40MSwyLjA0LDAsMy43Ny44LDUuMTgsMi40MSwxLjQxLDEuNiwyLjExLDQuMTYsMi4xMSw3LjY1djkzLjYxWk0zNTguNzMsNjIwLjQyYzMuOTMtMi4xOCw3LjY1LTUuNzMsMTEuMTYtMTAuNjRsNC43NCwxMC42NGgtMTUuOVpNNDA2LjQ4LDYyMC40MmgtOS4xOXYtMTY5Ljg2aC0yMi42bC0zLjk0LDkuMTljLTQuNzYtNy43OC0xMC44NC0xMS42Ni0xOC4yMy0xMS42NmgtMTUuNDZjLTkuODIsMC0xNy43NywzLjEzLTIzLjg0LDkuNC02LjA4LDYuMjctOS4xMSwxNC4zMS05LjExLDI0LjEzdjEwNi43M2MwLDEwLjYsMy4xMywxOS4wMyw5LjQsMjUuMywyLjk2LDIuOTYsNi4zMiw1LjIxLDEwLjA3LDYuNzhoLTQ0Ljk5YzMuMzUtMS41MSw2LjQ0LTMuNjcsOS4yNi02LjQ4LDYuMDctNi4wNyw5LjExLTEzLjM5LDkuMTEtMjEuOTR2LTM0LjI2aC0zOS4zN3YyNS41MmMwLDIuNzItLjcxLDQuODYtMi4xMSw2LjQyLTEuNDEsMS41Ni0zLjA5LDIuMzMtNS4wMywyLjMzLTQuODYtLjM5LTcuMjktMy4zLTcuMjktOC43NXYtMzguMDVoNTMuOHYtNjYuNjNjMC04LjQ2LTIuOTctMTUuNjUtOC44OS0yMS41OC01LjkzLTUuOTMtMTMuMTItOC44OS0yMS41OC04Ljg5aC0zMi4yMmMtOC4zNiwwLTE1LjUzLDIuOTctMjEuNTEsOC44OS01Ljk4LDUuOTMtOC45NywxMy4xMi04Ljk3LDIxLjU4djExMy40NGMuMTksOC43NSwzLjI4LDE2LjExLDkuMjYsMjIuMDksMi43NSwyLjc1LDUuNzcsNC44NSw5LjA2LDYuMzRoLTc2LjM1di04Ni4wMmgyMC4yN2M1LjczLDAsMTEuMTUtMS4wMiwxNi4yNi0zLjA2LDUuMS0yLjA0LDkuMjYtNS40NywxMi40Ny0xMC4yOCwzLjIxLTQuODEsNC44MS0xMS4zNSw0LjgxLTE5LjYxdi03Ny4yOGMwLTcuNjgtMS42My0xNC4wNS00Ljg4LTE5LjEtMy4yNi01LjA1LTcuNDktOC44LTEyLjY5LTExLjIzLTQuMTEtMS45Mi04LjQyLTMuMDctMTIuOTMtMy40N2gyMzcuNHYyMzAuMDVaTTI0My4xOCw1MTQuMTR2LTI1LjY2YzAtMi43Mi43LTQuOTgsMi4xMS02Ljc4LDEuNDEtMS44LDMuMTMtMi43LDUuMTgtMi43czMuNjIuOSw1LjAzLDIuN2MxLjQxLDEuOCwyLjExLDQuMDYsMi4xMSw2Ljc4djI1LjY2aC0xNC40NFpNNDQ1Ljg1LDYyMC40MnYtNjAuMDdsLjczLTEuMzEsMjAuOTQsNjEuMzhoLTIxLjY3Wk01MjIuNTIsNjIwLjQyaC05LjgxbC00MC4wNC0xMDIuMjEsMzcuNDctNjcuNjVoLTQ1LjQ5bC0xOC44MSwzOS41MXYtOTkuN2g3Ni42N3YyMzAuMDVaTTYxMS43Niw2MjAuNTdoLTY5Ljk5di0yMzAuMzdoMzkuMzd2MTk5LjMxaDMwLjYydjMxLjA2Wk02ODIuOTEsNjIwLjI4bC00LjUyLTU3LjQ1aDBsLTEwLjA2LTEyNy43Mi0xMC4wNiwxMjcuNzJoMGwtNC41Miw1Ny40NWgtMzkuMDhsMjMuNjItMjMwLjM3aDYwLjA3bDIzLjQ3LDIzMC4zN2gtMzguOTNaTTgxOC40Miw0NzguMjdjMCw2LjktMS4wNywxMi40Mi0zLjIxLDE2LjU1LTIuMTQsNC4xMy01LjA4LDcuMTItOC44Miw4Ljk3LTMuNzQsMS44NS04LjA0LDIuNzctMTIuOSwyLjc3LDQuNTcsMCw4Ljc1Ljk3LDEyLjU0LDIuOTIsMy43OSwxLjk1LDYuOCw1LjEzLDkuMDQsOS41NSwyLjIzLDQuNDIsMy4zNSwxMC4yMywzLjM1LDE3LjQydjUxLjAzYzAsMjEuNDgtMTAuMjYsMzIuNTEtMzAuNzcsMzMuMWgtNjIuNHYtMjI4LjQ4aDY0LjU5YzcuODcsMCwxNC42LDIuOCwyMC4xOSw4LjM4LDUuNTksNS41OSw4LjM4LDEyLjM3LDguMzgsMjAuMzR2NTcuNDVaIi8+CiAgICA8L2c+CiAgICA8cmVjdCBjbGFzcz0iY2xzLTMiIHg9IjY5LjMyIiB5PSIzNzYuMiIgd2lkdGg9Ijc4NSIgaGVpZ2h0PSIyNTguNCIvPgogICAgPGcgY2xhc3M9ImNscy04Ij4KICAgICAgPGc+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzcuMTQiIHgyPSI3ODYuNzIiIHkyPSIzNy4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjQzLjY2IiB4Mj0iNzg2LjcyIiB5Mj0iNDMuNjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI1MC4xNyIgeDI9Ijc4Ni43MiIgeTI9IjUwLjE3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNTYuNjkiIHgyPSI3ODYuNzIiIHkyPSI1Ni42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjYzLjIiIHgyPSI3ODYuNzIiIHkyPSI2My4yIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iNjkuNzIiIHgyPSI3ODYuNzIiIHkyPSI2OS43MiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijc2LjIzIiB4Mj0iNzg2LjcyIiB5Mj0iNzYuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSI4Mi43NSIgeDI9Ijc4Ni43MiIgeTI9IjgyLjc1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iODkuMjYiIHgyPSI3ODYuNzIiIHkyPSI4OS4yNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9Ijk1Ljc4IiB4Mj0iNzg2LjcyIiB5Mj0iOTUuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDIuMjkiIHgyPSI3ODYuNzIiIHkyPSIxMDIuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMDguODEiIHgyPSI3ODYuNzIiIHkyPSIxMDguODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMTUuMzIiIHgyPSI3ODYuNzIiIHkyPSIxMTUuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjEuODQiIHgyPSI3ODYuNzIiIHkyPSIxMjEuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMjguMzUiIHgyPSI3ODYuNzIiIHkyPSIxMjguMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxMzQuODciIHgyPSI3ODYuNzIiIHkyPSIxMzQuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDEuMzkiIHgyPSI3ODYuNzIiIHkyPSIxNDEuMzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIxNDcuOSIgeDI9Ijc4Ni43MiIgeTI9IjE0Ny45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTU0LjQyIiB4Mj0iNzg2LjcyIiB5Mj0iMTU0LjQyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTYwLjkzIiB4Mj0iNzg2LjcyIiB5Mj0iMTYwLjkzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTY3LjQ1IiB4Mj0iNzg2LjcyIiB5Mj0iMTY3LjQ1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTczLjk2IiB4Mj0iNzg2LjcyIiB5Mj0iMTczLjk2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTgwLjQ4IiB4Mj0iNzg2LjcyIiB5Mj0iMTgwLjQ4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTg2Ljk5IiB4Mj0iNzg2LjcyIiB5Mj0iMTg2Ljk5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMTkzLjUxIiB4Mj0iNzg2LjcyIiB5Mj0iMTkzLjUxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjAwLjAyIiB4Mj0iNzg2LjcyIiB5Mj0iMjAwLjAyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjA2LjU0IiB4Mj0iNzg2LjcyIiB5Mj0iMjA2LjU0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjEzLjA1IiB4Mj0iNzg2LjcyIiB5Mj0iMjEzLjA1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjE5LjU3IiB4Mj0iNzg2LjcyIiB5Mj0iMjE5LjU3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjI2LjA4IiB4Mj0iNzg2LjcyIiB5Mj0iMjI2LjA4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMjMyLjYiIHgyPSI3ODYuNzIiIHkyPSIyMzIuNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjIzOS4xMSIgeDI9Ijc4Ni43MiIgeTI9IjIzOS4xMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI0NS42MyIgeDI9Ijc4Ni43MiIgeTI9IjI0NS42MyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1Mi4xNCIgeDI9Ijc4Ni43MiIgeTI9IjI1Mi4xNCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI1OC42NiIgeDI9Ijc4Ni43MiIgeTI9IjI1OC42NiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI2NS4xNyIgeDI9Ijc4Ni43MiIgeTI9IjI2NS4xNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3MS42OSIgeDI9Ijc4Ni43MiIgeTI9IjI3MS42OSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjQ0Ny42MiIgeTE9IjI3OC4yIiB4Mj0iNzg2LjcyIiB5Mj0iMjc4LjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyODQuNzIiIHgyPSI3ODYuNzIiIHkyPSIyODQuNzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTEuMjMiIHgyPSI3ODYuNzIiIHkyPSIyOTEuMjMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIyOTcuNzUiIHgyPSI3ODYuNzIiIHkyPSIyOTcuNzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMDQuMjYiIHgyPSI3ODYuNzIiIHkyPSIzMDQuMjYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTAuNzgiIHgyPSI3ODYuNzIiIHkyPSIzMTAuNzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMTcuMjkiIHgyPSI3ODYuNzIiIHkyPSIzMTcuMjkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMjMuODEiIHgyPSI3ODYuNzIiIHkyPSIzMjMuODEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzAuMzIiIHgyPSI3ODYuNzIiIHkyPSIzMzAuMzIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzMzYuODQiIHgyPSI3ODYuNzIiIHkyPSIzMzYuODQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDMuMzUiIHgyPSI3ODYuNzIiIHkyPSIzNDMuMzUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNDkuODciIHgyPSI3ODYuNzIiIHkyPSIzNDkuODciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNTYuMzgiIHgyPSI3ODYuNzIiIHkyPSIzNTYuMzgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSI0NDcuNjIiIHkxPSIzNjIuOSIgeDI9Ijc4Ni43MiIgeTI9IjM2Mi45Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iNDQ3LjYyIiB5MT0iMzY5LjQxIiB4Mj0iNzg2LjcyIiB5Mj0iMzY5LjQxIi8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxnIGNsYXNzPSJjbHMtNyI+CiAgICAgIDxnPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjUwLjE5IiB4Mj0iNDc2LjA0IiB5Mj0iNTAuMTkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI1Ni43IiB4Mj0iNDc2LjA0IiB5Mj0iNTYuNyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjYzLjIyIiB4Mj0iNDc2LjA0IiB5Mj0iNjMuMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI2OS43NCIgeDI9IjQ3Ni4wNCIgeTI9IjY5Ljc0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iNzYuMjUiIHgyPSI0NzYuMDQiIHkyPSI3Ni4yNSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjgyLjc3IiB4Mj0iNDc2LjA0IiB5Mj0iODIuNzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSI4OS4yOCIgeDI9IjQ3Ni4wNCIgeTI9Ijg5LjI4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iOTUuOCIgeDI9IjQ3Ni4wNCIgeTI9Ijk1LjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDIuMzEiIHgyPSI0NzYuMDQiIHkyPSIxMDIuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMDguODMiIHgyPSI0NzYuMDQiIHkyPSIxMDguODMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMTUuMzQiIHgyPSI0NzYuMDQiIHkyPSIxMTUuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjEuODYiIHgyPSI0NzYuMDQiIHkyPSIxMjEuODYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMjguMzciIHgyPSI0NzYuMDQiIHkyPSIxMjguMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxMzQuODkiIHgyPSI0NzYuMDQiIHkyPSIxMzQuODkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIxNDEuNCIgeDI9IjQ3Ni4wNCIgeTI9IjE0MS40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTQ3LjkyIiB4Mj0iNDc2LjA0IiB5Mj0iMTQ3LjkyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTU0LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMTU0LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTYwLjk1IiB4Mj0iNDc2LjA0IiB5Mj0iMTYwLjk1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTY3LjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMTY3LjQ2Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTczLjk4IiB4Mj0iNDc2LjA0IiB5Mj0iMTczLjk4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTgwLjQ5IiB4Mj0iNDc2LjA0IiB5Mj0iMTgwLjQ5Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTg3LjAxIiB4Mj0iNDc2LjA0IiB5Mj0iMTg3LjAxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMTkzLjUyIiB4Mj0iNDc2LjA0IiB5Mj0iMTkzLjUyIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjAwLjA0IiB4Mj0iNDc2LjA0IiB5Mj0iMjAwLjA0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjA2LjU1IiB4Mj0iNDc2LjA0IiB5Mj0iMjA2LjU1Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjEzLjA3IiB4Mj0iNDc2LjA0IiB5Mj0iMjEzLjA3Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjE5LjU4IiB4Mj0iNDc2LjA0IiB5Mj0iMjE5LjU4Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMjI2LjEiIHgyPSI0NzYuMDQiIHkyPSIyMjYuMSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzMi42MSIgeDI9IjQ3Ni4wNCIgeTI9IjIzMi42MSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjIzOS4xMyIgeDI9IjQ3Ni4wNCIgeTI9IjIzOS4xMyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI0NS42NCIgeDI9IjQ3Ni4wNCIgeTI9IjI0NS42NCIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1Mi4xNiIgeDI9IjQ3Ni4wNCIgeTI9IjI1Mi4xNiIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI1OC42NyIgeDI9IjQ3Ni4wNCIgeTI9IjI1OC42NyIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI2NS4xOSIgeDI9IjQ3Ni4wNCIgeTI9IjI2NS4xOSIvPgogICAgICAgIDxsaW5lIGNsYXNzPSJjbHMtMSIgeDE9IjEzNi45NCIgeTE9IjI3MS43IiB4Mj0iNDc2LjA0IiB5Mj0iMjcxLjciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyNzguMjIiIHgyPSI0NzYuMDQiIHkyPSIyNzguMjIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyODQuNzMiIHgyPSI0NzYuMDQiIHkyPSIyODQuNzMiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTEuMjUiIHgyPSI0NzYuMDQiIHkyPSIyOTEuMjUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIyOTcuNzYiIHgyPSI0NzYuMDQiIHkyPSIyOTcuNzYiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMDQuMjgiIHgyPSI0NzYuMDQiIHkyPSIzMDQuMjgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTAuNzkiIHgyPSI0NzYuMDQiIHkyPSIzMTAuNzkiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMTcuMzEiIHgyPSI0NzYuMDQiIHkyPSIzMTcuMzEiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMjMuODIiIHgyPSI0NzYuMDQiIHkyPSIzMjMuODIiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzAuMzQiIHgyPSI0NzYuMDQiIHkyPSIzMzAuMzQiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzMzYuODUiIHgyPSI0NzYuMDQiIHkyPSIzMzYuODUiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDMuMzciIHgyPSI0NzYuMDQiIHkyPSIzNDMuMzciLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNDkuODgiIHgyPSI0NzYuMDQiIHkyPSIzNDkuODgiLz4KICAgICAgICA8bGluZSBjbGFzcz0iY2xzLTEiIHgxPSIxMzYuOTQiIHkxPSIzNTYuNCIgeDI9IjQ3Ni4wNCIgeTI9IjM1Ni40Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzYyLjkxIiB4Mj0iNDc2LjA0IiB5Mj0iMzYyLjkxIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzY5LjQzIiB4Mj0iNDc2LjA0IiB5Mj0iMzY5LjQzIi8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzc1Ljk0IiB4Mj0iNDc2LjA0IiB5Mj0iMzc1Ljk0Ii8+CiAgICAgICAgPGxpbmUgY2xhc3M9ImNscy0xIiB4MT0iMTM2Ljk0IiB5MT0iMzgyLjQ2IiB4Mj0iNDc2LjA0IiB5Mj0iMzgyLjQ2Ii8+CiAgICAgIDwvZz4KICAgIDwvZz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTMiIHBvaW50cz0iMTA2LjQyIDM0Ni44NiAyNDEuMzQgMzQ2LjM4IDI5MC42MyAxODUuNzEgMzMyLjc5IDI0Ni43NyA0NjIuNDEgNDYuMTUgNjI0LjUgMzc2LjYzIDY4My42MiAzMjEuODcgNjk3LjE5IDM0My42OCA4MTguNDIgMzQzLjY4Ii8+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik04MTguNDIsMzE5Ljk0aC0xMDkuMzlsLTYyLjUxLTEzMS4zMi0yNS42OCw4OS4xNi0yOS41Ni02NS40Mi0xNS41MSwxOC45LTkzLjUyLTExMi45LTU5LjEyLDExNC4zNi03LjI3LTE5LjM4cy01NS44NiwxMDcuMS02OC41OSwxMzIuMjljLTEzLjM4LTIyLjg1LTYwLjMxLTEwNS4xNS02MC4zMS0xMDUuMTVsLTc3LjA1LDk1Ljk0LTUuMzMtMTkuMzgtMjkuMDcsMzguNzdoLTY5LjEiLz4KICAgIDxwb2x5bGluZSBjbGFzcz0iY2xzLTUiIHBvaW50cz0iMTA2LjQyIDM2NSAxODcuMDQgMzY1IDIyMC44NSAyOTEuODMgMjMyIDMzNi45IDMwMC44MSAyMjYuNDEgMzI2Ljk3IDI5NC4yNSA0NDIuMyAxNDQuNTIgNTE2LjQ0IDI5OS41OCA1NTAuODUgMjM3LjA3IDYzNi4xMyAzNDQuNjUgNjkwLjQgMjQ1LjggNzIwLjkzIDMzMS41NyA4MTguNDIgMzMxLjU3Ii8+CiAgPC9nPgo8L3N2Zz4="

    # ═══════════════════════════════════════════════════════
    # BUILD HTML
    # ═══════════════════════════════════════════════════════

    h = f'''<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kinetyka VO₂ — {esc(str(name))}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f8fafc;color:#0f172a;line-height:1.5;}}
.wrap{{max-width:920px;margin:0 auto;padding:20px;}}
.card{{background:white;border-radius:12px;border:1px solid #e2e8f0;padding:18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);}}
.section-title{{font-size:14px;font-weight:700;color:#0f172a;padding-bottom:6px;margin-bottom:10px;border-bottom:2px solid #e2e8f0;}}
.sub-header{{margin:10px 0 6px;font-size:11px;font-weight:700;color:#475569;border-top:1px solid #e2e8f0;padding-top:6px;text-transform:uppercase;letter-spacing:0.5px;}}
.domain-card{{border-radius:10px;border:1px solid #e2e8f0;padding:14px;flex:1;min-width:200px;}}
.domain-mod{{border-left:4px solid #3b82f6;}}
.domain-hvy{{border-left:4px solid #d97706;}}
.domain-sev{{border-left:4px solid #dc2626;}}
.tau-bar{{height:8px;border-radius:4px;margin:4px 0;}}
.stage-table{{width:100%;border-collapse:collapse;font-size:12px;}}
.stage-table th{{text-align:left;padding:6px 8px;color:#64748b;font-weight:600;font-size:11px;border-bottom:2px solid #e2e8f0;}}
.stage-table td{{padding:6px 8px;border-bottom:1px solid #f1f5f9;}}
.stage-table tr:hover{{background:#f8fafc;}}
.flag-item{{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;margin:2px;}}
.priority-row{{display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #f1f5f9;}}
.priority-num{{flex:0 0 28px;height:28px;border-radius:50%;background:#0f172a;color:white;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;}}
@media print{{body{{background:white;}} .wrap{{max-width:100%;padding:10px;}} .card{{break-inside:avoid;box-shadow:none;page-break-inside:avoid;}}}}
</style></head><body><div class="wrap">'''

    # ═══════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════
    h += f'''<div style="padding:20px 24px;background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);border-radius:16px;color:white;margin-bottom:14px;position:relative;overflow:hidden;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="flex:1;min-width:0;">
      <div style="font-size:11px;color:#94a3b8;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">⚡ Raport kinetyki VO₂ &amp; Slow Component</div>
      <div style="font-size:28px;font-weight:800;margin-bottom:4px;letter-spacing:-0.5px;">{esc(str(name))}</div>
      <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">Test kinetyczny: {test_date} | Wiek: {age} | {weight} kg | {height} cm</div>
      <div style="font-size:12px;color:#93c5fd;font-weight:600;margin-bottom:4px;">Dyscyplina: {sport_label}</div>
      <div style="display:flex;gap:14px;font-size:11px;">
        <a href="https://www.peaklab.com.pl" target="_blank" style="color:#93c5fd;text-decoration:none;">🌐 www.peaklab.com.pl</a>
        <a href="https://www.instagram.com/peak_lab_" target="_blank" style="color:#93c5fd;text-decoration:none;">📷 @peak_lab_</a>
      </div>
    </div>
    <div style="flex-shrink:0;margin-left:16px;">
      <img src="data:image/svg+xml;base64,{_logo}" style="height:90px;opacity:0.9;" alt="PeakLab">
    </div>
  </div>
</div>'''

    # ═══════════════════════════════════════════════════════
    # PHENOTYPE HERO CARD
    # ═══════════════════════════════════════════════════════
    pheno_icon = pheno_info.get('icon', '🔬')
    pheno_label = pheno_info.get('label_pl', phenotype)
    pheno_desc = pheno_info.get('desc', '')
    pheno_arch = pheno_info.get('archetype', '')
    lim_primary = limitation.get('primary', '?')
    lim_evidence = limitation.get('evidence', [])
    conf_pct = int(pheno_conf * 100) if pheno_conf < 1 else int(pheno_conf)

    fiber_profile = fiber.get('profile', '?')
    fiber_type1 = fiber.get('estimated_type_I_pct', '?')
    
    h += f'''<div class="card">
  <div class="section-title">Fenotyp kinetyczny</div>
  <div style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;">
    <div style="flex:0 0 auto;text-align:center;">
      <div style="font-size:52px;margin-bottom:4px;">{pheno_icon}</div>
      <div style="font-size:20px;font-weight:800;color:#0f172a;">{esc(str(pheno_label))}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:2px;">Pewność: {conf_pct}%</div>
    </div>
    <div style="flex:1;min-width:250px;">
      <div style="font-size:13px;color:#334155;margin-bottom:8px;">{esc(str(pheno_desc))}</div>
      <div style="font-size:11px;color:#64748b;margin-bottom:10px;">Archetyp: {esc(str(pheno_arch))}</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <div style="font-size:11px;color:#64748b;">Limitacja: {badge_class(lim_primary)}</div>
        <div style="font-size:11px;color:#64748b;">Włókna: {badge(str(fiber_type1) + " Typ I", "#2563eb")}</div>
      </div>
    </div>
  </div>
</div>'''

    # ═══════════════════════════════════════════════════════
    # SPORT-PHENOTYPE COMPATIBILITY
    # ═══════════════════════════════════════════════════════
    sport_fits = SPORT_PHENOTYPE_FIT.get(sport, {})
    fit_score, fit_comment = sport_fits.get(phenotype, (0, 'Brak danych o kompatybilności'))
    fit_colors = {3: '#16a34a', 2: '#d97706', 1: '#ea580c', 0: '#64748b'}
    fit_labels = {3: 'OPTYMALNY', 2: 'DOBRY', 1: 'CZĘŚCIOWY', 0: '—'}
    fit_icons = {3: '✅', 2: '🔶', 1: '⚠️', 0: '❔'}
    fc = fit_colors.get(fit_score, '#64748b')

    h += f'''<div class="card" style="border-left:4px solid {fc};">
  <div style="display:flex;gap:16px;align-items:flex-start;">
    <div style="font-size:28px;">{fit_icons.get(fit_score, '❔')}</div>
    <div style="flex:1;">
      <div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">
        Kompatybilność fenotypu z dyscypliną
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:6px;">
        <span style="font-size:16px;font-weight:800;color:#0f172a;">{pheno_icon} {esc(str(pheno_label))}</span>
        <span style="font-size:14px;color:#64748b;">×</span>
        <span style="font-size:16px;font-weight:800;color:#0f172a;">{sport_label}</span>
        <span style="font-size:14px;color:#64748b;">→</span>
        <span style="display:inline-block;padding:3px 12px;border-radius:6px;font-size:12px;font-weight:700;background:{fc}15;color:{fc};border:1px solid {fc}40;">{fit_labels.get(fit_score, '—')}</span>
      </div>
      <div style="font-size:13px;color:#334155;">{esc(fit_comment)}</div>
    </div>
  </div>
</div>'''

    # ═══════════════════════════════════════════════════════
    # BASELINE CONTEXT (mini)
    # ═══════════════════════════════════════════════════════
    h += f'''<div class="card" style="padding:12px 18px;">
  <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center;">
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Baseline (test do odmowy)</div>
    <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:12px;">
      <div><span style="color:#64748b;">VO₂max:</span> <b>{_n(vo2max)}</b> ml/kg/min</div>
      <div><span style="color:#64748b;">VT1:</span> <b>{_n(vt1_speed)}</b> km/h ({_n(vt1_pct, ".0f")}%)</div>
      <div><span style="color:#64748b;">VT2:</span> <b>{_n(vt2_speed)}</b> km/h ({_n(vt2_pct, ".0f")}%)</div>
      <div><span style="color:#64748b;">Heavy zone:</span> <b>{_n(heavy_width, ".0f")}%</b></div>
    </div>
  </div>
</div>'''

    # ═══════════════════════════════════════════════════════
    # STAGE TABLE
    # ═══════════════════════════════════════════════════════
    h += '<div class="card"><div class="section-title">Protokół kinetyczny — stadia CWR</div>'
    h += '<table class="stage-table"><thead><tr>'
    h += '<th>Stadium</th><th>Prędkość</th><th>Domena</th><th>τ on [s]</th><th>SC [%]</th><th>VO₂ śr.</th><th>%max</th><th>HR śr.</th><th>RER</th><th>Czas</th>'
    h += '</tr></thead><tbody>'

    domain_colors = {'MODERATE':'#3b82f6','HEAVY':'#d97706','SEVERE':'#dc2626','VERY_SEVERE':'#7c3aed'}
    domain_short = {'MODERATE':'MOD','HEAVY':'HVY','SEVERE':'SEV','VERY_SEVERE':'V.SEV'}
    
    # Domain validation
    dv_stages = domain_val.get('stages', []) if isinstance(domain_val, dict) else []
    dv_map = {s.get('stage_num'): s for s in dv_stages} if dv_stages else {}

    for s in stages:
        snum = s.get('stage_num', 0)
        dom = s.get('domain', '?')
        speed = s.get('speed_kmh', 0)
        tau = s.get('tau_on_s')
        tau_cls = s.get('tau_class', '')
        sc_pct = s.get('sc_pct')
        sc_cls = s.get('sc_class', '')
        vo2_mean = s.get('vo2_mean_mlmin', s.get('vo2kg_mean', 0))
        vo2kg = s.get('vo2kg_mean', 0)
        pct_max = s.get('pct_vo2max', 0)
        hr = s.get('hr_mean', 0)
        rer = s.get('rer_mean', 0)
        dur = s.get('duration_s', 0)

        # Domain match check
        dv = dv_map.get(snum, {})
        dom_match = dv.get('domain_match', True) if dv else True
        dom_icon = '✓' if dom_match else '⚠'

        dcol = domain_colors.get(dom, '#64748b')
        dshort = domain_short.get(dom, dom[:3])
        
        tau_str = f'{tau:.0f}' if tau else '-'
        sc_str = f'{sc_pct:.1f}' if sc_pct is not None else '-'
        dur_str = f'{dur/60:.1f}min' if dur >= 120 else f'{dur:.0f}s'

        h += f'''<tr>
  <td><b>S{snum}</b></td>
  <td>{speed} km/h</td>
  <td><span style="color:{dcol};font-weight:700;">{dshort}</span> {dom_icon}</td>
  <td><b>{tau_str}</b> <span style="font-size:10px;color:{domain_colors.get(tau_cls, '#64748b')};">{tau_cls}</span></td>
  <td>{sc_str} <span style="font-size:10px;color:#94a3b8;">{sc_cls}</span></td>
  <td>{_n(vo2kg, ".0f")} ml/kg</td>
  <td>{_n(pct_max, ".0f")}%</td>
  <td>{hr}</td>
  <td>{_n(rer, ".2f")}</td>
  <td>{dur_str}</td>
</tr>'''

    h += '</tbody></table></div>'

    # ═══════════════════════════════════════════════════════
    # KINETICS ON — Domain cards
    # ═══════════════════════════════════════════════════════
    h += '<div class="card"><div class="section-title">Kinetyka ON — adaptacja VO₂ do wysiłku</div>'
    h += '<div style="display:flex;gap:12px;flex-wrap:wrap;">'

    # tau score mapping (0-100)
    def tau_score(tau_val, domain):
        if tau_val is None: return 0
        if domain == 'MODERATE':
            thresholds = [(15, 100), (25, 75), (40, 50), (60, 25)]
        elif domain == 'HEAVY':
            thresholds = [(20, 100), (35, 75), (50, 50), (70, 25)]
        else:
            return 30  # severe always slower
        for t, s in thresholds:
            if tau_val <= t: return s
        return 10

    for s in stages:
        dom = s.get('domain', '?')
        if dom not in ('MODERATE', 'HEAVY', 'SEVERE'): continue

        tau = s.get('tau_on_s')
        tau_cls = s.get('tau_class', '')
        sc_pct = s.get('sc_pct', 0)
        sc_cls = s.get('sc_class', '')
        pct_max = s.get('pct_vo2max', 0)
        speed = s.get('speed_kmh', 0)
        
        dom_css = {'MODERATE': 'domain-mod', 'HEAVY': 'domain-hvy', 'SEVERE': 'domain-sev'}
        dom_label = {'MODERATE': 'Moderate', 'HEAVY': 'Heavy', 'SEVERE': 'Severe'}
        dom_desc = {
            'MODERATE': f'<VT1 — {speed} km/h',
            'HEAVY': f'VT1↔VT2 — {speed} km/h',
            'SEVERE': f'>VT2 — {speed} km/h',
        }

        ts = tau_score(tau, dom)
        tau_bar_col = '#16a34a' if ts >= 75 else ('#d97706' if ts >= 50 else '#dc2626')
        tau_bar_w = max(5, min(100, ts))

        h += f'''<div class="domain-card {dom_css.get(dom, '')}">
  <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:2px;">{dom_label.get(dom, dom)}</div>
  <div style="font-size:10px;color:#94a3b8;margin-bottom:8px;">{dom_desc.get(dom, '')}</div>
  <div style="font-size:28px;font-weight:800;color:#0f172a;">{_n(tau, ".0f")}s</div>
  <div style="font-size:11px;margin-bottom:4px;">{badge_class(tau_cls)}</div>
  <div class="tau-bar" style="background:#e5e7eb;">
    <div style="width:{tau_bar_w}%;height:100%;border-radius:4px;background:{tau_bar_col};"></div>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#64748b;">
    SC: <b>{_n(sc_pct, ".1f")}%</b> ({sc_cls}) · {_n(pct_max, ".0f")}% VO₂max
  </div>
</div>'''

    h += '</div>'

    # tau ratio
    tau_mod = kin_profile.get('tau_moderate')
    tau_hvy = kin_profile.get('tau_heavy')
    tau_ratio = kin_profile.get('tau_ratio_heavy_mod')
    ratio_interp = kin_profile.get('tau_ratio_interpretation', '')
    if tau_ratio:
        h += f'<div style="margin-top:10px;font-size:12px;color:#334155;">τ ratio heavy/moderate = <b>{_n(tau_ratio, ".2f")}</b> (norma: 1.0–1.5) — {ratio_interp}</div>'
    h += '</div>'

    # ═══════════════════════════════════════════════════════
    # SLOW COMPONENT
    # ═══════════════════════════════════════════════════════
    sc_hvy = kin_profile.get('sc_heavy_pct')
    sc_hvy_cls = kin_profile.get('sc_heavy_class', '')
    sc_sev = kin_profile.get('sc_severe_pct')
    sc_sev_cls = kin_profile.get('sc_severe_class', '')
    fiber_str = fiber.get('estimated_type_I_pct', '?')

    h += f'''<div class="card">
  <div class="section-title">Slow Component — ekonomia mięśniowa</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:250px;">
      <div style="font-size:12px;color:#64748b;margin-bottom:4px;">SC w domenie Heavy (powyżej VT1)</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px;">
        <span style="font-size:32px;font-weight:800;color:#0f172a;">{_n(sc_hvy, ".1f")}%</span>
        {badge_class(sc_hvy_cls)}
      </div>
      <div style="font-size:12px;color:#64748b;margin-bottom:4px;">SC w domenie Severe (powyżej VT2)</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:12px;">
        <span style="font-size:24px;font-weight:700;color:#0f172a;">{_n(sc_sev, ".1f")}%</span>
        {badge_class(sc_sev_cls)}
      </div>
      <div style="font-size:11px;color:#94a3b8;">Norma: 3–8% (heavy), 8–15% (severe). <3% = znakomita ekonomia</div>
    </div>
    <div style="flex:1;min-width:200px;background:#f8fafc;border-radius:8px;padding:14px;">
      <div style="font-size:12px;font-weight:700;color:#475569;margin-bottom:6px;">Proxy profilu włókien mięśniowych</div>
      <div style="font-size:11px;color:#334155;margin-bottom:4px;">Estymacja na podstawie SC (Barstow 1996)</div>
      <div style="font-size:18px;font-weight:700;color:#2563eb;margin-bottom:4px;">{fiber_str} Typ I</div>
      <div style="font-size:10px;color:#94a3b8;">Profil: {fiber_profile} · Pewność: {fiber.get("confidence", "LOW")}</div>
      <div style="font-size:10px;color:#94a3b8;margin-top:4px;font-style:italic;">Pośrednia estymacja — nie zastępuje biopsji mięśniowej</div>
    </div>
  </div>
</div>'''

    # ═══════════════════════════════════════════════════════
    # OFF-KINETICS — Recovery
    # ═══════════════════════════════════════════════════════
    valid_off = [o for o in off_ks if o.get('status') == 'OK']
    if valid_off:
        h += '<div class="card"><div class="section-title">Off-kinetics — recovery między stadiami</div>'
        for ok in valid_off:
            trans = ok.get('transition', '?')
            tau_off = ok.get('tau_off_s')
            tau_r2 = ok.get('tau_off_r2')
            t_half = ok.get('t_half_s') or ok.get('t_half_from_tau_s')
            rec_cls = ok.get('recovery_class', '')
            pct_60 = ok.get('pct_recovered_60s')
            pct_120 = ok.get('pct_recovered_120s')

            h += f'''<div style="display:flex;gap:16px;align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9;">
  <div style="flex:0 0 100px;font-size:12px;font-weight:700;color:#0f172a;">{trans}</div>
  <div style="flex:1;font-size:12px;color:#334155;">
    τ_off={_n(tau_off, ".0f")}s (R²={_n(tau_r2, ".2f")}) · T½={_n(t_half, ".0f")}s · Rec 60s: {_n(pct_60, ".0f")}% / 120s: {_n(pct_120, ".0f")}%
  </div>
  <div>{badge_class(rec_cls)}</div>
</div>'''
        h += '</div>'

    # ═══════════════════════════════════════════════════════
    # E22: CROSS-CORRELATION SECTIONS
    # ═══════════════════════════════════════════════════════

    if e22 and e22.get('status') == 'OK':
        stage_cx = e22.get('stage_cross', [])
        composites = e22.get('composites', {})
        hrr_data = e22.get('hrr', {})
        nirs_data = e22.get('nirs', {})

        # ── Drift Analysis Card ──
        if stage_cx:
            h += '<div class="card"><div class="section-title">Analiza driftów — Triple Drift Pattern</div>'
            h += '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr style="background:#f8fafc;">'
            h += '<th style="padding:6px;text-align:left;">Stadium</th>'
            h += '<th style="padding:6px;text-align:center;">HR drift</th>'
            h += '<th style="padding:6px;text-align:center;">VO₂ drift</th>'
            h += '<th style="padding:6px;text-align:center;">O₂P drift</th>'
            h += '<th style="padding:6px;text-align:center;">O₂P Stability</th>'
            h += '<th style="padding:6px;text-align:center;">Pattern</th>'
            h += '</tr></thead><tbody>'
            pattern_colors = {
                'STABLE': '#16a34a', 'METABOLIC_SC': '#d97706',
                'MUSCLE_RECRUITMENT': '#ea580c', 'CARDIOVASCULAR': '#dc2626', 'MIXED': '#64748b'
            }
            for sc in stage_cx:
                pat = sc.get('drift_pattern', 'MIXED')
                pcol = pattern_colors.get(pat, '#64748b')
                dom = sc.get('domain', '')
                dcol = {'MODERATE': '#3b82f6', 'HEAVY': '#d97706', 'SEVERE': '#dc2626'}.get(dom, '#64748b')
                h += f'<tr style="border-top:1px solid #f1f5f9;">'
                h += f'<td style="padding:6px;"><span style="color:{dcol};font-weight:700;">S{sc.get("stage_num")} {dom}</span><br><span style="font-size:10px;color:#94a3b8;">{sc.get("speed_kmh","")} km/h</span></td>'
                h += f'<td style="padding:6px;text-align:center;font-weight:600;">{sc.get("hr_drift_bpm","-"):+.1f} bpm</td>'
                h += f'<td style="padding:6px;text-align:center;font-weight:600;">{sc.get("vo2_drift_mlmin","-"):+.0f} ml/min</td>'
                h += f'<td style="padding:6px;text-align:center;font-weight:600;">{sc.get("o2p_drift_ml","-"):+.2f} ml</td>'
                stab = sc.get('o2p_stability')
                stab_str = f'{stab:.3f}' if stab else '-'
                stab_cls = sc.get('o2p_stability_class', '')
                stab_col = '#16a34a' if stab_cls == 'STABLE' else '#d97706' if stab_cls == 'MILD_DRIFT' else '#dc2626' if stab_cls == 'CARDIAC_DRIFT' else '#64748b'
                h += f'<td style="padding:6px;text-align:center;"><span style="color:{stab_col};font-weight:600;">{stab_str}</span></td>'
                h += f'<td style="padding:6px;text-align:center;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;background:{pcol}15;color:{pcol};border:1px solid {pcol}40;">{pat}</span></td>'
                h += '</tr>'
            h += '</tbody></table>'

            # Interpretation note
            dominant_pattern = max(set(sc.get('drift_pattern','MIXED') for sc in stage_cx), key=lambda p: sum(1 for s in stage_cx if s.get('drift_pattern') == p))
            if dominant_pattern == 'METABOLIC_SC':
                h += '<div style="margin-top:8px;padding:8px 12px;background:#fffbeb;border-radius:6px;font-size:11px;color:#92400e;">💡 Dominujący wzorzec: <b>Metaboliczny SC</b> — HR i VO₂ rosną, ale O₂ pulse stabilny. To oznacza rekrutację Type II fiber bez cardiac drift. Cardiac output jest stabilny.</div>'
            elif dominant_pattern == 'MUSCLE_RECRUITMENT':
                h += '<div style="margin-top:8px;padding:8px 12px;background:#fef3c7;border-radius:6px;font-size:11px;color:#92400e;">💡 Dominujący wzorzec: <b>Rekrutacja mięśniowa</b> — VO₂ driftuje znacznie bardziej niż HR. O₂ pulse rośnie = SV rośnie. Limitacja obwodowa (mięśniowa), nie centralna.</div>'
            elif dominant_pattern == 'CARDIOVASCULAR':
                h += '<div style="margin-top:8px;padding:8px 12px;background:#fee2e2;border-radius:6px;font-size:11px;color:#991b1b;">⚠️ Dominujący wzorzec: <b>Cardiovascular drift</b> — HR↑, VO₂↑, O₂ pulse↓. Możliwe przyczyny: dehydracja, termoregulacja, niska wydolność sercowa.</div>'
            h += '</div>'

        # ── Metabolic Efficiency + Economy Durability Card ──
        mei_avg = composites.get('mei_avg')
        if mei_avg is not None or stage_cx:
            h += '<div class="card"><div class="section-title">Efektywność metaboliczna & ekonomia</div>'
            h += '<div style="display:flex;gap:16px;flex-wrap:wrap;">'

            # MEI gauge
            mei_cls = composites.get('mei_class', '?')
            mei_col = '#16a34a' if mei_cls == 'EXCELLENT' else '#d97706' if mei_cls == 'GOOD' else '#dc2626'
            h += f'''<div style="flex:1;min-width:200px;text-align:center;padding:16px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;letter-spacing:0.5px;">Metabolic Efficiency Index</div>
  <div style="font-size:32px;font-weight:800;color:{mei_col};">{_n(mei_avg, ".2f")}</div>
  <div style="font-size:11px;color:{mei_col};font-weight:600;">{mei_cls}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:4px;">&lt;3.0 excellent · 3–6 good · &gt;6 flag</div>
</div>'''

            # Economy durability per stage
            h += '<div style="flex:2;min-width:300px;">'
            h += '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr style="background:#f8fafc;">'
            h += '<th style="padding:5px;text-align:left;">Stadium</th>'
            h += '<th style="padding:5px;text-align:center;">RER drift</th>'
            h += '<th style="padding:5px;text-align:center;">MEI</th>'
            h += '<th style="padding:5px;text-align:center;">Economy loss</th>'
            h += '<th style="padding:5px;text-align:center;">VE/VCO₂</th>'
            h += '</tr></thead><tbody>'
            for sc in stage_cx:
                dom = sc.get('domain', '')
                dcol = {'MODERATE': '#3b82f6', 'HEAVY': '#d97706', 'SEVERE': '#dc2626'}.get(dom, '#64748b')
                eco_loss = sc.get('economy_loss_pct')
                eco_col = '#16a34a' if eco_loss is not None and eco_loss < 5 else '#d97706' if eco_loss is not None and eco_loss < 8 else '#dc2626'
                h += f'<tr style="border-top:1px solid #f1f5f9;">'
                h += f'<td style="padding:5px;color:{dcol};font-weight:600;">S{sc.get("stage_num")} {dom}</td>'
                rd = sc.get('rer_drift')
                h += f'<td style="padding:5px;text-align:center;">{rd:+.3f}</td>' if rd is not None else '<td style="padding:5px;text-align:center;">-</td>'
                mi = sc.get('metabolic_efficiency_idx')
                h += f'<td style="padding:5px;text-align:center;">{_n(mi, ".2f")}</td>'
                h += f'<td style="padding:5px;text-align:center;color:{eco_col};font-weight:600;">{_n(eco_loss, ".1f")}%</td>' if eco_loss is not None else '<td style="padding:5px;text-align:center;">-</td>'
                vv = sc.get('ve_vco2_ratio')
                h += f'<td style="padding:5px;text-align:center;">{_n(vv, ".1f")}</td>'
                h += '</tr>'
            h += '</tbody></table></div>'
            h += '</div></div>'

        # ── Recovery Integration Card ──
        if hrr_data.get('hrr_1min_bpm') is not None:
            h += '<div class="card"><div class="section-title">Recovery Integration — HRR ↔ τ off-kinetics</div>'
            h += '<div style="display:flex;gap:16px;flex-wrap:wrap;">'

            # HRR box
            hrr1 = hrr_data.get('hrr_1min_bpm', 0)
            hrr3 = hrr_data.get('hrr_3min_bpm')
            hrr_cls = hrr_data.get('hrr_class', '?')
            hrr_col = '#16a34a' if hrr_cls in ('EXCELLENT','GOOD') else '#d97706' if hrr_cls == 'NORMAL' else '#dc2626'
            h += f'''<div style="flex:1;text-align:center;padding:14px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">HRR 1 min</div>
  <div style="font-size:28px;font-weight:800;color:{hrr_col};">{hrr1:.0f} <span style="font-size:14px;">bpm</span></div>
  <div style="font-size:11px;color:{hrr_col};">{hrr_cls}</div>'''
            if hrr3: h += f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">HRR 3min: {hrr3:.0f} bpm</div>'
            h += '</div>'

            # τ_off box
            tau_off = hrr_data.get('tau_off_mean_s')
            if tau_off:
                h += f'''<div style="flex:1;text-align:center;padding:14px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">τ off-kinetics (mean)</div>
  <div style="font-size:28px;font-weight:800;color:#0f172a;">{tau_off:.0f} <span style="font-size:14px;">s</span></div>
  <div style="font-size:10px;color:#94a3b8;">Metabolic recovery speed</div>
</div>'''

            # Dissociation box
            rd = hrr_data.get('recovery_dissociation')
            rtype = hrr_data.get('recovery_type', '?')
            rtype_pl = hrr_data.get('recovery_type_pl', '')
            rd_col = '#16a34a' if rtype in ('BALANCED_FAST','BALANCED') else '#d97706' if rtype in ('AUTONOMIC_FAST','METABOLIC_FAST') else '#dc2626'
            if rd is not None:
                h += f'''<div style="flex:1;text-align:center;padding:14px;background:#f8fafc;border-radius:8px;border:2px solid {rd_col}30;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Recovery Dissociation</div>
  <div style="font-size:28px;font-weight:800;color:{rd_col};">{rd:.2f}</div>
  <div style="font-size:11px;color:{rd_col};font-weight:600;">{rtype}</div>
  <div style="font-size:10px;color:#64748b;margin-top:2px;">{rtype_pl}</div>
</div>'''

            h += '</div></div>'

        # ── Breathing Pattern Card ──
        if stage_cx and any(sc.get('bf_mean') for sc in stage_cx):
            h += '<div class="card"><div class="section-title">Wzorzec oddechowy per domain</div>'
            h += '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr style="background:#f8fafc;">'
            h += '<th style="padding:5px;text-align:left;">Stadium</th>'
            h += '<th style="padding:5px;text-align:center;">BF [/min]</th>'
            h += '<th style="padding:5px;text-align:center;">VT [L]</th>'
            h += '<th style="padding:5px;text-align:center;">VE [L/min]</th>'
            h += '<th style="padding:5px;text-align:center;">BF/VT ratio</th>'
            h += '<th style="padding:5px;text-align:center;">VT drift</th>'
            h += '</tr></thead><tbody>'
            prev_vt = None
            for sc in stage_cx:
                dom = sc.get('domain', '')
                dcol = {'MODERATE': '#3b82f6', 'HEAVY': '#d97706', 'SEVERE': '#dc2626'}.get(dom, '#64748b')
                vt_now = sc.get('vt_mean_L')
                vt_plateau = ''
                if prev_vt and vt_now and abs(vt_now - prev_vt) < 0.1:
                    vt_plateau = ' ⚠️ plateau'
                prev_vt = vt_now
                h += f'<tr style="border-top:1px solid #f1f5f9;">'
                h += f'<td style="padding:5px;color:{dcol};font-weight:600;">S{sc.get("stage_num")} {dom}</td>'
                h += f'<td style="padding:5px;text-align:center;">{_n(sc.get("bf_mean"), ".1f")}</td>'
                h += f'<td style="padding:5px;text-align:center;">{_n(vt_now, ".2f")}{vt_plateau}</td>'
                h += f'<td style="padding:5px;text-align:center;">{_n(sc.get("ve_mean"), ".1f")}</td>'
                h += f'<td style="padding:5px;text-align:center;">{_n(sc.get("bf_vt_ratio"), ".0f")}%</td>'
                vt_d = sc.get('vt_drift_L')
                h += f'<td style="padding:5px;text-align:center;">{vt_d:+.3f}L</td>' if vt_d is not None else '<td style="padding:5px;text-align:center;">-</td>'
                h += '</tr>'
            h += '</tbody></table></div>'

        # ── NIRS / SmO₂ Kinetics Card ──
        if nirs_data.get('available') and nirs_data.get('stages'):
            nirs_stages = [ns for ns in nirs_data['stages'] if ns.get('smo2_mean') is not None]
            if nirs_stages:
                ch_name = nirs_data.get('channel', 'SmO2')
                smo2_rest = nirs_data.get('smo2_rest')
                smo2_min = nirs_data.get('smo2_min')
                desat_tot = nirs_data.get('desat_total')

                h += f'<div class="card"><div class="section-title">SmO₂ / NIRS — kinetyka ekstrakcji O₂ ({ch_name})</div>'

                # Summary boxes
                h += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;">'
                if smo2_rest:
                    h += f'''<div style="flex:1;text-align:center;padding:10px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">SmO₂ REST</div>
  <div style="font-size:24px;font-weight:800;color:#16a34a;">{smo2_rest}%</div>
</div>'''
                if smo2_min:
                    min_col = '#dc2626' if smo2_min < 30 else '#d97706' if smo2_min < 50 else '#16a34a'
                    h += f'''<div style="flex:1;text-align:center;padding:10px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">SmO₂ MIN</div>
  <div style="font-size:24px;font-weight:800;color:{min_col};">{smo2_min}%</div>
</div>'''
                if desat_tot:
                    h += f'''<div style="flex:1;text-align:center;padding:10px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Desaturacja</div>
  <div style="font-size:24px;font-weight:800;color:#d97706;">{desat_tot}%</div>
</div>'''
                # Extraction matching from NIRS
                matchings = [ns.get('extraction_matching') for ns in nirs_stages if ns.get('extraction_matching')]
                if matchings:
                    avg_match = sum(matchings) / len(matchings)
                    m_col = '#16a34a' if 0.7 <= avg_match <= 1.3 else '#d97706'
                    h += f'''<div style="flex:1;text-align:center;padding:10px;background:#f8fafc;border-radius:8px;border:2px solid {m_col}30;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">O₂ Extraction Match</div>
  <div style="font-size:24px;font-weight:800;color:{m_col};">{avg_match:.2f}</div>
  <div style="font-size:10px;color:#94a3b8;">τ_SmO₂ / τ_VO₂ (≈1.0 = matched)</div>
</div>'''
                h += '</div>'

                # Per-stage table
                h += '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr style="background:#f8fafc;">'
                h += '<th style="padding:5px;text-align:left;">Stadium</th>'
                h += '<th style="padding:5px;text-align:center;">SmO₂ mean</th>'
                h += '<th style="padding:5px;text-align:center;">SmO₂ min</th>'
                h += '<th style="padding:5px;text-align:center;">Drift</th>'
                h += '<th style="padding:5px;text-align:center;">Desat slope</th>'
                h += '<th style="padding:5px;text-align:center;">τ SmO₂ est</th>'
                h += '<th style="padding:5px;text-align:center;">τ VO₂</th>'
                h += '<th style="padding:5px;text-align:center;">Match</th>'
                h += '</tr></thead><tbody>'
                for ns in nirs_stages:
                    dom = ns.get('domain', '')
                    dcol = {'MODERATE': '#3b82f6', 'HEAVY': '#d97706', 'SEVERE': '#dc2626'}.get(dom, '#64748b')
                    match = ns.get('extraction_matching')
                    mc = '#16a34a' if match and 0.7 <= match <= 1.3 else '#d97706' if match else '#94a3b8'
                    h += f'<tr style="border-top:1px solid #f1f5f9;">'
                    h += f'<td style="padding:5px;color:{dcol};font-weight:600;">S{ns.get("stage_num")} {dom}</td>'
                    h += f'<td style="padding:5px;text-align:center;">{_n(ns.get("smo2_mean"), ".1f")}%</td>'
                    h += f'<td style="padding:5px;text-align:center;">{_n(ns.get("smo2_min"), ".1f")}%</td>'
                    dr = ns.get('smo2_drift')
                    h += f'<td style="padding:5px;text-align:center;">{dr:+.1f}%</td>' if dr is not None else '<td style="padding:5px;text-align:center;">-</td>'
                    ds = ns.get('desat_slope_pct_min')
                    h += f'<td style="padding:5px;text-align:center;">{ds:+.2f}%/min</td>' if ds is not None else '<td style="padding:5px;text-align:center;">-</td>'
                    ts = ns.get('tau_smo2_est_s')
                    h += f'<td style="padding:5px;text-align:center;">{ts:.0f}s</td>' if ts is not None else '<td style="padding:5px;text-align:center;">-</td>'
                    tv = ns.get('tau_on_vo2')
                    h += f'<td style="padding:5px;text-align:center;">{tv:.0f}s</td>' if tv is not None else '<td style="padding:5px;text-align:center;">-</td>'
                    h += f'<td style="padding:5px;text-align:center;color:{mc};font-weight:700;">{match:.2f}</td>' if match else '<td style="padding:5px;text-align:center;">-</td>'
                    h += '</tr>'
                h += '</tbody></table>'

                # Interpretation
                if nirs_data.get('bp1_vs_vt1_s') is not None:
                    bp1_diff = nirs_data['bp1_vs_vt1_s']
                    agree = abs(bp1_diff) < 60
                    h += f'<div style="margin-top:8px;padding:8px 12px;background:{"#f0fdf4" if agree else "#fef3c7"};border-radius:6px;font-size:11px;color:{"#166534" if agree else "#92400e"};">'
                    h += f'{"✅" if agree else "⚠️"} SmO₂ breakpoint vs VT1: {bp1_diff:+.0f}s {"(zgoda)" if agree else "(rozbieżność)"}.'
                    if matchings:
                        avg_m = sum(matchings) / len(matchings)
                        if avg_m > 1.3:
                            h += ' Ekstrakcja wolniejsza niż delivery → limitacja obwodowa.'
                        elif avg_m < 0.7:
                            h += ' Ekstrakcja szybsza niż delivery → bottleneck centralny.'
                        else:
                            h += ' Delivery i ekstrakcja dobrze dopasowane.'
                    h += '</div>'
                h += '</div>'

        # ── Fingerprint + Trainability Card ──
        fp = composites.get('fingerprint', {})
        tr = composites.get('trainability', {})
        if fp:
            h += '<div class="card"><div class="section-title">Aerobic Fitness Fingerprint</div>'
            h += '<div style="display:flex;gap:16px;flex-wrap:wrap;">'

            # Fingerprint bars
            h += '<div style="flex:2;min-width:300px;">'
            for dim_name, dim_key, dim_emoji in [
                ('Capacity (VO₂max)', 'capacity', '🫀'),
                ('Thresholds (VT1/VT2)', 'thresholds', '📈'),
                ('Kinetics (τ/SC)', 'kinetics', '⚡'),
                ('Recovery (HRR/τ_off)', 'recovery', '💚'),
            ]:
                val = fp.get(dim_key, 0)
                bar_col = '#16a34a' if val >= 70 else '#d97706' if val >= 40 else '#dc2626'
                h += f'''<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
    <span>{dim_emoji} {dim_name}</span><span style="font-weight:700;color:{bar_col};">{val}/100</span>
  </div>
  <div style="background:#f1f5f9;border-radius:4px;height:10px;overflow:hidden;">
    <div style="width:{val}%;height:100%;background:{bar_col};border-radius:4px;"></div>
  </div>
</div>'''
            h += '</div>'

            # Trainability gauge
            if tr.get('total_score') is not None:
                ts = tr['total_score']
                tc = tr.get('class', '?')
                tc_pl = tr.get('class_pl', '')
                tc_col = '#dc2626' if tc == 'HIGH' else '#d97706' if tc == 'MODERATE' else '#16a34a'
                h += f'''<div style="flex:1;min-width:180px;text-align:center;padding:16px;background:#f8fafc;border-radius:8px;">
  <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;letter-spacing:0.5px;">Trainability Index</div>
  <div style="font-size:36px;font-weight:800;color:{tc_col};">{ts}</div>
  <div style="font-size:12px;color:{tc_col};font-weight:600;">{tc}</div>
  <div style="font-size:10px;color:#64748b;margin-top:4px;">{tc_pl}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:6px;">0=optymalny · 100=duży potencjał</div>
</div>'''
            h += '</div></div>'

        # ── Performance Model Card ──
        perf = composites.get('performance', {})
        if perf and perf.get('sport'):
            sport_n = perf['sport']
            h += f'<div class="card"><div class="section-title">Performance Model — {sport_n.upper()}</div>'
            if sport_n == 'hyrox':
                h += f'''<div style="display:flex;gap:12px;flex-wrap:wrap;">
  <div style="flex:1;text-align:center;padding:12px;background:#f8fafc;border-radius:8px;">
    <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Transition Tax</div>
    <div style="font-size:24px;font-weight:800;color:#d97706;">{perf.get("transition_tax_s",0):.0f}s</div>
    <div style="font-size:10px;color:#94a3b8;">τ × 8 transitions × 0.7</div>
  </div>
  <div style="flex:1;text-align:center;padding:12px;background:#f8fafc;border-radius:8px;">
    <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Sustainable Run Pace</div>
    <div style="font-size:24px;font-weight:800;color:#3b82f6;">{perf.get("sustainable_pace_kmh",0):.1f} km/h</div>
    <div style="font-size:10px;color:#94a3b8;">88% VT2 / economy loss</div>
  </div>
  <div style="flex:1;text-align:center;padding:12px;background:#f8fafc;border-radius:8px;">
    <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Run Segments Total</div>
    <div style="font-size:24px;font-weight:800;color:#0f172a;">{perf.get("run_segments_min",0):.1f} min</div>
    <div style="font-size:10px;color:#94a3b8;">8 × 1km @ sustainable pace</div>
  </div>
</div>
<div style="margin-top:8px;padding:8px 12px;background:#f0f9ff;border-radius:6px;font-size:11px;color:#1e40af;">
  📊 {perf.get("model_note","")}. Poprawa τ o 5s = ~28s mniej transition tax.
</div>'''
            elif sport_n == 'run':
                h += '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                if perf.get('half_marathon_pace_kmh'):
                    hm_min = int(perf['half_marathon_time_min'])
                    hm_sec = int((perf['half_marathon_time_min'] - hm_min) * 60)
                    h += f'''<div style="flex:1;text-align:center;padding:12px;background:#f8fafc;border-radius:8px;">
    <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">Half Marathon</div>
    <div style="font-size:24px;font-weight:800;color:#3b82f6;">{hm_min}:{hm_sec:02d}</div>
    <div style="font-size:10px;color:#94a3b8;">{perf["half_marathon_pace_kmh"]:.1f} km/h</div>
  </div>'''
                if perf.get('_10k_pace_kmh'):
                    t10 = perf['_10k_time_min']
                    t10_m = int(t10); t10_s = int((t10 - t10_m) * 60)
                    h += f'''<div style="flex:1;text-align:center;padding:12px;background:#f8fafc;border-radius:8px;">
    <div style="font-size:10px;text-transform:uppercase;color:#94a3b8;">10K</div>
    <div style="font-size:24px;font-weight:800;color:#16a34a;">{t10_m}:{t10_s:02d}</div>
    <div style="font-size:10px;color:#94a3b8;">{perf["_10k_pace_kmh"]:.1f} km/h</div>
  </div>'''
                h += '</div>'
            h += '</div>'

    # ═══════════════════════════════════════════════════════
    # LIMITATION DIAGNOSIS (enhanced with E22 triangulation)
    # ═══════════════════════════════════════════════════════
    h += f'''<div class="card">
  <div class="section-title">Diagnoza limitacji</div>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div style="flex:0 0 auto;">
      <div style="font-size:14px;font-weight:700;margin-bottom:4px;">{badge_class(lim_primary)}</div>
    </div>
    <div style="flex:1;min-width:250px;">'''

    for ev in lim_evidence:
        h += f'<div style="font-size:12px;color:#334155;padding:2px 0;">• {esc(str(ev))}</div>'

    h += '</div></div>'

    # E22 Limitation Triangulation
    if e22 and e22.get('status') == 'OK':
        lt = e22.get('composites', {}).get('limitation_triangle', {})
        vent = e22.get('ventilatory', {})

        if lt.get('n_signals', 0) >= 2:
            cp = lt.get('central_pct', 0)
            pp = lt.get('peripheral_pct', 0)
            ip = lt.get('integrated_pct', 0)
            conf = lt.get('confidence', 0)
            prim = lt.get('primary', '?')
            prim_pl = lt.get('primary_pl', '')

            h += '<div style="margin-top:12px;padding:12px;background:#f8fafc;border-radius:8px;">'
            h += '<div style="font-size:11px;font-weight:700;color:#0f172a;margin-bottom:6px;">🔺 Multi-signal Limitation Triangulation</div>'

            # 3 bars
            for label, val, col in [('Centralna', cp, '#dc2626'), ('Obwodowa', pp, '#d97706'), ('Zintegrowana', ip, '#16a34a')]:
                is_prim = (label == 'Centralna' and prim == 'CENTRAL') or (label == 'Obwodowa' and prim == 'PERIPHERAL') or (label == 'Zintegrowana' and prim == 'INTEGRATED')
                bw = f'border:2px solid {col};' if is_prim else ''
                h += f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
  <span style="width:90px;font-size:11px;color:#64748b;">{label}</span>
  <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;{bw}">
    <div style="width:{val}%;height:100%;background:{col};border-radius:3px;"></div>
  </div>
  <span style="width:35px;text-align:right;font-size:11px;font-weight:700;color:{col};">{val}%</span>
</div>'''

            h += f'<div style="font-size:11px;color:#64748b;margin-top:4px;">Dominacja: <b style="color:#0f172a;">{prim_pl}</b> (confidence: {conf}%, sygnałów: {lt.get("n_signals",0)})</div>'
            h += '</div>'

        # Ventilatory Integration Index
        if vent.get('kinetic_ventilatory_index'):
            kvi = vent['kinetic_ventilatory_index']
            integ = vent.get('integration', '?')
            integ_pl = vent.get('integration_pl', '')
            ve_slope = vent.get('ve_vco2_slope', 0)
            tau_m = vent.get('tau_mod', 0)

            integ_col = '#16a34a' if integ == 'FULLY_INTEGRATED' else '#d97706' if integ in ('VQ_MISMATCH','PERIPHERAL_LIMITED') else '#dc2626'
            h += f'''<div style="margin-top:8px;padding:10px 12px;background:{integ_col}08;border:1px solid {integ_col}30;border-radius:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="font-size:11px;font-weight:700;color:{integ_col};">Kinetic-Ventilatory Index: {kvi:.2f}</span>
      <span style="font-size:10px;color:#64748b;margin-left:8px;">({integ_pl})</span>
    </div>
    <div style="font-size:10px;color:#94a3b8;">τ={tau_m:.0f}s / VE/VCO₂={ve_slope:.1f}</div>
  </div>
</div>'''

    h += '</div>'

    # ═══════════════════════════════════════════════════════
    # TRAINING PRIORITIES
    # ═══════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════
    # TRAINING PRIORITIES — expanded & sport-contextualized
    # ═══════════════════════════════════════════════════════
    
    # Build comprehensive training recommendations based on kinetic profile
    tau_mod = kin_profile.get('tau_moderate')
    tau_hvy = kin_profile.get('tau_heavy')
    tau_mod_cls = kin_profile.get('tau_moderate_class', '')
    tau_hvy_cls = kin_profile.get('tau_heavy_class', '')
    sc_hvy = kin_profile.get('sc_heavy_pct', 0)
    sc_hvy_cls = kin_profile.get('sc_heavy_class', '')
    sc_sev = kin_profile.get('sc_severe_pct', 0)
    tau_ratio = kin_profile.get('tau_ratio_heavy_mod')
    heavy_width = kin_profile.get('heavy_zone_width_pct', 0)
    vt1_pct = kin_profile.get('vt1_pct_vo2max', 0)

    lim_primary = limitation.get('primary', '?')
    rec_class = e21.get('summary', {}).get('recovery_class')

    # Determine sport-specific emphasis
    sport_emphasis = {
        'hyrox': {'repeat_effort': True, 'endurance_base': True, 'recovery_speed': True,
                  'economy_focus': 'running segments', 'power_focus': 'station work'},
        'crossfit': {'repeat_effort': True, 'endurance_base': False, 'recovery_speed': True,
                     'economy_focus': 'metcons', 'power_focus': 'strength WODs'},
        'run': {'repeat_effort': False, 'endurance_base': True, 'recovery_speed': False,
                'economy_focus': 'running economy', 'power_focus': None},
        'triathlon': {'repeat_effort': False, 'endurance_base': True, 'recovery_speed': True,
                      'economy_focus': 'transition efficiency', 'power_focus': 'bike power'},
        'soccer': {'repeat_effort': True, 'endurance_base': True, 'recovery_speed': True,
                   'economy_focus': 'repeat sprint', 'power_focus': 'explosive actions'},
        'mma': {'repeat_effort': True, 'endurance_base': True, 'recovery_speed': True,
                'economy_focus': 'round pacing', 'power_focus': 'exchange bursts'},
    }.get(sport, {'repeat_effort': True, 'endurance_base': True, 'recovery_speed': True,
                  'economy_focus': 'general', 'power_focus': None})

    h += '<div class="card"><div class="section-title">Priorytety treningowe — kontekst kinetyczny</div>'

    # Sport context intro
    h += f'''<div style="background:#f0f9ff;border-radius:8px;padding:12px 16px;margin-bottom:14px;border-left:4px solid #3b82f6;">
  <div style="font-size:12px;font-weight:700;color:#1e40af;margin-bottom:4px;">📋 Kontekst dyscypliny: {sport_label}</div>
  <div style="font-size:12px;color:#334155;">Poniższe rekomendacje uwzględniają wymagania {sport_label.split(" ",1)[-1] if " " in sport_label else sport} — '''

    if sport_emphasis.get('repeat_effort'):
        h += f'zdolność do powtarzania wysiłków, szybka recovery między seriami'
    else:
        h += f'ekonomia wysiłku, utrzymanie stabilnego steady-state'
    if sport_emphasis.get('recovery_speed'):
        h += f', optymalizacja off-kinetics.'
    else:
        h += '.'
    h += '</div></div>'

    # ── Priority blocks ──
    priority_num = 0

    # PRIORITY 1: τ moderate optimization
    if tau_mod and tau_mod_cls in ('TRAINED', 'ACTIVE', 'SLOW'):
        priority_num += 1
        if tau_mod_cls == 'TRAINED':
            target_tau = 16
            urgency = 'ROZWÓJ'
            urgency_col = '#2563eb'
            methods = [
                ('SIT — Sprint Interval Training', f'4–6× 30s all-out / 4min recovery. Na bieżni: {vt1_speed}+ km/h sprint.', 'Rekrutacja Type II + stymulacja mitochondrialna — najskuteczniejszy bodziec na τ (Bailey 2009)'),
                ('Priming — podwójne rozgrzewki', f'6min @ {_n(vt1_speed, ".0f")} km/h → 10min rest → 6min @ {_n(vt1_speed, ".0f")} km/h przed wysiłkiem głównym.', 'Przyspiesza kinetykę VO₂ o 15–25% w kolejnym wysiłku (Burnley & Jones 2007)'),
                ('Tempo repeats', f'4–6× 4min @ {_n(vt2_speed, ".1f")}–{_n(float(vt2_speed)*1.02 if vt2_speed else 0, ".1f")} km/h / 2min jog.', 'Stymulacja heavy→severe transition, poprawa τ i VT2'),
            ]
        elif tau_mod_cls == 'ACTIVE':
            target_tau = 25
            urgency = 'PRIORYTET'
            urgency_col = '#d97706'
            methods = [
                ('HIIT klasyczny', f'6–8× 3min @ {_n(float(vt2_speed)*0.95 if vt2_speed else 0, ".1f")} km/h / 2min active recovery.', 'Poprawa O₂ delivery + mitochondrialna biogeneza'),
                ('Fartlek z progresją', f'30min z naprzemiennymi segmentami: 3min @ {_n(vt1_speed, ".0f")} / 2min @ {_n(vt2_speed, ".0f")} km/h.', 'Naturalna stymulacja on-kinetics w zmiennych warunkach'),
                ('Baza aerobowa Z2', f'60–90min @ {_n(float(vt1_speed)*0.85 if vt1_speed else 0, ".0f")}–{_n(float(vt1_speed)*0.92 if vt1_speed else 0, ".0f")} km/h (RER <0.90).', 'Fundament — gęstość kapilarna i mitochondrialna'),
            ]
        else:  # SLOW
            target_tau = 35
            urgency = 'KRYTYCZNY'
            urgency_col = '#dc2626'
            methods = [
                ('Base building Z2', f'Minimum 4×/tyg 45–90min @ {_n(float(vt1_speed)*0.80 if vt1_speed else 0, ".0f")}–{_n(float(vt1_speed)*0.90 if vt1_speed else 0, ".0f")} km/h.', 'Gęstość kapilarna i mitochondrialna biogeneza — bez tego dalsze interwały nieefektywne'),
                ('Sweet spot', f'3–4× 8min @ {_n(float(vt1_speed)*1.0 if vt1_speed else 0, ".0f")}–{_n(float(vt2_speed)*0.95 if vt2_speed else 0, ".0f")} km/h / 3min easy.', 'Optymalna stymulacja przy zarządzalnym obciążeniu'),
            ]

        h += f'''<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;border-left:4px solid {urgency_col};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:800;color:#0f172a;">#{priority_num}</span>
      <span style="font-size:14px;font-weight:700;color:#0f172a;margin-left:8px;">Poprawa kinetyki ON (τ moderate)</span>
    </div>
    <span style="padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;background:{urgency_col}15;color:{urgency_col};border:1px solid {urgency_col}40;">{urgency}</span>
  </div>
  <div style="display:flex;gap:20px;margin-bottom:10px;font-size:12px;">
    <div><span style="color:#64748b;">Aktualnie:</span> <b>{_n(tau_mod, ".0f")}s</b> ({tau_mod_cls})</div>
    <div><span style="color:#64748b;">Cel:</span> <b>{target_tau}s</b> (ELITE)</div>
    <div><span style="color:#64748b;">Oczekiwany czas:</span> <b>6–12 tyg</b></div>
  </div>'''

        for method_name, prescription, rationale in methods:
            h += f'''<div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 {method_name}</div>
    <div style="font-size:12px;color:#334155;margin-bottom:2px;">{prescription}</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">{rationale}</div>
  </div>'''

        # Frequency
        freq = '1–2×/tyg SIT + 1×/tyg tempo' if tau_mod_cls == 'TRAINED' else ('2–3×/tyg HIIT/tempo' if tau_mod_cls == 'ACTIVE' else '4×/tyg base + 1×/tyg sweet spot')
        h += f'<div style="font-size:11px;color:#475569;margin-top:8px;">📅 <b>Częstotliwość:</b> {freq}</div>'
        h += '</div>'

    # PRIORITY 2: τ heavy (if different class than moderate)
    if tau_hvy and tau_hvy_cls != tau_mod_cls and tau_hvy_cls in ('ACTIVE', 'SLOW'):
        priority_num += 1
        h += f'''<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;border-left:4px solid #d97706;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:800;color:#0f172a;">#{priority_num}</span>
      <span style="font-size:14px;font-weight:700;color:#0f172a;margin-left:8px;">Poprawa kinetyki Heavy domain (τ heavy)</span>
    </div>
    <span style="padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;background:#d9770615;color:#d97706;border:1px solid #d9770640;">PRIORYTET</span>
  </div>
  <div style="display:flex;gap:20px;margin-bottom:10px;font-size:12px;">
    <div><span style="color:#64748b;">Aktualnie:</span> <b>{_n(tau_hvy, ".0f")}s</b> ({tau_hvy_cls})</div>
    <div><span style="color:#64748b;">Cel:</span> <b>25s</b> (TRAINED)</div>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Threshold intervals</div>
    <div style="font-size:12px;color:#334155;">4–6× 5min @ {_n(vt2_speed, ".0f")} km/h (próg VT2) / 3min active recovery @ {_n(float(vt1_speed)*0.85 if vt1_speed else 0, ".0f")} km/h.</div>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Heavy domain repeats</div>
    <div style="font-size:12px;color:#334155;">6–8× 3min @ {_n(float(vt1_speed)*1.05 if vt1_speed else 0, ".1f")}–{_n(float(vt2_speed)*0.95 if vt2_speed else 0, ".1f")} km/h / 2min easy.</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">Bezpośrednia stymulacja heavy domain — poprawa O₂ extraction i kapilaryzacji</div>
  </div>
</div>'''

    # PRIORITY 3: SC management (show differently based on current level)
    priority_num += 1
    if sc_hvy_cls in ('HIGH', 'VERY_HIGH'):
        sc_urgency = 'PRIORYTET'
        sc_col = '#ea580c'
        sc_methods = f'''<div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Trening bazowy Z2 + ekonomia</div>
    <div style="font-size:12px;color:#334155;">Zwiększ objętość Z2 o 15–20%. 4–5×/tyg 45–90min @ RER &lt;0.88.</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">Wysoki SC ({_n(sc_hvy, ".1f")}%) = rekrutacja Type II zbyt wcześnie → buduj bazę mitochondrialną Type I</div>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Plyometryka + stiffness</div>
    <div style="font-size:12px;color:#334155;">2×/tyg: skip drills, bounding, drop jumps. Poprawa economy reduces SC.</div>
  </div>'''
    else:
        sc_urgency = 'UTRZYMANIE'
        sc_col = '#16a34a'
        sc_methods = f'''<div style="background:#f0fdf4;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#166534;margin-bottom:3px;">✅ Doskonała ekonomia mięśniowa — utrzymuj</div>
    <div style="font-size:12px;color:#334155;">SC heavy: {_n(sc_hvy, ".1f")}% | SC severe: {_n(sc_sev, ".1f")}% — poniżej 3% to poziom elitarny.</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">Kontynuuj bazę aerobową Z2 i pracę nad ekonomią biegową. Nie zmieniaj tego co działa.</div>
  </div>'''

    h += f'''<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;border-left:4px solid {sc_col};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:800;color:#0f172a;">#{priority_num}</span>
      <span style="font-size:14px;font-weight:700;color:#0f172a;margin-left:8px;">Slow Component — ekonomia mięśniowa</span>
    </div>
    <span style="padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;background:{sc_col}15;color:{sc_col};border:1px solid {sc_col}40;">{sc_urgency}</span>
  </div>
  {sc_methods}
</div>'''

    # PRIORITY 4: Sport-specific (HYROX/CrossFit: repeat effort, recovery)
    if sport_emphasis.get('repeat_effort') and sport_emphasis.get('recovery_speed'):
        priority_num += 1
        h += f'''<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;border-left:4px solid #7c3aed;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:800;color:#0f172a;">#{priority_num}</span>
      <span style="font-size:14px;font-weight:700;color:#0f172a;margin-left:8px;">Repeat-effort capacity ({sport_label.split(" ",1)[-1] if " " in sport_label else sport})</span>
    </div>
    <span style="padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;background:#7c3aed15;color:#7c3aed;border:1px solid #7c3aed40;">SPORT-SPECIFIC</span>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Transition training (on/off kinetyka)</div>
    <div style="font-size:12px;color:#334155;">8–10× (90s @ {_n(float(vt2_speed)*1.05 if vt2_speed else 0, ".1f")} km/h → 90s @ {_n(float(vt1_speed)*0.75 if vt1_speed else 0, ".0f")} km/h). Symulacja zmian stacja↔bieg.</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">Trenuje szybkość przejścia z high-output na recovery — kluczowe w {sport_emphasis.get("economy_focus", "")}</div>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Race-pace simulation</div>
    <div style="font-size:12px;color:#334155;">4 rundy: (station work 3–4min @ max → 1km bieg @ {_n(float(vt2_speed)*0.92 if vt2_speed else 0, ".1f")} km/h). Rest: 2min walk.</div>
    <div style="font-size:10px;color:#94a3b8;font-style:italic;">Specificity — trenuj kinetykę w warunkach zawodów</div>
  </div>
  <div style="background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:3px;">💪 Off-kinetics priming</div>
    <div style="font-size:12px;color:#334155;">Pre-race warm-up: 6min @ VT1 → 3min rest → 3min @ VT2 → 5min rest → START. Przyspiesza τ o ~20%.</div>
  </div>
</div>'''

    # PRIORITY 5: Periodization note
    priority_num += 1
    h += f'''<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;border-left:4px solid #64748b;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:800;color:#0f172a;">#{priority_num}</span>
      <span style="font-size:14px;font-weight:700;color:#0f172a;margin-left:8px;">Periodyzacja i re-test</span>
    </div>
    <span style="padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;background:#64748b15;color:#64748b;border:1px solid #64748b40;">PLAN</span>
  </div>
  <div style="font-size:12px;color:#334155;padding:4px 0;">
    📅 <b>Re-test kinetyki:</b> za 8–12 tygodni po bloku treningowym. Porównaj τ i SC.
  </div>
  <div style="font-size:12px;color:#334155;padding:4px 0;">
    📊 <b>Markerami postępu</b> będą: τ moderate (&lt;20s = ELITE), τ heavy (&lt;25s), SC heavy &lt;3%.
  </div>
  <div style="font-size:12px;color:#334155;padding:4px 0;">
    ⚠️ <b>Uwaga:</b> τ reaguje na trening szybciej niż VO₂max (4–6 tyg vs 8–12 tyg). SC zmienia się wolniej (12–16 tyg).
  </div>
</div>'''

    h += '</div>'

    # ═══════════════════════════════════════════════════════
    # FLAGS & PROTOCOL VALIDATION
    # ═══════════════════════════════════════════════════════
    h += '<div class="card"><div class="section-title">Walidacja protokołu</div>'
    
    # Domain validation summary
    h += '<div style="margin-bottom:8px;">'
    for dv in (domain_val.get('stages', []) if isinstance(domain_val, dict) else []):
        snum = dv.get('stage_num', 0)
        match = dv.get('domain_match', True)
        dom_exp = dv.get('domain_expected', '?')
        dom_det = dv.get('domain', '?')
        icon = '✅' if match else '⚠️'
        h += f'<span style="margin-right:12px;font-size:12px;">S{snum}: {icon} {dom_det}'
        if not match:
            h += f' <span style="color:#ea580c;font-size:10px;">(oczekiwano: {dom_exp})</span>'
        h += '</span>'
    h += '</div>'

    # Flags
    if flags:
        h += '<div style="margin-top:6px;">'
        for fl in flags:
            is_warn = any(w in str(fl) for w in ['MISMATCH', 'SHORT', 'LOW_R2', 'PRIMING'])
            fl_col = '#ea580c' if is_warn else '#64748b'
            h += f'<span class="flag-item" style="background:{fl_col}10;color:{fl_col};border:1px solid {fl_col}30;">{esc(str(fl))}</span>'
        h += '</div>'

    h += f'<div style="margin-top:8px;font-size:11px;color:#94a3b8;">Engine: E14 v{e14.get("version", "?")} + E21 v{e21.get("version", "?")}</div>'
    h += '</div>'

    # ═══════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════
    h += f'''<div style="text-align:center;padding:12px;font-size:10px;color:#94a3b8;">
  PeakLab · Raport kinetyki VO₂ · Wygenerowano automatycznie
</div>'''

    h += '</div></body></html>'
    return h



def generate_kinetics_charts(df, results):
    """Generate all kinetics charts as base64 PNGs.
    Returns dict with keys: trace, tau, sc, recovery"""
    charts = {}
    if df is None:
        return charts
    # Handle different column name conventions (canonicalized vs raw)
    _t_col = 'Time_s' if 'Time_s' in df.columns else 'Time_sec' if 'Time_sec' in df.columns else None
    _vo2_col = 'VO2_ml_min' if 'VO2_ml_min' in df.columns else 'VO2_mlmin' if 'VO2_mlmin' in df.columns else None
    if _t_col is None or _vo2_col is None:
        return charts
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as _pd
        import base64 as _b64
        from io import BytesIO as _BytesIO

        # Alias columns so rest of function uses consistent names
        df = df.copy()
        if 'Time_s' not in df.columns and _t_col:
            df['Time_s'] = df[_t_col]
        if 'VO2_ml_min' not in df.columns and _vo2_col:
            df['VO2_ml_min'] = df[_vo2_col]
        if 'HR_bpm' not in df.columns:
            for _hc in ['HR_bpm', 'HR', 'hr']:
                if _hc in df.columns:
                    df['HR_bpm'] = df[_hc]; break

        plt.rcParams.update({'font.size':10,'axes.facecolor':'#fff','figure.facecolor':'#fff',
            'axes.edgecolor':'#e2e8f0','axes.grid':True,'grid.color':'#f1f5f9','grid.alpha':0.8,
            'axes.spines.top':False,'axes.spines.right':False})
        _DC = {'MODERATE':'#3b82f6','HEAVY':'#d97706','SEVERE':'#dc2626','VERY_SEVERE':'#7c3aed'}

        def _f2b(fig, dpi=150):
            buf = _BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.15)
            buf.seek(0); b = _b64.b64encode(buf.read()).decode('utf-8'); plt.close(fig); return b

        e14 = results.get('E14', {}); e02 = results.get('E02', {}); e01 = results.get('E01', {})
        stages = e14.get('stages', [])

        # CHART 1: VO2 trace
        try:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            t = df['Time_s'].values / 60; vo2 = df['VO2_ml_min'].values
            vo2s = _pd.Series(vo2).rolling(15, center=True, min_periods=3).mean().values
            ax.set_ylim(0, max(vo2) * 1.15)
            for s in stages:
                t0, t1 = s.get('t_start',0)/60, s.get('t_end',0)/60
                dom = s.get('domain','MODERATE'); col = _DC.get(dom,'#94a3b8')
                ax.axvspan(t0, t1, alpha=0.08, color=col)
                mid = (t0+t1)/2
                ax.text(mid, max(vo2)*1.10, f"S{s['stage_num']}  {s.get('speed_kmh',0)} km/h",
                        ha='center', va='center', fontsize=8, fontweight='bold', color=col,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.12, edgecolor=col, linewidth=0.5))
            ax.scatter(t, vo2, s=3, alpha=0.15, color='#64748b')
            ax.plot(t, vo2s, color='#0f172a', linewidth=1.8)
            vt1v = e02.get('vt1_vo2_abs'); vt2v = e02.get('vt2_vo2_abs'); vmx = e01.get('vo2peak_abs_mlmin')
            if vt1v:
                ax.axhline(float(vt1v), color='#3b82f6', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(t.max()*0.98, float(vt1v), 'VT1', ha='right', va='bottom', fontsize=8, color='#3b82f6', fontweight='bold')
            if vt2v:
                ax.axhline(float(vt2v), color='#dc2626', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(t.max()*0.98, float(vt2v), 'VT2', ha='right', va='bottom', fontsize=8, color='#dc2626', fontweight='bold')
            if vmx:
                ax.axhline(float(vmx), color='#0f172a', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(t.max()*0.98, float(vmx), 'VO\u2082max', ha='right', va='bottom', fontsize=8, color='#0f172a', fontweight='bold')
            ax.set_xlabel('Czas [min]'); ax.set_ylabel('VO\u2082 [ml/min]')
            ax.set_title('Przebieg VO\u2082 \u2014 test kinetyczny CWR', fontsize=12, fontweight='bold', pad=10)
            charts['trace'] = _f2b(fig)
        except: pass

        # CHART 2: tau bars
        try:
            dd, tts, ccs = [], [], []
            for s in stages:
                if s.get('tau_on_s') and s.get('domain') in ('MODERATE','HEAVY','SEVERE'):
                    dd.append(s['domain']); tts.append(s['tau_on_s']); ccs.append(_DC.get(s['domain'],'#64748b'))
            if dd:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                x = np.arange(len(dd)); bars = ax.bar(x, tts, color=ccs, width=0.6, edgecolor='white', linewidth=1, zorder=3)
                ax.axhspan(0,20,alpha=0.06,color='#7c3aed'); ax.axhspan(20,35,alpha=0.06,color='#16a34a')
                ax.axhspan(35,55,alpha=0.06,color='#d97706'); ax.axhspan(55,100,alpha=0.06,color='#dc2626')
                ax.text(len(dd)-0.3,10,'ELITE',fontsize=7,color='#7c3aed',ha='right',style='italic')
                ax.text(len(dd)-0.3,27,'TRAINED',fontsize=7,color='#16a34a',ha='right',style='italic')
                ax.text(len(dd)-0.3,44,'ACTIVE',fontsize=7,color='#d97706',ha='right',style='italic')
                ax.text(len(dd)-0.3,70,'SLOW',fontsize=7,color='#dc2626',ha='right',style='italic')
                for b, tau in zip(bars, tts):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f'{tau:.0f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
                dl = {'MODERATE':'Moderate\n(<VT1)','HEAVY':'Heavy\n(VT1\u2194VT2)','SEVERE':'Severe\n(>VT2)'}
                ax.set_xticks(x); ax.set_xticklabels([dl.get(d,d) for d in dd], fontsize=9)
                ax.set_ylabel('\u03c4 on-kinetics [s]')
                ax.set_title('Sta\u0142a czasowa adaptacji VO\u2082', fontsize=11, fontweight='bold', pad=8)
                ax.set_ylim(0, max(tts)*1.25)
                charts['tau'] = _f2b(fig)
        except: pass

        # CHART 3: SC drift
        try:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True); plotted = 0
            for s in stages:
                dom = s.get('domain','')
                if dom not in ('HEAVY','SEVERE') or plotted >= 2: continue
                ax = axes[plotted]; t0, t1 = s.get('t_start',0), s.get('t_end',0)
                mask = (df['Time_s']>=t0) & (df['Time_s']<=t1); sub = df[mask]
                if len(sub) < 10: continue
                tr = (sub['Time_s'].values-t0)/60; vo2 = sub['VO2_ml_min'].values
                vo2s = _pd.Series(vo2).rolling(11, center=True, min_periods=3).mean().values
                col = _DC.get(dom,'#64748b')
                ax.scatter(tr, vo2, s=8, alpha=0.2, color=col); ax.plot(tr, vo2s, color=col, linewidth=2)
                v = ~np.isnan(vo2s)
                if v.sum() > 5:
                    z = np.polyfit(tr[v], vo2s[v], 1)
                    ax.plot(tr, np.poly1d(z)(tr), '--', color='#0f172a', linewidth=1, alpha=0.6)
                    ax.text(0.95, 0.05, f'drift: {z[0]:+.0f} ml/min\u00b2', transform=ax.transAxes, ha='right', va='bottom',
                            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.set_title(f"S{s['stage_num']} {dom} ({s.get('speed_kmh',0)} km/h) \u2014 SC: {s.get('sc_pct',0):.1f}%",
                             fontsize=10, fontweight='bold', color=col)
                ax.set_xlabel('Czas [min]')
                if plotted == 0: ax.set_ylabel('VO\u2082 [ml/min]')
                plotted += 1
            fig.suptitle('Slow Component \u2014 drift VO\u2082 w czasie', fontsize=11, fontweight='bold', y=1.02)
            fig.tight_layout()
            charts['sc'] = _f2b(fig)
        except: pass

        # CHART 4: Recovery
        try:
            off_ks = e14.get('off_kinetics', [])
            valid = [o for o in off_ks if o.get('status')=='OK' and o.get('pct_recovered_60s') is not None]
            if valid:
                fig, ax = plt.subplots(figsize=(6, 3.5)); tps = [30,60,90,120]; w = 0.25; x = np.arange(len(tps))
                rcols = ['#d97706','#dc2626']
                for i, ok in enumerate(valid[:2]):
                    vals = [min(100, float(ok.get(f'pct_recovered_{tp}s',0) or 0)) for tp in tps]
                    off = (i - len(valid[:2])/2 + 0.5) * w
                    bars = ax.bar(x+off, vals, width=w, color=rcols[i%2], alpha=0.8, edgecolor='white', label=ok.get('transition',''))
                    for b, v in zip(bars, vals):
                        if v > 0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax.axhspan(0,40,alpha=0.04,color='#dc2626'); ax.axhspan(40,70,alpha=0.04,color='#d97706')
                ax.axhspan(70,100,alpha=0.04,color='#16a34a')
                ax.set_xticks(x); ax.set_xticklabels([f'{tp}s' for tp in tps])
                ax.set_ylabel('% recovered')
                ax.set_title('Recovery VO\u2082 \u2014 % odbudowy po wysi\u0142ku', fontsize=11, fontweight='bold', pad=8)
                ax.set_ylim(0, 115); ax.legend(fontsize=8, loc='upper left')
                charts['recovery'] = _f2b(fig)
        except: pass

        # CHART 5: Dual-axis VO2 + SmO2 trace
        try:
            e22 = results.get('E22', {})
            nirs = e22.get('nirs', {})
            if nirs.get('available') and nirs.get('time_series'):
                ts_nirs = nirs['time_series']
                t_sm = np.array(ts_nirs['t']) / 60
                s_sm = np.array(ts_nirs['smo2'])

                fig, ax1 = plt.subplots(figsize=(10, 4.5))
                t_vo2 = df['Time_s'].values / 60
                vo2 = df['VO2_ml_min'].values
                vo2s = _pd.Series(vo2).rolling(15, center=True, min_periods=3).mean().values

                # Domain backgrounds
                for s in stages:
                    t0, t1 = s.get('t_start',0)/60, s.get('t_end',0)/60
                    dom = s.get('domain','MODERATE')
                    col = _DC.get(dom,'#94a3b8')
                    ax1.axvspan(t0, t1, alpha=0.06, color=col)

                # VO2 on left axis
                ax1.scatter(t_vo2, vo2, s=2, alpha=0.1, color='#64748b')
                ax1.plot(t_vo2, vo2s, color='#0f172a', linewidth=1.5, label='VO₂')
                ax1.set_xlabel('Czas [min]')
                ax1.set_ylabel('VO₂ [ml/min]', color='#0f172a')
                ax1.tick_params(axis='y', labelcolor='#0f172a')
                ax1.set_ylim(0, max(vo2) * 1.15)

                # SmO2 on right axis
                ax2 = ax1.twinx()
                smo2_smooth = _pd.Series(s_sm).rolling(5, center=True, min_periods=1).mean().values
                ax2.plot(t_sm, smo2_smooth, color='#dc2626', linewidth=2.5, label='SmO₂', zorder=5)
                ax2.scatter(t_sm, s_sm, s=15, alpha=0.4, color='#dc2626', zorder=4)
                ax2.set_ylabel('SmO₂ [%]', color='#dc2626')
                ax2.tick_params(axis='y', labelcolor='#dc2626')
                ax2.set_ylim(max(0, min(s_sm) - 10), min(100, max(s_sm) + 10))

                # Legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

                ax1.set_title('VO₂ + SmO₂ — Delivery vs Extraction', fontsize=12, fontweight='bold', pad=10)
                fig.tight_layout()
                charts['nirs_dual'] = _f2b(fig)
        except: pass

        # CHART 6: Fingerprint Radar
        try:
            e22 = results.get('E22', {})
            fp = e22.get('composites', {}).get('fingerprint', {})
            if len(fp) >= 3:
                categories = []
                values = []
                for dim_name, dim_key in [('Capacity', 'capacity'), ('Thresholds', 'thresholds'),
                                           ('Kinetics', 'kinetics'), ('Recovery', 'recovery')]:
                    if dim_key in fp:
                        categories.append(dim_name)
                        values.append(fp[dim_key])

                if len(categories) >= 3:
                    N = len(categories)
                    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                    values_plot = values + [values[0]]
                    angles_plot = angles + [angles[0]]

                    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
                    ax.fill(angles_plot, values_plot, color='#3b82f6', alpha=0.15)
                    ax.plot(angles_plot, values_plot, color='#3b82f6', linewidth=2.5, marker='o', markersize=6)

                    # Value labels
                    for a, v, cat in zip(angles, values, categories):
                        ax.text(a, v + 8, f'{v}', ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')

                    ax.set_xticks(angles)
                    ax.set_xticklabels(categories, fontsize=10, fontweight='600')
                    ax.set_ylim(0, 100)
                    ax.set_yticks([25, 50, 75, 100])
                    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color='#94a3b8')
                    ax.grid(color='#e2e8f0', linewidth=0.5)

                    # Zone bands
                    for r, col in [(25, '#dc262620'), (50, '#d9770620'), (75, '#16a34a15'), (100, '#3b82f615')]:
                        circle = plt.Circle((0, 0), r, transform=ax.transData + ax.transAxes,
                                           fill=False, edgecolor=col, linewidth=0)

                    ax.set_title('Aerobic Fitness Fingerprint', fontsize=12, fontweight='bold', pad=20)
                    fig.tight_layout()
                    charts['radar'] = _f2b(fig)
        except: pass

    except Exception as _chart_err:
        print(f"⚠️ generate_kinetics_charts error: {_chart_err}")
        import traceback; traceback.print_exc()
    return charts


def inject_kinetics_charts(html, charts):
    """Insert chart PNGs into kinetics report HTML at correct positions."""
    if not charts:
        return html
    if charts.get('trace'):
        c1 = f'<div class="card"><div class="section-title">Przebieg VO\u2082 \u2014 pe\u0142ny test kinetyczny</div><img src="data:image/png;base64,{charts["trace"]}" style="width:100%;border-radius:8px;"></div>'
        html = html.replace(
            '<div class="card"><div class="section-title">Protok\u00f3\u0142 kinetyczny',
            c1 + '\n<div class="card"><div class="section-title">Protok\u00f3\u0142 kinetyczny')
    if charts.get('tau') or charts.get('sc'):
        parts = []
        if charts.get('tau'):
            parts.append(f'<div style="flex:1;min-width:280px;"><img src="data:image/png;base64,{charts["tau"]}" style="width:100%;border-radius:8px;"></div>')
        if charts.get('sc'):
            parts.append(f'<div style="flex:1.5;min-width:380px;"><img src="data:image/png;base64,{charts["sc"]}" style="width:100%;border-radius:8px;"></div>')
        c23 = f'<div class="card"><div class="section-title">Wizualizacja kinetyki</div><div style="display:flex;gap:12px;flex-wrap:wrap;">{"".join(parts)}</div></div>'
        html = html.replace(
            '<div class="card">\n  <div class="section-title">Slow Component',
            c23 + '\n<div class="card">\n  <div class="section-title">Slow Component')
    if charts.get('recovery'):
        c4 = f'<div class="card"><div class="section-title">Wizualizacja recovery</div><img src="data:image/png;base64,{charts["recovery"]}" style="width:70%;border-radius:8px;margin:0 auto;display:block;"></div>'
        html = html.replace(
            '<div class="card">\n  <div class="section-title">Diagnoza limitacji',
            c4 + '\n<div class="card">\n  <div class="section-title">Diagnoza limitacji')
    if charts.get('nirs_dual'):
        c5 = f'<div class="card"><div class="section-title">VO\u2082 + SmO\u2082 \u2014 Delivery vs Extraction</div><img src="data:image/png;base64,{charts["nirs_dual"]}" style="width:100%;border-radius:8px;"></div>'
        # Insert before Triple Drift if exists, else before Diagnoza
        if 'Triple Drift' in html:
            html = html.replace(
                '<div class="card"><div class="section-title">Analiza drift',
                c5 + '\n<div class="card"><div class="section-title">Analiza drift')
        else:
            html = html.replace(
                '<div class="card">\n  <div class="section-title">Diagnoza limitacji',
                c5 + '\n<div class="card">\n  <div class="section-title">Diagnoza limitacji')
    if charts.get('radar'):
        c6 = f'<div style="text-align:center;"><img src="data:image/png;base64,{charts["radar"]}" style="width:50%;min-width:280px;border-radius:8px;"></div>'
        # Insert into fingerprint card
        if 'Aerobic Fitness Fingerprint' in html:
            html = html.replace(
                '<div class="card"><div class="section-title">Aerobic Fitness Fingerprint</div>',
                f'<div class="card"><div class="section-title">Aerobic Fitness Fingerprint</div>{c6}')
    return html

print("\u2705 Kinetics Report module (render + charts + inject) za\u0142adowany.")
