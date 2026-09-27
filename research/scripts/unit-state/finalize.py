from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'.tools/plotting'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];notable=[];first_flags={};comparisons={};cleanup={};runs={}
for name in ('basic','charge','charge-extended'):
    directory=HERE/name
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    runs[name]={k:summary[k] for k in ('samples_by_type','field_count','module_sha256','unknown_counts')}
    cleanup[name]=summary['cleanup']
    assert cleanup[name]['private_files_removed'] and cleanup[name]['outputs_verified_and_preserved']
    errors=[];max_alive_error=0;max_native_error=0;max_ammo_error=0;shown=set()
    for line in (directory/'tww3_bai_map_capture_events.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['event']=='probe_error':errors.append(r)
        if r['event']!='state_sample' or r['name'] not in ('spears','archers','general'):continue
        s=r['readings']['sensors'];v=lambda k:s.get(k,{}).get('value')
        for flag in ('cco.IsRouting','cco.IsShattered','cco.IsWithdrawing','cco.IsAwaitingOrderAfterRally'):
            if v(flag) is True:first_flags.setdefault(name+':'+r['name']+':'+flag,{'ms':r['ms'],'stage':r['stage']})
        if v('cco.IsAlive') and v('cco.HealthMax'):
            max_alive_error=max(max_alive_error,abs(v('cco.HealthValue')/v('cco.HealthMax')-v('cco.HealthPercent')))
            max_native_error=max(max_native_error,abs(v('native.unary_hitpoints')-v('cco.HealthPercent')))
        if v('native.starting_ammo'):
            max_ammo_error=max(max_ammo_error,abs(v('native.ammo_left')/v('native.starting_ammo')-v('cco.PrimaryAmmoPercent')))
        if r['stage']=='withdraw' and v('cco.HealthValue')==0 and v('cco.HealthPercent')>0 and r['name'] not in shown:
            shown.add(r['name']);notable.append({'run':name,'name':r['name'],'ms':r['ms'],'stage':r['stage'],'values':{k:v(k) for k in ('cco.HealthValue','cco.HealthPercent','native.unary_hitpoints','native.number_of_men_alive','cco.IsAlive','cco.IsWithdrawing','native.is_leaving_battle')}})
        if name=='charge-extended' and r['name']=='archers':
            rows.append({'seconds':r['ms']/1000,'hp':v('native.unary_hitpoints'),'men':v('native.unary_of_men_alive'),'morale':v('cco.MoralePercent'),'card':v('cco.card.stat_morale.Value'),'routing':v('cco.IsRouting'),'shattered':v('cco.IsShattered')})
    assert not errors
    comparisons[name]={'max_HealthValue_div_HealthMax_vs_HealthPercent_when_alive':max_alive_error,'max_native_vs_CCO_HP_when_alive':max_native_error,'max_ammo_fraction_vs_CCO':max_ammo_error}
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
x=[r['seconds'] for r in rows]
axes[0].plot(x,[r['hp'] for r in rows],label='Native initial-HP fraction')
axes[0].plot(x,[r['men'] for r in rows],label='Native alive-men fraction')
axes[0].set(ylabel='Fraction',title='Empire Archers: natural damage and morale under cavalry attack')
axes[0].legend(loc='lower left')
axes[1].plot(x,[r['morale'] for r in rows],color='#aa3333',label='MoralePercent (left)')
axes[1].set_ylabel('MoralePercent');axes[1].axhline(0,color='#aaaaaa',linewidth=.7)
twin=axes[1].twinx();twin.plot(x,[r['card'] for r in rows],color='#3333aa',label='Card stat_morale.Value (right)')
twin.set_ylabel('Card numeric readout');axes[1].legend(loc='upper left');twin.legend(loc='lower left')
axes[2].step(x,[int(r['routing']) for r in rows],where='post',label='CCO IsRouting')
axes[2].step(x,[int(r['shattered']) for r in rows],where='post',label='CCO IsShattered',linestyle='--')
axes[2].set(yticks=[0,1],ylabel='Boolean',xlabel='Simulated battle time (s)');axes[2].legend(loc='upper left')
for ax in axes:ax.grid(alpha=.2)
fig.text(.5,.015,'0.5 s samples; diagnostic teleport setup, no forced rout. Card value and MoralePercent are distinct readouts.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(HERE/'natural-routing.png',dpi=160);plt.close(fig)
handoff={'status':'three_bounded_diagnostic_runs_completed','runs':runs,'total_own_samples':sum(sum(r['samples_by_type'].values()) for r in runs.values()),'first_true_flags':first_flags,'comparisons_alive_only':comparisons,'withdrawal_caveat_examples':notable,'cleanup':cleanup,'current_module_sha256':sha(ROOT/'src/units/state.lua'),'tests':'python -m unittest discover -s tests -p test_unit_state.py -v; 6 passed','notes':['basic capture predates correction of three unit-returning threat getters; its type_userdata entries are wrapper schema mistakes, not unsupported engine APIs. charge and charge-extended use the corrected module.', 'Card stat_morale was discovered by enumerating UnitDetailsContext.StatList keys. Its numeric values are recorded separately from morale state/percent; no exact panic-reserve conversion is claimed.','The own-only reader withholds all detailed enemy sensors even for visible enemies; visibility is its only enemy readout.','No abilities/flying/barriers/undead mechanics exercised; all probes use common state getters.','HoldingPosition/ActiveBehaviours were not found under those names in the consulted battle_unit/CcoBattleUnit references. Known behaviour flags and StatusList are captured; no unsupported equivalence is invented.','False-only fields establish a readable boolean, not the transition or its trigger.','Six tests and these three runs do not prove full arbitrary API/situation coverage or MATCH-IDLE-001 completeness.']}
(HERE/'handoff.json').write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'total_own_samples':handoff['total_own_samples'],'first_true_flags':first_flags,'comparisons_alive_only':comparisons},ensure_ascii=False,indent=2))
