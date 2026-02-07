"""
Network Construction Module
===========================

This module provides functions to create and configure the foundational biological components 
of the MMN simulation. It includes factories for:

- Neuron Groups (Excitatory/Inhibitory)
- Synaptic Connections (static, plastic, STDP)
- Cortical Columns (Layered P/PE/I structure)
- Memory Modules (Recurrent chains)

Functions:
    - create_neuron_group: Builds a Brian2 NeuronGroup with specified equations.
    - create_cortical_column: Assembles a predictive coding column (P, PE, I populations).
    - create_memory_module: Creates a memory trace module with sequential activation.
    - create_synaptic_connection: Helper for creating static or distance-dependent synapses.
    - create_stdp_synapse: Creates synapses with Spike-Timing Dependent Plasticity (STDP).
"""

from brian2 import *
import numpy as np

def create_neuron_group(n_neurons, name, neuron_type, params):
    core_eqs = '''du/dt = a*(b*v - u)/(1*ms) : volt
    a:1
    b:1
    c:volt
    d:volt'''
    dv_core = '(0.04*v**2/mV + 5*v + 140*mV - u)'

    if 'sigma_noise' in params:
        noise_term = f"+ {params['sigma_noise']} * sqrt(1*ms) * xi * mV"
    else:
        noise_map = {'input': '2.5', 'excitatory': 'sqrt(2.0)', 'inhibitory': 'sqrt(3.8)'}
        noise_term = f"+ {noise_map.get(neuron_type, '1.0')} * sqrt(1*ms) * xi * mV"

    if neuron_type == 'input':
        dv_eq = f'dv/dt = ({dv_core} + stimulus_current(t, i) {noise_term})/(1*ms) : volt'
        syn_eq = 'I_syn = 0*mV : volt'
    elif neuron_type == 'excitatory':
        dv_eq = f'dv/dt = ({dv_core} + I_ampa + I_gaba + I_nmda {noise_term})/(1*ms) : volt'
        syn_eq = '''
        I_syn = I_gaba + I_ampa + I_nmda : volt
        I_ampa = g_ampa*(V_E - v)*s_ampa : volt
        ds_ampa/dt=-s_ampa/tau_ampa : 1
        I_gaba = g_gaba*(V_I - v)*s_gaba : volt
        ds_gaba/dt=-s_gaba/tau_gaba : 1
        I_nmda = g_nmda*(V_E - v)*s_nmda / (1 + Mg2_conc * exp(-0.062*v/mV) / 3.57) : volt
        ds_nmda/dt = -s_nmda/tau_nmda_decay + alpha_nmda*x_nmda*(1-s_nmda) : 1
        dx_nmda/dt = -x_nmda/tau_nmda_rise : 1
        V_E:volt
        V_I:volt
        g_ampa:1
        g_gaba:1
        g_nmda:1
        Mg2_conc:1
        tau_ampa:second
        tau_gaba:second
        tau_nmda_rise:second
        tau_nmda_decay:second
        alpha_nmda:Hz
        '''
    elif neuron_type == 'inhibitory':
        dv_eq = f'dv/dt = ({dv_core} + I_ampa + I_nmda {noise_term})/(1*ms) : volt'
        syn_eq = '''
        I_syn = I_ampa + I_nmda : volt
        I_ampa = g_ampa*(V_E - v)*s_ampa : volt
        ds_ampa/dt=-s_ampa/tau_ampa : 1
        I_nmda = g_nmda*(V_E - v)*s_nmda / (1 + Mg2_conc * exp(-0.062*v/mV) / 3.57) : volt
        ds_nmda/dt = -s_nmda/tau_nmda_decay + alpha_nmda*x_nmda*(1-s_nmda) : 1
        dx_nmda/dt = -x_nmda/tau_nmda_rise : 1
        V_E:volt
        g_ampa:1
        g_nmda:1
        Mg2_conc:1
        tau_ampa:second
        tau_nmda_rise:second
        tau_nmda_decay:second
        alpha_nmda:Hz
        '''
    full_eqs = dv_eq + '\n' + core_eqs + '\n' + syn_eq
    group = NeuronGroup(n_neurons, full_eqs, threshold='v>=30*mV', reset='v=c; u=u+d', method='heun', name=name)
    params_copy = params.copy();
    params_copy.pop('sigma_noise', None)
    for key, value in params_copy.items(): setattr(group, key, value)
    group.v = -65 * mV;
    group.u = 0
    return group


def create_synaptic_connection(source, target, conn_prob, w_model, on_pre_action, delay_model=None, cond=None,
                               name=None):
    syn_name = name if name is not None else f'syn_{source.name}_{target.name}'
    syn = Synapses(source, target, model='w:1', on_pre=on_pre_action, name=syn_name)

    if cond:
        syn.connect(condition=cond)
    else:
        syn.connect(p=conn_prob)

    syn.w = w_model
    if delay_model:
        syn.delay = delay_model

    return syn


def create_plastic_synapse(source, target, conn_prob, initial_w, delay_model=None, conn_data=None, plasticity_on=True):
    print(f"'{source.name}' -> '{target.name}' PLASTIC synapse (Rule: NMDA-gated STDP, Plasticity: {'ON' if plasticity_on else 'OFF'}).")

    taup_val = 30*ms   # τp
    cp_val   = 60.0    # cp
    cd_val   = 5.0   # cd
    Th_val   = 0.6     # threshold
    eta_val  = 5e-6    # learning rate
    I_to_u   = 1.0     # INMDA scaling

    if plasticity_on:
        stdp_model = '''
        w : 1
        dApre/dt  = -Apre/taup : 1 (event-driven)
        dApost/dt = -Apost/taup : 1 (event-driven)
        taup : second (constant)
        cp   : 1 (constant)
        cd   : 1 (constant)
        Th   : 1 (constant)
        eta  : 1 (constant)
        Iu   : 1 (constant)
        wmin : 1 (constant)
        wmax : 1 (constant)
        '''
        on_pre_action = '''
        s_ampa_post += w
        x_nmda_post += w * 0.25
        Apre += 1
        w = w + eta*( cp*clip((I_nmda_post/mV - Th), 0, 1e9)*Apost*x_gate_pre - cd*x_gate_pre )
        w = clip(w, wmin, wmax)
        '''
        on_post_action = '''
        Apost += 1
        w = w + eta*( cp*clip((I_nmda_post/mV - Th), 0, 1e9)*Apre*x_gate_pre )
        w = clip(w, wmin, wmax)
        '''
    else:
        stdp_model = 'w : 1'
        on_pre_action = 's_ampa_post += w; x_nmda_post += w * 0.2'
        on_post_action = ''

    syn = Synapses(source, target, model=stdp_model,
                   on_pre=on_pre_action, on_post=on_post_action,
                   name=f'nmda_stdp_syn_{source.name}_{target.name}')

    if conn_data is not None:
        syn.connect(i=conn_data['i'], j=conn_data['j'])
    else:
        syn.connect(p=conn_prob)

    syn.w    = initial_w
    syn.taup = taup_val
    syn.cp   = cp_val
    syn.cd   = cd_val
    syn.Th   = Th_val
    syn.eta  = eta_val
    syn.Iu   = I_to_u
    syn.wmin = 0.0
    syn.wmax = 10.0

    if delay_model:
        syn.delay = delay_model

    return syn


def create_hebbian_synapse(pre_grp, post_grp, w_init, *,
                           conn_data=None,
                           delay_model='rand()*15*ms',
                           step=0.1,           # constant increase amount (dW)
                           w_min=0.0, w_max=10.0,
                           name=None):
    if name is None:
        name = f"hebb_{pre_grp.name}_to_{post_grp.name}"

    syn = Synapses(
        pre_grp, post_grp,
        model='''
        w     : 1
        w_min : 1
        w_max : 1
        step  : 1
        ''',
        on_pre='''
        s_ampa_post = s_ampa_post + w
        x_nmda_post = x_nmda_post + w*0.2
        w = clip(w + step, w_min, w_max)
        ''',
        name=name
    )
    if conn_data is not None:
        syn.connect(i=conn_data['i'], j=conn_data['j'])
    else:
        syn.connect(p=0.5)
    syn.w = w_init
    syn.w_min = w_min
    syn.w_max = w_max
    syn.step = step
    syn.delay = delay_model
    return syn


def create_stdp_synapse(pre_grp, post_grp, w_init, *,
                        conn_data=None,
                        delay_model='rand()*15*ms',
                        w_min=0.0, w_max=4.0,
                        A_plus=0.02,        # LTP strength
                        A_minus=-0.03,      # LTD trace (Apost += A_minus at post spike)
                        taupre_ms=15.0,
                        taupost_ms=25.0,
                        multiplicative=True,   # weight-dependent STDP
                        ltd_gain=2.0,          # LTD effect multiplier
                        name=None):

    """
    Creates a synapse with Spike-Timing Dependent Plasticity (STDP).
    
    The rule implements a weight-dependent or additive STDP:
    - LTP (Long-Term Potentiation): Occurs when pre-spike precedes post-spike.
    - LTD (Long-Term Depression): Occurs when post-spike precedes pre-spike (via trace).
    
    Args:
        pre_grp, post_grp: Source and target neuron groups.
        w_init: Initial weight matrix or value.
        conn_data: Dictionary {'i': [], 'j': []} for specific connectivity (optional).
        delay_model: String or Quantity for synaptic delay.
        w_min, w_max: Clipping bounds for weights.
        A_plus: Amplitude of potentiation.
        A_minus: Amplitude of depression (usually negative).
        taupre_ms, taupost_ms: Time constants for STDP traces.
        multiplicative: If True, uses soft bounds (wdiff ~ (w_max-w)). 
                        If False, uses additive updates (hard bounds).
    
    Returns:
        Brian2 Synapses object.
    """

    if name is None:
        name = f"stdp_{pre_grp.name}_to_{post_grp.name}"

    model = '''
    w      : 1
    w_min  : 1
    w_max  : 1
    A_plus : 1
    A_minus: 1
    taupre  : second
    taupost : second
    dApre/dt  = -Apre/taupre  : 1 (event-driven)
    dApost/dt = -Apost/taupost: 1 (event-driven)
    '''

    if multiplicative:
        on_pre = '''
        s_ampa_post += w
        x_nmda_post += w*0.2
        w = clip(w + (''' + str(float(ltd_gain)) + ''')*Apost*(w - w_min), w_min, w_max)
        Apre += A_plus
        '''
        on_post = '''
        w = clip(w + Apre*(w_max - w), w_min, w_max)
        Apost += A_minus
        '''
    else:
        on_pre = '''
        s_ampa_post += w
        x_nmda_post += w*0.2
        w = clip(w + (''' + str(float(ltd_gain)) + ''')*Apost, w_min, w_max)
        Apre += A_plus
        '''
        on_post = '''
        w = clip(w + Apre, w_min, w_max)
        Apost += A_minus
        '''

    syn = Synapses(pre_grp, post_grp, model=model,
                   on_pre=on_pre, on_post=on_post, name=name)

    if conn_data is not None:
        syn.connect(i=conn_data['i'], j=conn_data['j'])
    else:
        syn.connect(p=0.5)

    syn.w = w_init
    syn.w_min = w_min
    syn.w_max = w_max
    syn.A_plus = A_plus
    syn.A_minus = A_minus
    syn.taupre = taupre_ms * ms
    syn.taupost = taupost_ms * ms
    syn.Apre = 0
    syn.Apost = 0
    syn.delay = delay_model
    return syn


def create_cortical_column(column_id, N_exc, N_inh, params_exc, params_inh, synaptic_weights, record_states=True):
    """
    Creates a Cortical Column.
    """
    print(f"Creating Cortical Column '{column_id}'... (Detailed Monitors: {record_states})")
    PE = create_neuron_group(N_exc, f'PE_{column_id}', 'excitatory', params_exc)
    P = create_neuron_group(N_exc, f'P_{column_id}', 'excitatory', params_exc)
    I = create_neuron_group(N_inh, f'I_{column_id}', 'inhibitory', params_inh)
    w = synaptic_weights;
    delay = 'rand()*15*ms'
    on_pre_exc = 's_ampa_post = s_ampa_post + w; x_nmda_post = x_nmda_post + w * 0.2'
    on_pre_inh = 's_gaba_post = s_gaba_post + w'
    syn_P_I = create_synaptic_connection(P, I, 0.55, w["w_EI"], on_pre_exc, delay_model=delay,
                                         name=f'syn_P_I_{column_id}')
    syn_I_PE = create_synaptic_connection(I, PE, 0.55, w["w_IE"], on_pre_inh, delay_model=delay,
                                          name=f'syn_I_PE_{column_id}')
    syn_PE_P = create_synaptic_connection(PE, P, 0.55, w["w_EE"], on_pre_exc, delay_model=delay,
                                          name=f'syn_PE_P_{column_id}')

    monitors = {
        'spikemon_pe': SpikeMonitor(PE, name=f'spikemon_pe_{column_id}'),
        'ratemon_pe': PopulationRateMonitor(PE, name=f'ratemon_pe_{column_id}'),
        'spikemon_p': SpikeMonitor(P, name=f'spikemon_p_{column_id}'),
        'ratemon_p': PopulationRateMonitor(P, name=f'ratemon_p_{column_id}'),
        'spikemon_i': SpikeMonitor(I, name=f'spikemon_i_{column_id}'),
        'ratemon_i': PopulationRateMonitor(I, name=f'ratemon_i_{column_id}')
    }

    if record_states:
        neurons_to_record = range(min(10, N_exc))
        monitors['statemon_pe'] = StateMonitor(PE, ['v', 'I_syn'], record=neurons_to_record, dt=1 * ms,
                                               name=f'statemon_pe_{column_id}')
        monitors['statemon_p'] = StateMonitor(P, ['v', 'I_syn'], record=neurons_to_record, dt=1 * ms,
                                              name=f'statemon_p_{column_id}')
        monitors['statemon_i'] = StateMonitor(I, ['v', 'I_syn'], record=range(min(10, N_inh)), dt=1 * ms,
                                              name=f'statemon_i_{column_id}')

    return {'PE': PE, 'P': P, 'I': I, 'syn_P_I': syn_P_I, 'syn_I_PE': syn_I_PE, 'syn_PE_P': syn_PE_P, **monitors}


def create_memory_module(module_id, params_mem, N_E_mem, N_I_mem):
    print(f"Creating customized 'Memory Module {module_id}'...")
    params_exc_mem = params_mem['exc']
    params_inh_mem = params_mem['inh']
    w = params_mem['weights']

    E_chain = create_neuron_group(N_E_mem, f'E_chain_Mem_{module_id}', 'excitatory', params_exc_mem)
    I_pool = create_neuron_group(N_I_mem, f'I_pool_Mem_{module_id}', 'inhibitory', params_inh_mem)

    on_pre_e = 's_ampa_post = s_ampa_post + w; x_nmda_post = x_nmda_post + w*0.2'
    on_pre_i = 's_gaba_post = s_gaba_post + w'

    syn_ee_mem = create_synaptic_connection(E_chain, E_chain, None, f'{w["w_EE_mem"]}*(1+0.05*randn())', on_pre_e,
                                            delay_model=w["CHAIN_DELAY"], cond='i==j-1')
    syn_ei_mem = create_synaptic_connection(E_chain, I_pool, w["p_EI"], f'{w["w_EI_mem"]}*(1+0.05*randn())', on_pre_e,
                                            delay_model=w["E_TO_I_DELAY"])
    syn_ie_mem = create_synaptic_connection(I_pool, E_chain, w["p_IE"], f'{w["w_IE_mem"]}*(1+0.05*randn())', on_pre_i,
                                            delay_model=w["I_TO_E_DELAY"])

    spikemon_mem_e = SpikeMonitor(E_chain, name=f's_mem_e_{module_id}')

    return {'E_chain': E_chain, 'I_pool': I_pool, 'syn_ee': syn_ee_mem, 'syn_ei': syn_ei_mem,
            'syn_ie': syn_ie_mem, 'spikemon_mem_e': spikemon_mem_e}


def create_simple_memory_module(module_id, simple_params, N_chain):
    theta = float(simple_params.get('theta', 1.0))
    v_reset = float(simple_params.get('v_reset', 0.0))
    v_rest = float(simple_params.get('v_rest', 0.0))
    tau_m = simple_params.get('tau_m', 5 * ms)
    t_ref = simple_params.get('t_ref', 1 * ms)
    t_ref_first = simple_params.get('t_ref_first', t_ref)
    J_ff = float(simple_params.get('J_ff', 1.1))
    d_ff = simple_params.get('d_ff', 1.5 * ms)  # 400*1.5ms ≈ 600ms
    tau_gate = simple_params.get('tau_gate', 80 * ms)

    eqs = f'''
    dv/dt = (-(v - {v_rest}))/tau_m : 1 (unless refractory)
    tau_m   : second
    tau_ref : second
    dx_gate/dt = (1 - x_gate)/tau_gate : 1
    tau_gate : second
    '''

    E_chain = NeuronGroup(
        N_chain, model=eqs,
        threshold=f'v >= {theta}',
        reset=f'v = {v_reset}',
        refractory='tau_ref',
        method='euler',
        name=f'E_chain_LIF_{module_id}'
    )
    E_chain.v = v_rest
    E_chain.tau_m = tau_m
    E_chain.tau_gate = tau_gate
    E_chain.x_gate = 1.0
    E_chain.tau_ref = t_ref * np.ones(N_chain)
    E_chain.tau_ref[0] = t_ref_first  # allow re-trigger on every tone

    syn_ee_mem = Synapses(E_chain, E_chain, model='w:1',
                          on_pre=f'v_post = v_post + w*{J_ff}', name=f'syn_chain_{module_id}')
    syn_ee_mem.connect(condition='i==j-1')
    syn_ee_mem.delay = d_ff
    syn_ee_mem.w = 1.0

    spikemon_mem_e = SpikeMonitor(E_chain, name=f's_mem_e_{module_id}')
    return {'E_chain': E_chain, 'I_pool': None, 'syn_ee': syn_ee_mem,
            'syn_ei': None, 'syn_ie': None, 'spikemon_mem_e': spikemon_mem_e}
