import numpy as np
import pandas as pd
import random

from cpu_extruder_funcs import *
from cpu_cohesin_funcs import *
from cpu_rnap_funcs import *

import warnings
warnings.filterwarnings("ignore")

def cpu_make_n_steps(
    thread,
    n_steps,
    timestep,
    bead_sizes,
    elong_vel,
    rnap_pos,
    rnap_bristle,
    rnap_state,
    rnap_curr_intron,
    rnap_occupied,
    rnap_gene,
    genes,
    introns,
    coh_occupied,
    extr_occupied,
    extruders_pos,
    extruders_topo_leg,
    extruders_direction,
    extruder_change_direction_prob,
    cohesins_pos,
    rnap_remove_extruder_prob,
    extruder_unload_prob,
    extruder_load_prob,
    diff_prob,
    rng_states,
    rna_bead_params,
    topology_state,
    prob_func_name = None,
    extr_coh_interaction = True,
    extruder_rnap_interaction = True,
    smc_rnap_step_prob = 0.0,
    initialization_steps = False,
):
    set_prob_func(prob_func_name)
    for _ in range(n_steps):
        cpu_unload_rnap(
            thread, 
            rnap_pos,
            rnap_bristle,
            rnap_state,
            rnap_curr_intron,
            rnap_occupied,
            rnap_gene,
        )

        cpu_load_rnap(
            thread, 
            rnap_pos,
            rnap_bristle,
            rnap_state,
            rnap_curr_intron,
            rnap_occupied,
            rnap_gene,
            genes,
            rng_states,
        )

        cpu_translocate_rnap(
            thread, 
            timestep,
            bead_sizes,
            elong_vel,
            rnap_pos,
            rnap_bristle,
            rnap_state,
            rnap_curr_intron,
            rnap_occupied,
            rnap_gene,
            genes,
            introns,
            coh_occupied,
            extr_occupied,
            extruders_pos,
            extruders_topo_leg,
            extruders_direction,
            cohesins_pos,
            rng_states,
            rnap_remove_extruder_prob,
        )

        if not initialization_steps:
            cpu_unload_extruders(
                thread, 
                rnap_occupied,
                extr_occupied,
                extruders_pos,
                extruders_topo_leg,
                extruders_direction,
                rng_states,
                extruder_unload_prob,
                rnap_remove_extruder_prob,
            )

            cpu_extruder_change_direction(
                thread,
                extruders_pos,
                extruder_change_direction_prob,
                extruders_direction,
                rng_states,
            )

            cpu_load_extruders(
                thread, 
                rnap_occupied,
                coh_occupied,
                extr_occupied,
                extruders_pos,
                extruders_topo_leg,
                extruders_direction,
                rng_states,
                extruder_load_prob,
                topology_state,
            )

            cpu_translocate_extruder(
                thread, 
                rnap_occupied,
                rnap_pos,
                rnap_bristle,
                coh_occupied,
                extr_occupied,
                extruders_pos,
                extruders_topo_leg,
                extruders_direction,
                cohesins_pos,
                rng_states,
                rna_bead_params,
                extr_coh_interaction,
                extruder_rnap_interaction,
                smc_rnap_step_prob,
            )

        cpu_translocate_cohesin(
            thread, 
            diff_prob,
            rnap_occupied,
            coh_occupied,
            extr_occupied,
            cohesins_pos,
            rng_states,
            smc_rnap_step_prob,
        )
    
    return (
        timestep,
        bead_sizes,
        elong_vel,
        rnap_pos,
        rnap_bristle,
        rnap_state,
        rnap_curr_intron,
        rnap_occupied,
        rnap_gene,
        genes,
        introns,
        coh_occupied,
        extr_occupied,
        extruders_pos,
        extruders_topo_leg,
        extruders_direction,
        extruder_change_direction_prob,
        cohesins_pos,
        rnap_remove_extruder_prob,
        extruder_unload_prob,
        extruder_load_prob,
        diff_prob,
        rng_states,
        rna_bead_params,
        topology_state,
        prob_func_name,
        extr_coh_interaction,
        extruder_rnap_interaction,
        smc_rnap_step_prob,
    )

### Supplement functions
def load_sisters_for_cpu(cohesins_pos, occupied, args): 
    """
    A function to load one cohesin 
    """
    for i in range(cohesins_pos.shape[0]):
        for _ in range(1000):
            a = random.randint(0, args["N"]//2-1)
            if (occupied[a] == 0) and (occupied[args["N"]-1-a] == 0) and (a > 0) and (a < (args["N"]//2) - 1):
                occupied[a] = -(i+1)
                occupied[args["N"]-1-a] = i+1
                cohesins_pos[i, 0] = a
                cohesins_pos[i, -1] = args["N"]-1-a
                break
    return cohesins_pos, occupied

def load_coh_in_tandems(cohesins_pos, occupied, args, tandems):
    tmp_tandems = tandems.copy()
    tmp_tandems['counter'] = 1

    assert sum(occupied) == 0

    for i in range(cohesins_pos.shape[0]):
        a = random.randint(0, args["N"]//2-1)
        
        # Находим тандем, в который попала случайная позиция
        for idx, tandem in tmp_tandems.iterrows():
            if tandem['start_beads'] <= a <= tandem['end_beads']:
                # Если направление тандема положительное, вычитаем из конца
                if tandem['direction'] == '+':
                    position = tandem['end_beads'] - tandem['counter']
                # Если направление тандема отрицательное, вычитаем из начала
                else:  # direction == '-'
                    position = tandem['start_beads'] + tandem['counter']
                
                # Проверяем, что позиция свободна
                assert occupied[position] == 0 and occupied[args["N"]-1-position] == 0
                occupied[position] = -(i+1)
                occupied[args["N"]-1-position] = i+1
                cohesins_pos[i, 0] = position
                cohesins_pos[i, -1] = args["N"]-1-position
                
                # Увеличиваем счетчик для этого тандема
                prev_counter = tmp_tandems.at[idx, 'counter']
                tmp_tandems.at[idx, 'counter'] += 1
                assert prev_counter != tmp_tandems.at[idx, 'counter']
                break
            elif a < tandem['start_beads']:
                # Если мы находимся между тандемами, найдем ближайший
                if idx > 0:  # Проверяем, что есть предыдущий тандем
                    # Расстояние до текущего тандема
                    dist_current = tandem['start_beads'] - a
                    # Расстояние до предыдущего тандема
                    prev_tandem = tmp_tandems.iloc[idx-1]
                    dist_prev = a - prev_tandem['end_beads']
                    
                    # Выбираем ближайший тандем
                    if dist_prev < dist_current:
                        # Используем предыдущий тандем
                        if prev_tandem['direction'] == '+':
                            position = prev_tandem['end_beads'] - prev_tandem['counter']
                        else:  # direction == '-'
                            position = prev_tandem['start_beads'] + prev_tandem['counter']
                        
                        assert occupied[position] == 0 and occupied[args["N"]-1-position] == 0
                        occupied[position] = -(i+1)
                        occupied[args["N"]-1-position] = i+1
                        cohesins_pos[i, 0] = position
                        cohesins_pos[i, -1] = args["N"]-1-position
                        
                        # Увеличиваем счетчик для предыдущего тандема
                        prev_counter = tmp_tandems.at[idx-1, 'counter']
                        tmp_tandems.at[idx-1, 'counter'] += 1
                        assert prev_counter != tmp_tandems.at[idx-1, 'counter']
                        break
                    else:
                        if tandem['direction'] == '+':
                            position = tandem['end_beads'] - tandem['counter']
                        else:  # direction == '-'
                            position = tandem['start_beads'] + tandem['counter']
                        
                        assert occupied[position] == 0 and occupied[args["N"]-1-position] == 0
                        occupied[position] = -(i+1)
                        occupied[args["N"]-1-position] = i+1
                        cohesins_pos[i, 0] = position
                        cohesins_pos[i, -1] = args["N"]-1-position
                        
                        # Увеличиваем счетчик для предыдущего тандема
                        prev_counter = tmp_tandems.at[idx, 'counter']
                        tmp_tandems.at[idx, 'counter'] += 1
                        assert prev_counter != tmp_tandems.at[idx, 'counter']
                        break

    return cohesins_pos, occupied

def cpu_intons_maker(chosen_genes_beads):
    max_introns = 1
    for gene in chosen_genes_beads:
        introns = gene[-1]
        max_introns = max([max_introns, len(introns)])
        
    introns_array = np.zeros((len(chosen_genes_beads), max_introns, 2), dtype = int)
    for ig, gene in enumerate(chosen_genes_beads):
        introns = gene[-1]
        for i_int, intron in enumerate(introns):
            introns_array[ig, i_int, :] = intron
        for i_int in range(len(introns), max_introns):
            introns_array[ig, i_int, :] = [0,0]
    return introns_array

def create_tandems_df(chosen_genes, late_add='', filter_multiplier = 1.0):
    """
    Объединяет гены в тандемы.
    Тандем - это гены, которые находятся друг за другом на одной хромосоме и имеют одинаковое направление.
    
    Parameters:
    -----------
    chosen_genes : pandas.DataFrame
        DataFrame с информацией о генах
    late_add : str, optional
        Дополнительный суффикс для колонки 'rnap_sep_min_FPKM'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с информацией о тандемах
    """
    # Сортируем транскрипты по хромосоме и начальной позиции
    sorted_transcripts = chosen_genes[['chrom', 'start', 'end', 'direction', 'len', 'rnap_sep_min_FPKM' + late_add]].copy()
    sorted_transcripts = sorted_transcripts[sorted_transcripts['rnap_sep_min_FPKM' + late_add]*filter_multiplier < sorted_transcripts['len']]
    sorted_transcripts = sorted_transcripts.sort_values(['chrom', 'start']).reset_index(drop=True)

    # Создаем пустой список для хранения тандемов
    tandems = []

    # Инициализируем переменные для первого тандема
    current_chrom = None
    current_direction = None
    tandem_start = None
    tandem_end = None
    genes_len = 0

    # Проходим по всем транскриптам
    for idx, row in sorted_transcripts.iterrows():
        # Если это первый транскрипт или новая хромосома или новое направление
        if (current_chrom != row['chrom'] or current_direction != row['direction']):
            # Если уже был тандем, добавляем его в список
            if current_chrom is not None:
                tandems.append({
                    'chrom': current_chrom,
                    'start': tandem_start,
                    'end': tandem_end,
                    'direction': current_direction,
                    'tandem_len': tandem_end - tandem_start,
                    'genes_len': genes_len
                })
            
            # Начинаем новый тандем
            current_chrom = row['chrom']
            current_direction = row['direction']
            tandem_start = row['start']
            tandem_end = row['end']
            genes_len = row['len']
        else:
            # Продолжаем текущий тандем
            tandem_end = row['end']
            genes_len += row['len']

    # Добавляем последний тандем
    if current_chrom is not None:
        tandems.append({
            'chrom': current_chrom,
            'start': tandem_start,
            'end': tandem_end,
            'direction': current_direction,
            'tandem_len': tandem_end - tandem_start,
            'genes_len': genes_len
        })

    # Создаем DataFrame из списка тандемов
    return pd.DataFrame(tandems)

def initiate_steps_estimation(chosen_genes_beads, args, n_lifetimes = 10):
    max_for_gene = 0
    for gene in chosen_genes_beads:
        max_for_gene = max([(abs(gene[0] - gene[2])+1) * args['TIMESTEP'] * args['BEADSIZE_GENE'] / args['RNAP_VEL'], max_for_gene])

    return max([int(np.ceil(max_for_gene/args['steps_wo_saving'])), 
                int(np.ceil((n_lifetimes)*(int(np.max(args['LIFETIME']))/args['steps_wo_saving'])))])

def cpu_params_set_maker(nproc, max_rnaps, chosen_genes_beads, bead_sizes, args, seed = 1, tandems = None):
    # gpu arrays maker
    N = args['N']
    N_coh = args['N_coh']
    
    timestep = args['TIMESTEP']
    cpu_bead_sizes = np.array(bead_sizes, dtype = float) # args['BEADSIZE_GENE']
    elong_vel = args['RNAP_VEL']

    rnap_pos = np.zeros((nproc, max_rnaps,), dtype = int)
    rnap_bristle = np.zeros((nproc, max_rnaps, 2), dtype = float)
    rnap_state = np.zeros((nproc, max_rnaps,), dtype = int)
    rnap_curr_intron = np.zeros((nproc, max_rnaps,), dtype = int)
    rnap_occupied = np.zeros((nproc, N), dtype = int)
    rnap_gene = np.zeros((nproc, max_rnaps,), dtype = int)

    genes = [gene[:-1] for gene in chosen_genes_beads]
    introns = cpu_intons_maker(chosen_genes_beads)

    coh_occupied_base = np.zeros((nproc, N,), dtype = int)
    cohesins_pos_base = np.zeros((nproc, N_coh, 2), dtype = int)
    if tandems is not None:
        for i in range(cohesins_pos_base.shape[0]):
            (cohesins_pos_base[i, :, :], coh_occupied_base[i,:]) = load_coh_in_tandems(cohesins_pos_base[i, :, :], coh_occupied_base[i,:], args, tandems)
    else:
        for i in range(cohesins_pos_base.shape[0]):
            (cohesins_pos_base[i, :, :], coh_occupied_base[i,:]) = load_sisters_for_cpu(cohesins_pos_base[i, :, :], coh_occupied_base[i,:], args)
    cohesins_pos = cohesins_pos_base

    coh_occupied = coh_occupied_base
    extr_occupied = np.zeros((nproc, N,), dtype = int)

    maxcondesins = args['maxcondesins']
    extruders_pos = -1*np.ones((nproc, maxcondesins, 2), dtype = int)
    extruders_topo_leg = -1*np.ones((nproc, maxcondesins, 2), dtype = float)
    extruders_direction = -1*np.ones((nproc, maxcondesins,), dtype = int)
    extruder_change_direction_prob = args['EXTR_CHNG_DIR_PROB']

    rng_states = seed
    rnap_remove_extruder_prob = args['RNAP_REM_COND_PROB']
    if type(args['LIFETIME']) is float:
        extruder_unload_prob = np.ones((nproc, maxcondesins,), dtype = float) / args['LIFETIME']
    else:
        extruder_unload_prob = np.ones((nproc, maxcondesins,), dtype = float) / np.array(list(args['LIFETIME']) + list(args['LIFETIME2']), dtype=float)
        assert extruder_unload_prob.shape == (nproc, maxcondesins,)
    extruder_load_prob = 1 / args["BOUNDTIME"]
    diff_prob = args["Diffusion_prob"]
    topology_state = np.array(args["Topology_state"], dtype = float)
    
    rna_bead_params = [90.0, 15.0, np.pi, 45.0] # [RNA_bases_per_bead, RNA_bead_size, pi, extruder_d_bp]
    prob_func_name = args['prob_func_name']
    extr_coh_interaction = args['extr_coh_interaction']
    extruder_rnap_interaction = args.get('extruder_rnap_interaction', True)
    smc_rnap_step_prob = args.get('smc_rnap_step_prob', 0.0)
    
    return (
        timestep,
        cpu_bead_sizes,
        elong_vel,
        rnap_pos,
        rnap_bristle,
        rnap_state,
        rnap_curr_intron,
        rnap_occupied,
        rnap_gene,
        genes,
        introns,
        coh_occupied,
        extr_occupied,
        extruders_pos,
        extruders_topo_leg,
        extruders_direction,
        extruder_change_direction_prob,
        cohesins_pos,
        rnap_remove_extruder_prob,
        extruder_unload_prob,
        extruder_load_prob,
        diff_prob,
        rng_states,
        rna_bead_params,
        topology_state,
        prob_func_name,
        extr_coh_interaction,
        extruder_rnap_interaction,
        smc_rnap_step_prob,
    )