import numpy as np
import random
from cpu_cohesin_funcs import test_dna_end, test_chromatid_change
from cpu_prob_functions import *

def set_prob_func(prob_func_name):
    global PFN
    PFN = prob_func_name

def cpu_unload_one_extruder(
    thread, 
    ext_n,
    extr_occupied,
    extruders_pos,
    extruders_topo_leg,
    extruders_direction,
):
    if extr_occupied[thread, extruders_pos[thread, ext_n, 0]] == -(ext_n+1):
        extr_occupied[thread, extruders_pos[thread, ext_n, 0]] = 0
    if extr_occupied[thread, extruders_pos[thread, ext_n, 1]] == ext_n+1:
        extr_occupied[thread, extruders_pos[thread, ext_n, 1]] = 0

    extruders_pos[thread, ext_n, 1] = -1
    extruders_pos[thread, ext_n, 0] = -1

    # extruders_topo_leg[thread, ext_n, 0] = -1
    # extruders_topo_leg[thread, ext_n, 1] = -1

    extruders_direction[thread, ext_n] = -1

def cpu_unload_extruders(
    thread, 
    rnap_occupied,
    extr_occupied,
    extruders_pos,
    extruders_topo_leg,
    extruders_direction,
    rng_states,
    extruder_unload_prob,
    rnap_remove_extruder_prob,
):

    for ext_n in range(extruders_pos.shape[1]):
        if extruders_pos[thread, ext_n, 1] != -1:
            life_time_end = extruder_unload_prob[thread, ext_n] > random.random()
            legs_too_close = abs(extruders_pos[thread, ext_n, -1] - extruders_pos[thread, ext_n, 0]) <= 1

            if life_time_end or legs_too_close:
                cpu_unload_one_extruder(
                    thread, 
                    ext_n,
                    extr_occupied,
                    extruders_pos,
                    extruders_topo_leg,
                    extruders_direction,
                )
    return True

def cpu_extruder_change_direction(
    thread,
    extruders_pos,
    extruder_change_direction_prob,
    extruders_direction,
    rng_states,
):
    for ext_n in range(extruders_pos.shape[1]):
        if extruders_pos[thread, ext_n, -1] != -1:
            if extruder_change_direction_prob > random.random():
                extruders_direction[thread, ext_n] = abs(extruders_direction[thread, ext_n] - 1)

def cpu_load_one_extruder(
    thread, 
    ext_n,
    coord,
    extr_occupied,
    extruders_pos,
    extruders_topo_leg,
    extruders_direction,
    rng_states,
    topology_state,
):
    extruders_pos[thread, ext_n, 1] = coord + 1
    extruders_pos[thread, ext_n, 0] = coord

    if 0.5 > random.random():
        extruders_topo_leg[thread, ext_n, 0] = topology_state[0]
        extruders_topo_leg[thread, ext_n, 1] = topology_state[1]
    else:
        extruders_topo_leg[thread, ext_n, 0] = topology_state[1]
        extruders_topo_leg[thread, ext_n, 1] = topology_state[0]

    extruders_direction[thread, ext_n] = int(0.5 > random.random())

    extr_occupied[thread, extruders_pos[thread, ext_n, 1]] = ext_n+1
    extr_occupied[thread, extruders_pos[thread, ext_n, 0]] = -(ext_n+1)

def cpu_load_extruders(
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
):

    for ext_n in range(extruders_pos.shape[1]):
        if extruders_pos[thread, ext_n, -1] == -1:
            if extruder_load_prob > random.random():
                for _ in range(1000):
                    coord = max(int(float(extr_occupied.shape[1]-2) * random.random()), 1)
                    empty_place = (extr_occupied[thread, coord] == 0) 
                    empty_place = empty_place and (extr_occupied[thread, coord+1] == 0)
                    empty_place = empty_place and (coh_occupied[thread, coord] == 0)
                    empty_place = empty_place and (coh_occupied[thread, coord+1] == 0)
                    empty_place = empty_place and (rnap_occupied[thread, coord] == 0)
                    empty_place = empty_place and (rnap_occupied[thread, coord+1] == 0)
                    empty_place = empty_place and (not test_dna_end(coord, extr_occupied.shape[1]))
                    empty_place = empty_place and (not test_dna_end(coord+1, extr_occupied.shape[1]))
                    if empty_place:
                        cpu_load_one_extruder(
                            thread, 
                            ext_n,
                            coord,
                            extr_occupied,
                            extruders_pos,
                            extruders_topo_leg,
                            extruders_direction,
                            rng_states,
                            topology_state,
                        )
                        break
    return True

def test_cohesin_absence(
    thread,
    test_pos,
    coh_occupied,
    extr_coh_interaction,
):
    if extr_coh_interaction:
        return coh_occupied[thread, test_pos] == 0
    else:
        return True

def check_rnap_interaction(
    thread,
    test_pos,
    rnap_occupied,
    extruder_rnap_interaction
):
    """
    Проверяет взаимодействие экструдера с РНК-полимеразой на заданной позиции.
    Возвращает True, если экструдер может двигаться, иначе False.
    
    Parameters:
    -----------
    thread : int
        Номер потока
    test_pos : int
        Позиция для проверки
    rnap_occupied : array
        Массив занятых РНК-полимеразой позиций
    extruder_rnap_interaction : bool
        Флаг, включающий взаимодействие экструдера и РНК-полимеразы
        
    Returns:
    --------
    bool
        True если движение разрешено, False если запрещено
    """
    if extruder_rnap_interaction:
        return rnap_occupied[thread, test_pos] == 0
    else:
        return True

def cpu_translocate_extruder(
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
    extruder_rnap_interaction=True,
    smc_rnap_step_prob=0.0,
):
    for ext_n in range(extruders_pos.shape[1]):
        if extruders_pos[thread, ext_n, -1] != -1:
            leg = extruders_direction[thread, ext_n]
            step = leg*2 - 1
            is_topology = extruders_topo_leg[thread, ext_n, leg]

            valid_steps = min(min(extr_occupied.shape[1] - 1 - extruders_pos[thread, ext_n, leg], extruders_pos[thread, ext_n, leg] - 1), 3)
            
            k = 0
            n_max_steps = -1
            for i in range(valid_steps,-1,-1):
                if test_cohesin_absence(thread, extruders_pos[thread, ext_n, leg] + step*i, coh_occupied, extr_coh_interaction) and \
                    check_rnap_interaction(thread, extruders_pos[thread, ext_n, leg] + step*i, rnap_occupied, extruder_rnap_interaction) and \
                    (extr_occupied[thread, extruders_pos[thread, ext_n, leg] + step*i] == 0):
                    n_max_steps = i
                    break

            if n_max_steps == 0:
                pass

            if n_max_steps == 1:
                k = 1

            if n_max_steps == 2:
                prob = 1.0
                if extruder_rnap_interaction:
                    rnap_n = rnap_occupied[thread, extruders_pos[thread, ext_n, leg] + step] - 1
                    if rnap_n >= 0:
                        prob = prob * ext_step_over_rnap_prob(
                            thread,
                            rnap_n,
                            rnap_pos,
                            rnap_bristle,
                            rnap_occupied,
                            rna_bead_params,
                            is_topology,
                        )

                if (extr_occupied[thread, extruders_pos[thread, ext_n, leg] + step] != 0) or \
                    (not test_cohesin_absence(thread, extruders_pos[thread, ext_n, leg] + step, coh_occupied, extr_coh_interaction)):
                    prob = prob * 5e-2
                if prob > random.random():
                    k = 2
            if n_max_steps == 3:
                prob = 1.0
                for tmp_step_k in range(1,3):
                    if extruder_rnap_interaction:
                        rnap_n = rnap_occupied[thread, extruders_pos[thread, ext_n, leg] + step*tmp_step_k] - 1
                        if rnap_n >= 0:
                            prob = prob * ext_step_over_rnap_prob(
                                            thread,
                                            rnap_n,
                                            rnap_pos,
                                            rnap_bristle,
                                            rnap_occupied,
                                            rna_bead_params,
                                            is_topology,
                                        )

                    if (extr_occupied[thread, extruders_pos[thread, ext_n, leg] + step*tmp_step_k] != 0) or \
                        (not test_cohesin_absence(thread, extruders_pos[thread, ext_n, leg] + step*tmp_step_k, coh_occupied, extr_coh_interaction)):
                        prob = prob * 5e-2
                if prob > random.random():
                    k = 3

            if (k > 0) and test_cohesin_absence(thread, extruders_pos[thread, ext_n, leg] + step*k, coh_occupied, extr_coh_interaction) and \
                check_rnap_interaction(thread, extruders_pos[thread, ext_n, leg] + step*k, rnap_occupied, extruder_rnap_interaction) and \
                (extr_occupied[thread, extruders_pos[thread, ext_n, leg] + step*k] == 0) and \
                (not test_chromatid_change(extruders_pos[thread, ext_n, leg], extruders_pos[thread, ext_n, leg] + step*k, extr_occupied.shape[1])) and \
                (not test_dna_end(extruders_pos[thread, ext_n, leg] + step*k, extr_occupied.shape[1])):
                if extr_occupied[thread, extruders_pos[thread, ext_n, leg]] == (ext_n+1) * (2*leg-1):
                    extr_occupied[thread, extruders_pos[thread, ext_n, leg]] = 0       
                extr_occupied[thread, extruders_pos[thread, ext_n, leg] + step*k] = (ext_n+1) * (2*leg-1)
                extruders_pos[thread, ext_n, leg] += step*k

            # Перемещаем другую ногу экструдера диффузией на одну клетку, если там свободно
            other_leg = 1 - leg
            step_dif = 1 if random.random() > 0.5 else -1
            target_pos = extruders_pos[thread, ext_n, other_leg] + 2*step_dif

            if (not test_chromatid_change(extruders_pos[thread, ext_n, other_leg], target_pos, extr_occupied.shape[1])) and \
                (not test_dna_end(target_pos, extr_occupied.shape[1])):

                can_step = (coh_occupied[thread, target_pos] == 0) and \
                    (rnap_occupied[thread, target_pos] == 0) and \
                    (extr_occupied[thread, target_pos] == 0) and \
                    not (test_dna_end(target_pos, extr_occupied.shape[1]))
                
                if can_step:
                    can_overstep = (coh_occupied[thread, target_pos - step_dif] == 0) and \
                                    (extr_occupied[thread, target_pos - step_dif] == 0) and \
                                    not (test_dna_end(target_pos - step_dif, extr_occupied.shape[1]))
                    if rnap_occupied[thread, target_pos - step_dif] != 0:
                        if not (random.random() < smc_rnap_step_prob):
                            can_overstep = False
                
                    if can_overstep:
                        extr_occupied[thread, target_pos] = (2*other_leg-1) * (ext_n + 1)
                        extr_occupied[thread, extruders_pos[thread, ext_n, other_leg]] = 0
                        extruders_pos[thread, ext_n, other_leg] += 2*step_dif
                else:
                    if (coh_occupied[thread, target_pos - step_dif] == 0) and \
                        (extr_occupied[thread, target_pos - step_dif] == 0) and \
                        (rnap_occupied[thread, target_pos - step_dif] == 0) and \
                        not (test_dna_end(target_pos - step_dif, extr_occupied.shape[1])):

                        extr_occupied[thread, target_pos - step_dif] = (2*other_leg-1) * (ext_n + 1)
                        extr_occupied[thread, extruders_pos[thread, ext_n, other_leg]] = 0
                        extruders_pos[thread, ext_n, other_leg] += step_dif
    return True

def extruder_step_nontopology(
    thread,
    ext_n,
    leg,
    extruder_position,
    target_position,
    coh_occupied,
    extr_occupied,
    cohesins_pos,
    extruders_pos,
):
    if (not test_chromatid_change(extruder_position, target_position, extr_occupied.shape[1])) and (not test_dna_end(target_position, extr_occupied.shape[1])):
        if (coh_occupied[thread, target_position] == 0) and (extr_occupied[thread, target_position] == 0):
            extr_occupied[thread, extruder_position] = 0
            extr_occupied[thread, target_position] = (ext_n+1) * (2*leg-1)
            extruders_pos[thread, ext_n, leg] = target_position

        elif extr_occupied[thread, target_position] != 0:
            tmp_ext_n = abs(extr_occupied[thread, target_position]) - 1
            tmp_leg = 1 if extr_occupied[thread, target_position] > 0 else 0

            extruders_pos[thread, ext_n, leg] = target_position               # put extruder into target position
            extr_occupied[thread, target_position] = (ext_n+1) * (2*leg-1)

            extruders_pos[thread, tmp_ext_n, tmp_leg] = extruder_position     # put extruder from target position into exruder position
            extr_occupied[thread, extruder_position] = (tmp_ext_n+1) * (2*tmp_leg-1)

        elif coh_occupied[thread, target_position] != 0:
            tmp_coh_n = abs(coh_occupied[thread, target_position]) - 1
            tmp_leg = 1 if coh_occupied[thread, target_position] > 0 else 0

            extr_occupied[thread, target_position] = (ext_n+1) * (2*leg-1)   # put extruder into target position
            extr_occupied[thread, extruder_position] = 0
            extruders_pos[thread, ext_n, leg] = target_position
                
            coh_occupied[thread, extruder_position] = (tmp_coh_n+1) * (2*tmp_leg-1)  # put cohesin from target position into exruder position
            coh_occupied[thread, target_position] = 0
            cohesins_pos[thread, tmp_coh_n, tmp_leg] = extruder_position
            

def extruder_push_coh_legs_linear(
    thread, 
    start_position,
    direction,
    rnap_occupied,
    coh_occupied,
    extr_occupied,
    cohesins_pos,
):
    n = 0
    pos = start_position
    while coh_occupied[thread, pos] != 0:
        no_barier = (rnap_occupied[thread, pos + direction] == 0)
        no_barier = no_barier and (extr_occupied[thread, pos + direction] == 0)
        no_barier = no_barier and (not test_dna_end(pos + direction, coh_occupied.shape[1]))
        if  no_barier:
            n += 1
            pos += direction
        else:
            return False

    for _ in range(n):
        if coh_occupied[thread, pos - direction] != 0:
            k = abs(coh_occupied[thread, pos - direction]) - 1
            leg = 1 if coh_occupied[thread, pos - direction] > 0 else 0
            coh_occupied[thread, cohesins_pos[thread, k, leg] + direction] = (k+1) * (2*leg-1)
            coh_occupied[thread, cohesins_pos[thread, k, leg]] = 0
            cohesins_pos[thread, k, leg] += direction

        pos -= direction
    
    return True

def ext_step_over_rnap_prob(
    thread,
    rnap_n,
    rnap_pos,
    rnap_bristle,
    rnap_occupied,
    rna_bead_params,
    is_topology,
):
    global PFN
    if 'prob_function_polymer' == PFN:
        return prob_function_polymer(
                thread,
                rnap_n,
                rnap_pos,
                rnap_bristle,
                rnap_occupied,
                rna_bead_params,
                is_topology,
            )
    elif 'prob_function_step' == PFN:
        return prob_function_step(
                thread,
                rnap_n,
                rnap_pos,
                rnap_bristle,
                rnap_occupied,
                rna_bead_params,
                is_topology,
            )
    elif 'prob_function_passive' == PFN: 
        return prob_function_passive(
                thread,
                rnap_n,
                rnap_pos,
                rnap_bristle,
                rnap_occupied,
                rna_bead_params,
                is_topology,
            )
    elif 'prob_function_one_sided' == PFN: 
        return prob_function_one_sided(
                is_topology,
            )
    else:
        return prob_function_stable_prob(
            is_topology,
        )