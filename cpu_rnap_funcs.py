import numpy as np
import random
from cpu_cohesin_funcs import test_dna_end
from cpu_extruder_funcs import *

def rnap_elongation(
    thread,
    rnap_n,
    gene_n,
    timestep,
    elong_vel,
    rnap_bristle,
    rnap_curr_intron,
    introns,
):
    rnap_bristle[thread, rnap_n, 0] += elong_vel * timestep
    rnap_bristle[thread, rnap_n, 1] += elong_vel * timestep
    intron_n = rnap_curr_intron[thread, rnap_n]
    if introns[gene_n].shape[0] > intron_n:
        if introns[gene_n][intron_n][0] < rnap_bristle[thread, rnap_n, 1]:
            rnap_bristle[thread, rnap_n, 1] -= introns[gene_n][intron_n][1]
            rnap_curr_intron[thread, rnap_n] += 1

def rnap_do_step(
    thread, 
    rnap_n,
    direction,
    rnap_pos,
    rnap_state,
    rnap_occupied,
):
    pos = int(rnap_pos[thread, rnap_n])
    if (int(rnap_state[thread, rnap_n]) == 2) and (rnap_occupied[thread, pos + int(direction)] == 0):
        rnap_occupied[thread, pos] = 0
        rnap_pos[thread, rnap_n] += direction
        pos = int(rnap_pos[thread, rnap_n])
        rnap_occupied[thread, pos] = rnap_n + 1
        rnap_state[thread, rnap_n] = 1 # set state eq "elongation"
        return True
    else:
        return False

def rnap_time_step(
    thread, 
    rnap_n,
    gene_n,
    timestep,
    bead_size,
    elong_vel,
    rnap_pos,
    rnap_bristle,
    rnap_state,
    rnap_curr_intron,
    rnap_occupied,
    genes,
    introns,
):
    if int(rnap_state[thread, rnap_n]) == 1: # is state eq "elongation"
        rnap_elongation(
            thread, 
            rnap_n,
            gene_n,
            timestep,
            elong_vel,
            rnap_bristle,
            rnap_curr_intron,
            introns,
        )
        if rnap_bristle[thread, rnap_n, 0] >= bead_size:
            rnap_bristle[thread, rnap_n, 0] -= bead_size
            if int(rnap_pos[thread, rnap_n]) == int(genes[gene_n][2]):
                rnap_state[thread, rnap_n] = 0 # set state eq "unload"
            else:
                rnap_state[thread, rnap_n] = 2 # set state eq "do_step"

    stepped = False
    if rnap_state[thread, rnap_n] == 2:
        direction = genes[gene_n][1]
        stepped = rnap_do_step(
            thread, 
            rnap_n,
            int(direction),
            rnap_pos,
            rnap_state,
            rnap_occupied,
        )
    return stepped

def cpu_translocate_rnap(
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
):
  
    for rnap_n in range(rnap_state.shape[1]):
        if rnap_state[thread, rnap_n] > 0:
            gene_n = rnap_gene[thread, rnap_n]
            bead_size = bead_sizes[rnap_pos[thread, rnap_n]]
            stepped = rnap_time_step(
                thread, 
                rnap_n,
                gene_n,
                timestep,
                bead_size,
                elong_vel,
                rnap_pos,
                rnap_bristle,
                rnap_state,
                rnap_curr_intron,
                rnap_occupied,
                genes,
                introns,
            )

            if stepped:
                direction = genes[gene_n][1]
                pos = int(rnap_pos[thread, rnap_n])
                rnap_push_smc_legs_linear(
                    thread, 
                    pos,
                    int(direction),
                    rnap_occupied,
                    coh_occupied,
                    extr_occupied,
                    extruders_pos,
                    extruders_topo_leg,
                    extruders_direction,
                    cohesins_pos,
                    rng_states,
                    rnap_remove_extruder_prob,
                )


def cpu_unload_rnap(
    thread, 
    rnap_pos,
    rnap_bristle,
    rnap_state,
    rnap_curr_intron,
    rnap_occupied,
    rnap_gene,
):
    for rnap_n in range(rnap_state.shape[1]):
        if rnap_state[thread, rnap_n] == 0: # is state eq "unload" 
            rnap_occupied[thread, int(rnap_pos[thread, rnap_n])] = 0

            rnap_pos[thread, rnap_n] = -1
            rnap_bristle[thread, rnap_n, 0] = -1.0
            rnap_bristle[thread, rnap_n, 1] = -1.0
            rnap_state[thread, rnap_n] = -1 # set state eq "floating"
            rnap_curr_intron[thread, rnap_n] = -1
            rnap_gene[thread, rnap_n] = -1

def cpu_load_rnap(
    thread, 
    rnap_pos,
    rnap_bristle,
    rnap_state,
    rnap_curr_intron,
    rnap_occupied,
    rnap_gene,
    genes,
    rng_states,
):
    for gene_n in range(len(genes)):
        gene_start = int(genes[gene_n][0])
        prob = genes[gene_n][3]
        if (rnap_occupied[thread, gene_start] == 0) and (random.random() < prob):
            for rnap_n in range(rnap_state.shape[1]):
                if rnap_state[thread, rnap_n] == -1: # is state eq "floating"
                    rnap_pos[thread, rnap_n] = gene_start
                    rnap_bristle[thread, rnap_n, 0] = 0.0
                    rnap_bristle[thread, rnap_n, 1] = 0.0
                    rnap_state[thread, rnap_n] = 1 # set state eq "elongation"
                    rnap_curr_intron[thread, rnap_n] = 0
                    rnap_occupied[thread, gene_start] = rnap_n+1
                    rnap_gene[thread, rnap_n] = gene_n

                    break
    return True

def rnap_push_smc_legs_linear(
    thread, 
    start_position,
    direction,
    rnap_occupied,
    coh_occupied,
    extr_occupied,
    extruders_pos,
    extruders_topo_leg,
    extruders_direction,
    cohesins_pos,
    rng_states,
    rnap_remove_extruder_prob,
):
    n = 0
    pos = start_position
    while (coh_occupied[thread, pos] != 0) or (extr_occupied[thread, pos] != 0):
        no_barier = (not test_dna_end(pos + direction, coh_occupied.shape[1]))
        if no_barier:
            n += 1
            pos += direction
        else:
            return False


    for _ in range(n):
        if extr_occupied[thread, pos - direction] != 0:
            k = abs(extr_occupied[thread, pos - direction]) - 1
            leg = 1 if extr_occupied[thread, pos - direction] > 0 else 0
            extr_occupied[thread, extruders_pos[thread, k, leg] + direction] = (k+1) * (2*leg-1)
            extr_occupied[thread, extruders_pos[thread, k, leg]] = 0
            extruders_pos[thread, k, leg] += direction

        if coh_occupied[thread, pos - direction] != 0:
            k = abs(coh_occupied[thread, pos - direction]) - 1
            leg = 1 if coh_occupied[thread, pos - direction] > 0 else 0
            coh_occupied[thread, cohesins_pos[thread, k, leg] + direction] = (k+1) * (2*leg-1)
            coh_occupied[thread, cohesins_pos[thread, k, leg]] = 0
            cohesins_pos[thread, k, leg] += direction

        pos -= direction

    pos = start_position
    if (coh_occupied[thread, pos - direction] != 0) or (extr_occupied[thread, pos - direction] != 0):
        if extr_occupied[thread, pos - direction] != 0:
            k = abs(extr_occupied[thread, pos - direction]) - 1
            leg = 1 if extr_occupied[thread, pos - direction] > 0 else 0
            extr_occupied[thread, extruders_pos[thread, k, leg] + direction] = (k+1) * (2*leg-1)
            extr_occupied[thread, extruders_pos[thread, k, leg]] = 0
            extruders_pos[thread, k, leg] += direction

        if coh_occupied[thread, pos - direction] != 0:
            k = abs(coh_occupied[thread, pos - direction]) - 1
            leg = 1 if coh_occupied[thread, pos - direction] > 0 else 0
            coh_occupied[thread, cohesins_pos[thread, k, leg] + direction] = (k+1) * (2*leg-1)
            coh_occupied[thread, cohesins_pos[thread, k, leg]] = 0
            cohesins_pos[thread, k, leg] += direction

    return True