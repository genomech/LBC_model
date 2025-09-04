import numpy as np
import random

def test_dna_end(pos, N):
    if (pos >= N-1) or (pos <= 0) or (pos == N//2) or (pos == (N//2) - 1):
        return True
    else:
        return False
    
def test_chromatid_change(pos, target_pos, N):
    if pos > N//2:
        if target_pos <= N//2:
            return True
        else:
            return False
    elif pos < (N//2)-1:
        if target_pos >= (N//2)-1:
            return True
        else:
            return False
    else:
        return True

def cpu_translocate_cohesin(
    thread, 
    diff_prob,
    rnap_occupied,
    coh_occupied,
    extr_occupied,
    cohesins_pos,
    rng_states,
    smc_rnap_step_prob=0.0,
):
    ### Difusion
    for coh_n in range(cohesins_pos.shape[1]):
        for leg in (0,1):
            if random.random() < diff_prob:
                if random.random()  > 0.5:
                    step_dif = 1
                else:
                    step_dif = -1
                
                target_pos = cohesins_pos[thread, coh_n, leg] + 2*step_dif
                if (not test_chromatid_change(cohesins_pos[thread, coh_n, leg], target_pos, coh_occupied.shape[1])) and \
                    (not test_dna_end(target_pos, coh_occupied.shape[1])):
                    can_step = (coh_occupied[thread, target_pos] == 0) and \
                        (rnap_occupied[thread, target_pos] == 0) and \
                        (extr_occupied[thread, target_pos] == 0) and \
                        not (test_dna_end(target_pos, coh_occupied.shape[1]))
                    
                    if can_step:
                        can_overstep = (coh_occupied[thread, target_pos - step_dif] == 0) and \
                                        (extr_occupied[thread, target_pos - step_dif] == 0) and \
                                        not (test_dna_end(target_pos - step_dif, coh_occupied.shape[1]))
                        if rnap_occupied[thread, target_pos - step_dif] != 0:
                            if not (random.random() < smc_rnap_step_prob):
                                can_overstep = False
                    
                        if can_overstep:
                            coh_occupied[thread, target_pos] = (2*leg-1) * (coh_n + 1)
                            coh_occupied[thread, cohesins_pos[thread, coh_n, leg]] = 0
                            cohesins_pos[thread, coh_n, leg] += 2*step_dif
                    else:
                        if (coh_occupied[thread, target_pos - step_dif] == 0) and \
                            (extr_occupied[thread, target_pos - step_dif] == 0) and \
                            (rnap_occupied[thread, target_pos - step_dif] == 0) and \
                            not (test_dna_end(target_pos - step_dif, coh_occupied.shape[1])):

                            coh_occupied[thread, target_pos - step_dif] = (2*leg-1) * (coh_n + 1)
                            coh_occupied[thread, cohesins_pos[thread, coh_n, leg]] = 0
                            cohesins_pos[thread, coh_n, leg] += step_dif

    return True