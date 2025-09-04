import h5py 
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.hdf5_format import HDF5Reporter

from chicken_loop_3D import bondUpdater

import shutil
import os
import json
import shutil
import traceback

import time

from scripts import *
from scripts_cpu import *
from side_functions import *
from multiprocessing import Pool

import argparse


start_oocyte_dict = {
}


def multiproc_out_saver(out_to_save, value_id, k, batch_to_save):
    tmp_out = []
    for i in range(batch_to_save):
        tmp_out += [out_to_save[k*batch_to_save+i][value_id][0, :]]
    return tmp_out

# Преобразуем геномные координаты в координаты бусин
def genomic_to_beads(position, chosen_slice, bead_sizes):
    # Получаем начало моделируемого участка
    region_start = chosen_slice[1]
    # Вычитаем начало региона для получения относительной позиции
    relative_pos = position - region_start
    # Находим соответствующую бусину, используя массив размеров бусин
    bead_idx = 0
    accumulated_size = 0
    for i, size in enumerate(bead_sizes):
        if accumulated_size + size > relative_pos:
            return bead_idx
        accumulated_size += size
        bead_idx += 1
    return bead_idx


# cuda.select_device(0)

# device = cuda.get_current_device()


def main(command_line = None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-g", "--gpu_id", help="gpu id", type=int, required=True)
    parser.add_argument("-N", "--total_sims", help="total number of simulations", type=int, required=True)
    parser.add_argument("-c", "--chrom", help="chromosome name for modeling", type=str, required=True)
    parser.add_argument("-n", "--start_number", help="initiate 3D simulation from oocyte number n", type=int, required=False, default=0)
    parser.add_argument("-o", "--one_D", help="initiate 1D simulation", action='store_true')
    parser.add_argument("-t", "--three_D", help="initiate 3D simulation", action='store_true')
    parser.add_argument("-p", "--pref_name", help="name prefix", type=str, required=False, default='')
    parser.add_argument("--rnap_sep_type", help="rnap separator calculation type (one of the [old, mult, new])", type=str, required=False, default='mult')

    args = parser.parse_args(command_line)

    gpu_id = args.gpu_id
    nproc = args.total_sims
    chrom_for_modeling = args.chrom
    calculate_1D = args.one_D
    calculate_3D = args.three_D
    start_oo = args.start_number
    name_prefix = args.pref_name
    rnap_sep_type = args.rnap_sep_type
    
    batch_to_save = nproc
    restart_simulation = nproc
    start_seed = 2500 # 1007

    rna_seq_dir = './files'
    genes_df = gtf_reader(f'{rna_seq_dir}/valid_transcripts.gtf', transcripts_only=True)
    full_df =  pd.read_csv(f'{rna_seq_dir}/full.tsv', sep='\t')

    fpkm_lbc_intron = pd.concat([pd.read_csv(f'{rna_seq_dir}/LBC_intron_cov.forward.bed', sep='\t', header = None),
                        pd.read_csv(f'{rna_seq_dir}/LBC_intron_cov.reverse.bed', sep='\t', header = None)],
                        ignore_index=True)
    with open(f'{rna_seq_dir}/GV1-LBC_RNA_S267.stats', 'r') as file:
        depth_lbc_intron = int(file.readline()[:-1])
    fpkm_lbc_intron['len'] = fpkm_lbc_intron[2] - fpkm_lbc_intron[1]
    fpkm_lbc_intron = fpkm_lbc_intron.groupby(3)[['len', 6]].sum()
    fpkm_lbc_intron['FPKM_intron'] = fpkm_lbc_intron[6] * 10**9 / (fpkm_lbc_intron['len'] * depth_lbc_intron)
    fpkm_lbc_intron = fpkm_lbc_intron['FPKM_intron'].to_dict()


    fpkm_lbc_exon = pd.concat([pd.read_csv(f'{rna_seq_dir}/LBC_exon_cov.forward.bed', sep='\t', header = None),
                        pd.read_csv(f'{rna_seq_dir}/LBC_exon_cov.reverse.bed', sep='\t', header = None)],
                        ignore_index=True)
    with open(f'{rna_seq_dir}/GV1-LBC_RNA_S267.stats', 'r') as file:
        depth_lbc_exon = int(file.readline()[:-1])
    fpkm_lbc_exon['len'] = fpkm_lbc_exon[2] - fpkm_lbc_exon[1]
    fpkm_lbc_exon = fpkm_lbc_exon.groupby(3)[['len', 6]].sum()
    fpkm_lbc_exon['FPKM_exon'] = fpkm_lbc_exon[6] * 10**9 / (fpkm_lbc_exon['len'] * depth_lbc_exon)
    fpkm_lbc_exon = fpkm_lbc_exon['FPKM_exon'].to_dict()

    fpkm_postlbc_intron = pd.concat([pd.read_csv(f'{rna_seq_dir}/postLBC_intron_cov.forward.bed', sep='\t', header = None),
                        pd.read_csv(f'{rna_seq_dir}/postLBC_intron_cov.reverse.bed', sep='\t', header = None)],
                        ignore_index=True)
    with open(f'{rna_seq_dir}/GV1-large_RNA_S268.stats', 'r') as file:
        depth_postlbc_intron = int(file.readline()[:-1])
    fpkm_postlbc_intron['len'] = fpkm_postlbc_intron[2] - fpkm_postlbc_intron[1]
    fpkm_postlbc_intron = fpkm_postlbc_intron.groupby(3)[['len', 6]].sum()
    fpkm_postlbc_intron['FPKM_intron'] = fpkm_postlbc_intron[6] * 10**9 / (fpkm_postlbc_intron['len'] * depth_postlbc_intron)
    fpkm_postlbc_intron = fpkm_postlbc_intron['FPKM_intron'].to_dict()

    fpkm_postlbc_exon = pd.concat([pd.read_csv(f'{rna_seq_dir}/postLBC_exon_cov.forward.bed', sep='\t', header = None),
                        pd.read_csv(f'{rna_seq_dir}/postLBC_exon_cov.reverse.bed', sep='\t', header = None)],
                        ignore_index=True)
    with open(f'{rna_seq_dir}/GV1-large_RNA_S268.stats', 'r') as file:
        depth_postlbc_exon = int(file.readline()[:-1])
    fpkm_postlbc_exon['len'] = fpkm_postlbc_exon[2] - fpkm_postlbc_exon[1]
    fpkm_postlbc_exon = fpkm_postlbc_exon.groupby(3)[['len', 6]].sum()
    fpkm_postlbc_exon['FPKM_exon'] = fpkm_postlbc_exon[6] * 10**9 / (fpkm_postlbc_exon['len'] * depth_postlbc_exon)
    fpkm_postlbc_exon = fpkm_postlbc_exon['FPKM_exon'].to_dict()

    genes_df['len'] = genes_df['end'] - genes_df['start']
    genes_df['FPKM_intron_LBC'] = genes_df['transcript_id'].apply(lambda x:
                fpkm_lbc_intron[x] if x in fpkm_lbc_intron.keys() else None 
                                                          )
    genes_df['FPKM_intron_postLBC'] = genes_df['transcript_id'].apply(lambda x:
                fpkm_postlbc_intron[x] if x in fpkm_postlbc_intron.keys() else None  
                                                          )
    genes_df['FPKM_exon_LBC'] = genes_df['transcript_id'].apply(lambda x:
                fpkm_lbc_exon[x] if x in fpkm_lbc_exon.keys() else None  
                                                          )
    genes_df['FPKM_exon_postLBC'] = genes_df['transcript_id'].apply(lambda x:
                fpkm_postlbc_exon[x] if x in fpkm_postlbc_exon.keys() else None  
                                                          )

    genes_df['min_FPKM'] = genes_df[['FPKM_intron_LBC', 'FPKM_exon_LBC']].apply(lambda x:
                    x[0] if not np.isnan(x[0]) else x[1]
                                                            , axis = 1)

    genes_df['min_FPKM_postLBC'] = genes_df[['FPKM_intron_postLBC', 'FPKM_exon_postLBC']].apply(lambda x:
                    x[0] if not np.isnan(x[0]) else x[1]
                                                            , axis = 1)

    if rnap_sep_type == 'old':
        mask = (genes_df['type'] == 'transcript') & (genes_df['min_FPKM'] > 0)
        lib_depth = (genes_df[mask]['cov']*genes_df[mask]['len']).sum()
        min_rnap_sep = 45 # bp
        frag_len = 151
        
        fpkm_thresh = np.percentile(np.nan_to_num(genes_df[mask]['min_FPKM']), 99) # 99.5
        max_fpkm_i = genes_df[genes_df['min_FPKM'] == genes_df[(genes_df['min_FPKM'] < fpkm_thresh)]['min_FPKM'].max()].index[0]
        k_rnap_sep = min_rnap_sep * genes_df.loc[max_fpkm_i, 'min_FPKM']
        genes_df['rnap_sep_min_FPKM'] = genes_df['min_FPKM'].apply(lambda x:
                                                        1 / x * k_rnap_sep if x > 0 else 1e10)

        genes_df['rnap_sep_min_FPKM_late'] = genes_df['min_FPKM_postLBC'].apply(lambda x: 
                                                        1 / x * k_rnap_sep if x > 0 else 1e10)
    elif rnap_sep_type == 'mult':
        per = 95
        late_min_dist_mult = 100
        
        # Вычисляем расстояние для LBC
        mask = (genes_df['type'] == 'transcript') & (genes_df['min_FPKM'] > 0)
        lib_depth = (genes_df[mask]['cov']*genes_df[mask]['len']).sum()
        min_rnap_sep = 45 # bp
        frag_len = 151

        fpkm_thresh = np.percentile(np.nan_to_num(genes_df[mask]['min_FPKM']), per)
        max_fpkm_i = genes_df[genes_df['min_FPKM'] == genes_df[(genes_df['min_FPKM'] < fpkm_thresh)]['min_FPKM'].max()].index[0]
        k_rnap_sep = min_rnap_sep * genes_df.loc[max_fpkm_i, 'min_FPKM']
        genes_df['rnap_sep_min_FPKM'] = genes_df['min_FPKM'].apply(lambda x:
                                                        1 / x * k_rnap_sep if x > 0 else 1e10)

        # Вычисляем расстояние для postLBC
        min_rnap_sep_late = 45*late_min_dist_mult # bp
        mask = (genes_df['type'] == 'transcript') & (genes_df['min_FPKM_postLBC'] > 0)
        fpkm_thresh = np.percentile(genes_df[mask]['min_FPKM_postLBC'], per)
        max_fpkm_i = genes_df[genes_df['min_FPKM_postLBC'] == genes_df[(genes_df['min_FPKM_postLBC'] < fpkm_thresh)]['min_FPKM_postLBC'].max()].index[0]
        k_rnap_sep = min_rnap_sep_late * genes_df.loc[max_fpkm_i, 'min_FPKM_postLBC']
        genes_df['rnap_sep_min_FPKM_late'] = genes_df['min_FPKM_postLBC'].apply(lambda x: 
                                                        1 / x * k_rnap_sep if x > 0 else 1e10)
    elif rnap_sep_type == 'new':
        k_rnap_sep = 65.0
        k_rnap_sep_late = 1200.0
        
        min_rnap_sep = 45 # bp
        mask = (genes_df['type'] == 'transcript') & (genes_df['min_FPKM'] > 0)
        genes_df['rnap_sep_min_FPKM'] = genes_df[['min_FPKM', 'len']].apply(lambda x:
                                                        1 / x[0] * k_rnap_sep + min_rnap_sep if x[0] > 0 else 10*x[1], axis=1)

        min_rnap_sep_late = 45 # bp
        mask = (genes_df['type'] == 'transcript') & (genes_df['min_FPKM_postLBC'] > 0)
        genes_df['rnap_sep_min_FPKM_late'] = genes_df[['min_FPKM_postLBC', 'len']].apply(lambda x:
                                                        1 / x[0] * k_rnap_sep_late + min_rnap_sep_late if x[0] > 0 else 10*x[1], axis=1)
    else:
        assert False, f'Error, the rnap_sep_type = {rnap_sep_type} is not one of the [old, mult, new]'


    if chrom_for_modeling == 'chr1':
        chosen_slice_str = 'chr1:053,500,000-65,500,000'
    elif chrom_for_modeling == 'chr11':
        chosen_slice_str = 'chr11:000,000,000-19,822,608'
    elif chrom_for_modeling == 'chr1p100':
        chosen_slice_str = 'chr1:053,500,000-65,500,000'
    else:
        assert False, f'Error, the {chrom_for_modeling} is not one of the chr1 or chr11'

    for prob, ext_coh_int, ext_rnap_int, late_add in zip(
        [1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 1.0][gpu_id::4],
        [True, True, True, True, False, False, False, False][gpu_id::4],
        [True, True, True, True, True, True, True, True][gpu_id::4],
        ['', '', '_late', '_late', '', '', '_late', '_late'][gpu_id::4],
    ):
        for _ in ['']:
            ls1, lt1, ls2, lt2 = 200e3, 2e6, 50e3, 500e3
            args = {
            'BEADSIZE_GENE': 45, # bp
            'BEADSIZE_BASE': 45*5, # bp
            'RNAP_VEL': 100.0, # bp/s
            'COND_VEL': 675, # bp/s
            'RNAP_REM_COND_PROB': 0.0, # 1e-2,
            'BOUNDTIME': 1e0,
            'Diffusion_prob': 1e0,
            'Cohesin_separator': 100e3,
            }
            args['ave_loop_size'] = ls1
            args['LIFETIME_BP'] = lt1
            args['ave_loop_size2'] = ls2
            args['LIFETIME_BP2'] = lt2

            args['TIMESTEP'] = 1/5 # sec, or one condensin step, extruder vel ~200 nm/s and one step ~40 nm
            args['LIFETIME'] = (args['LIFETIME_BP'] * 0.33 / 5) / (args['TIMESTEP'] * 200) # in TIMESTEPs or just STEPs
            args['LIFETIME2'] = (args['LIFETIME_BP2'] * 0.33 / 5) / (args['TIMESTEP'] * 200) # in TIMESTEPs or just STEPs
            args['LIFETIME_STALLED'] = args['LIFETIME']
            args['EXTR_CHNG_DIR_PROB'] = 1.0
            args['Topology_state'] = [1e-2,1e-2]

            args['extr_coh_interaction'] = ext_coh_int
            args['extruder_rnap_interaction'] = ext_rnap_int

            # late_add = '_late' # '_late'
            cohB_add = '_cohB' if args['extr_coh_interaction'] else ''
            igP_add = '' if args['extruder_rnap_interaction'] else '_igP'
            lt_add = cohB_add + igP_add + '_10tms_s50k_lt500k_s200k_lt4m'
            
            chosen_slice = slice_str_parser(chosen_slice_str)
            if chosen_slice[0] == 'chr0':
                chosen_genes = pd.read_csv(rna_seq_dir +'/genes_for_modeling.csv') 
                chosen_genes['rnap_sep_from_max'] = chosen_genes['transcript_id'].apply(lambda x: 1e4 if x[-1] == 'w' else 1e3)
            else:
                chosen_genes = None

            work_dir_name = f'./{name_prefix}chroms_for_statistics'
            if not os.path.exists(work_dir_name):
                os.mkdir(work_dir_name)
                
            prob_func_name = None

            args['Topology_state'] = [prob, prob]
            if prob_func_name is not None:
                if 'one_sided' in prob_func_name:
                    args['Topology_state'] = [prob, 1.0]
            args['prob_func_name'] = prob_func_name

            folder_add = f'_prob_{int(prob*100)}' + late_add + lt_add
            if prob_func_name is not None:
                folder_add += f'_{prob_func_name}'

            folder_name = f'{work_dir_name}/{chosen_slice[0]}' + folder_add
            logs_path = f'{work_dir_name}/logs_{chosen_slice[0]}'+ folder_add
            
            if f'{chosen_slice[0]}' + folder_add in start_oocyte_dict.keys():
                calculate_1D = False
                start_oo = start_oocyte_dict[f'{chosen_slice[0]}' + folder_add]
            else:
                calculate_1D = True
                start_oo = 0
            
            try:
                if calculate_1D:
                    start_time = time.time()
                    end_time_old = start_time

                    activity_name = 'min_FPKM' # 'FPKM'
                    mask = (genes_df['chrom'] == chosen_slice[0]) & (genes_df['start'] > chosen_slice[1]) & (genes_df['end'] < \
                            chosen_slice[2]) & (genes_df['type'] == 'transcript') 
                    chosen_genes = chosen_genes if chosen_genes is not None else genes_df[mask].reset_index(drop=True) 
                    chosen_genes = gene_intersection_corrector(chosen_genes)
                    chosen_genes = chosen_genes[chosen_genes['len'] > 1e3].reset_index(drop=True)
                    # chosen_genes['rnap_sep_from_max'] = chosen_genes['rnap_sep_from_max'] * 1e9
                    # chosen_genes = chosen_genes.loc[458:458, :]

                    full_df = full_df[full_df['transcript_id'].isin(chosen_genes['transcript_id'])]
                    for transcript in chosen_genes['transcript_id']:
                        sample_transcript = chosen_genes[chosen_genes['transcript_id']==transcript].squeeze()
                        for idx in full_df[full_df['transcript_id']==transcript].index:
                            if full_df.loc[idx,'start'] < sample_transcript['start']:
                                full_df.loc[idx,'start']=sample_transcript['start']
                            if full_df.loc[idx,'end']>sample_transcript['end']:
                                full_df.loc[idx,'end']=sample_transcript['end']
                            if full_df.loc[idx,'end']-full_df.loc[idx,'start']<=0:
                                full_df = full_df.drop(index = idx)



                    chosen_genes_beads, N, bead_sizes = genes_to_beads_convert(chosen_genes, full_df, args, chosen_slice, 
                                                                       rnap_sep_name = 'rnap_sep_min_FPKM' + late_add)
                    args['N'] = N
                    tandems_df = create_tandems_df(chosen_genes, late_add = late_add, filter_multiplier = 1.0)
                    
                    # Применяем преобразование к start и end
                    tandems_df['start_beads'] = tandems_df['start'].apply(lambda x: genomic_to_beads(x, chosen_slice, bead_sizes))
                    tandems_df['end_beads'] = tandems_df['end'].apply(lambda x: genomic_to_beads(x, chosen_slice, bead_sizes))
                    
                    if os.path.exists(folder_name):
                        shutil.rmtree(folder_name)
                    os.mkdir(folder_name)

                    with open(logs_path, 'a') as log_file:
                        log_file.write(f'File generated\n')

                    N_coh = np.max([int(np.round((chosen_slice[2] - chosen_slice[1])/(args['Cohesin_separator']))), 1])
                    maxcondesins = np.max([int(np.ceil((chosen_slice[2] - chosen_slice[1])/(args['ave_loop_size'])*2)), 2])
                    if 'ave_loop_size2' in args.keys():
                        maxcondesins2 = np.max([int(np.ceil((chosen_slice[2] - chosen_slice[1])/(args['ave_loop_size2'])*2)), 2])
                        args['LIFETIME'] = [np.max(args['LIFETIME'])]*maxcondesins
                        args['LIFETIME2'] = [np.max(args['LIFETIME2'])]*maxcondesins2
                        maxcondesins += maxcondesins2

                    steps_wo_saving = 10 # int((2)*np.max(args['LIFETIME'])) # 10
                    args['steps_wo_saving'] = steps_wo_saving
                    initiate_steps = initiate_steps_estimation(chosen_genes_beads, args, n_lifetimes = 20.0) # tandems_df.apply(sliding_time_calculation, axis = 1).max()*2 # int((4)*np.max(args['LIFETIME'])) # initiate_steps_estimation(chosen_genes_beads, args)
                    first_modeling_steps_wo_saving = int((5)*np.max(args['LIFETIME']))

                    steps = int(np.ceil((2)*(int(np.max(args['LIFETIME']))/steps_wo_saving)))
                    args['steps'] = steps

                    max_rnaps = sum([abs(x[0]-x[2])+1 for x in chosen_genes_beads])
                    args['N_coh'] = int(N_coh)
                    args['maxcondesins'] = int(maxcondesins)
                    args['max_rnaps'] = int(max_rnaps)
                    with open(logs_path, 'a') as log_file:
                        log_file.write(f'max_rnaps: {max_rnaps}')


                    args['chosen_slice_str'] = chosen_slice_str
                    json.dump(args, open(f'{folder_name}/info.json', 'w'), indent = 4)
                    json.dump(bead_sizes, open(f'{folder_name}/bead_sizes.json', 'w'))


                    max_steps = int(steps * restart_simulation)
                    assert max_steps > 0
                    # assert (trajectory_len // steps_wo_saving) % steps == 0
                    assert max_steps % steps == 0


                    with open(logs_path, 'a') as log_file:
                        log_file.write(f'Start working with test file\n')
                        log_file.write(f'Total steps {max_steps}\n')

                    with h5py.File(f'{folder_name}/RNAp_pos_brists.h5', mode='w') as RNApfile, \
                        h5py.File(f'{folder_name}/Cohesins_positions.h5', mode='w') as COHfile, \
                        h5py.File(f'{folder_name}/Condensins_positions.h5', mode='w') as CONDfile:

                        RNApfile.attrs["N"] = N
                        CONDfile.attrs["N"] = N
                        COHfile.attrs["N"] = N

                        compression = "gzip"
                        rnap_pos_dset = RNApfile.create_dataset("positions",
                                                         shape=(restart_simulation,
                                                                steps,
                                                                max_rnaps),
                                                                dtype=np.int32, 
                                                                compression=compression,
                                                               )
                        brist_dset = RNApfile.create_dataset("bristles",
                                                             shape=(restart_simulation,
                                                                    steps,
                                                                    max_rnaps),
                                                                    dtype=np.int32, 
                                                                    compression=compression,
                                                            )

                        cond_dset = CONDfile.create_dataset("positions",
                                                         shape=(restart_simulation,
                                                                steps,
                                                                maxcondesins, 2),
                                                                dtype=np.int32, 
                                                                compression=compression,
                                                           )
                        coh_dset = COHfile.create_dataset("positions",
                                                         shape=(restart_simulation,
                                                                steps,
                                                                N_coh, 2),
                                                                dtype=np.int32, 
                                                                compression=compression,
                                                         )

                        for k in range(restart_simulation//batch_to_save):
                            sim_full_args = []
                            for nthr in range(restart_simulation):
                                sim_full_args += [(0, initiate_steps, 
                                                   *cpu_params_set_maker(1, max_rnaps, chosen_genes_beads, bead_sizes, args, seed = start_seed+k+1, tandems=tandems_df),
                                                   True, # Only extruders and RNAP can move and interact
                                                   )]
                            pool = Pool(processes=nproc)
                            out_init = pool.starmap(cpu_make_n_steps, sim_full_args)
                            pool.close()      
                            out_to_save = out_init.copy()
                            del(out_init, sim_full_args)
                            
                            with open(logs_path, 'a') as log_file:
                                log_file.write(f'Done with COH initiate steps in {verbose_timedelta(time.time() - start_time)}\n')

                            sim_full_args = []
                            for nthr in range(restart_simulation):
                                sim_full_args += [(0, first_modeling_steps_wo_saving, 
                                                   *out_to_save[nthr],
                                                   False, # Extruders can move and interact either
                                                   )]
                            pool = Pool(processes=nproc)
                            out_init = pool.starmap(cpu_make_n_steps, sim_full_args)
                            pool.close()      
                            del(sim_full_args)

                            with open(logs_path, 'a') as log_file:
                                log_file.write(f'Done with all initiate steps in {verbose_timedelta(time.time() - start_time)}\n')
                                
                            iter_stat = []
                            out_to_save = out_init.copy()
                            del(out_init)
                            for i in range(steps):
                                start_iter = time.time()

                                sim_full_args = []
                                for nthr in range(restart_simulation):
                                    sim_full_args += [(0, steps_wo_saving, 
                                                       *out_to_save[nthr],
                                                       False, # Extruders can move and interact either
                                                       )]
                                pool = Pool(processes=nproc)
                                out_to_save = pool.starmap(cpu_make_n_steps, sim_full_args)
                                pool.close()
                                del(sim_full_args)

                                end_calculus = time.time()

                                rnap_pos_dset[k*batch_to_save:(k+1)*batch_to_save, i, :] = multiproc_out_saver(out_to_save, 3, k, batch_to_save)
                                cond_dset[k*batch_to_save:(k+1)*batch_to_save, i, :] = multiproc_out_saver(out_to_save, 13, k, batch_to_save)
                                coh_dset[k*batch_to_save:(k+1)*batch_to_save, i, :] = multiproc_out_saver(out_to_save, 17, k, batch_to_save)

                                tmp_brist = []
                                brist_all = multiproc_out_saver(out_to_save, 4, k, batch_to_save)
                                for tn in range(len(brist_all)):
                                    tmp_brist += [brist_all[tn][:,1]]
                                brist_dset[k*batch_to_save:(k+1)*batch_to_save, i, :] = tmp_brist

                                end_writing = time.time()

                                calc_time = verbose_timedelta(end_calculus - start_iter)
                                write_time = verbose_timedelta(end_writing - end_calculus)
                                iter_time = verbose_timedelta(end_writing - start_iter)
                                iter_stat += [end_writing - start_iter]
                                total_time = verbose_timedelta(end_writing - start_time)
                                rem_est = verbose_timedelta(np.mean(iter_stat) * (steps - (i+1)))
                                with open(logs_path, 'a') as log_file:
                                    log_file.write(' '*250+'\n')
                                    log_file.write(f'Done with {i+1} steps from {steps}. Calculation: {calc_time} | Writing: {write_time} | Iteration: {iter_time} | Total: {total_time} | Remaining estimation: {rem_est}\n')

                            del(out_to_save, tmp_brist, brist_all)
                            time.sleep(2)

                            end_time = time.time()
                            with open(logs_path, 'a') as log_file:
                                log_file.write(f'Saved {(k+1)*batch_to_save} steps after {verbose_timedelta(end_time - start_time)}, delta {verbose_timedelta(end_time - end_time_old)}\n')
                            end_time_old = end_time

                    with open(logs_path, 'a') as log_file:
                        log_file.write("="*100 + "\n")

                if calculate_3D:
                    if not os.path.exists(f'{name_prefix}chroms_for_statistics_3D'):
                        os.mkdir(f'{name_prefix}chroms_for_statistics_3D')
                    
                    path_to_LEFs = f'./{name_prefix}chroms_for_statistics/{chosen_slice[0]}{folder_add}/'
                    path_to_file = f'./{name_prefix}chroms_for_statistics_3D/{chosen_slice[0]}{folder_add}/'

                    if not os.path.exists(path_to_file):
                        os.mkdir(path_to_file)

                    starting_conf_file =None

                    bris_bead_size_nm = 15.#0.33*5
                    beads_gr_size = 10

                    coh_file = h5py.File(path_to_LEFs + "Cohesins_positions.h5", mode='r')
                    cond_file = h5py.File(path_to_LEFs + "Condensins_positions.h5", mode='r')
                    rnap_file = h5py.File(path_to_LEFs + "RNAp_pos_brists.h5", mode='r')

                    sim_info = json.load(open(path_to_LEFs + 'info.json'))

                    bead_sizes = json.load(open(path_to_LEFs + 'bead_sizes.json'))

                    get_every_n_frames = 1


                    bonds_arr = bond_calculator(bead_sizes, beads_gr_size, base_bead_nm = 15.0, 
                                                bp_nuc = sim_info['BEADSIZE_BASE'], bp_tu = sim_info['BEADSIZE_GENE'], 
                                                df_nuc = 3.0, df_tu = 1.,)
                    min_R_nm = bonds_arr.min()/2

                    starting_conf_file = None

                    smcBondDist = 40.
                    BondDist = bonds_arr
                    bond_robustness = bond_robustness_calculator(bead_sizes, beads_gr_size, max_robustness = 1.5e-2,
                                                                bp_nuc = sim_info['BEADSIZE_BASE'], bp_tu = sim_info['BEADSIZE_GENE'], 
                                                                 df_nuc = 3.0, df_tu = 1.,)

                    smc_robustness = 2e-2
                    Erep = 2e-1 # 8e-2
                    Eattr = 2e-2 * beads_gr_size # for best P_s 
                    # Attr_n_sigmas = 1e2 # ~1 nm
                    Attr_dist_nm = 2.0
                    frac_dim = 3.
                    p_len_coef = 1e-2 # 1/2 * (N//2)**(2./frac_dim - 1.)
                    overwrite = True
                    collision_rate = 3e-2
                    start_col_rate = 6e-1
                    n_timesteps_mult = 2e0
                    mass = 1e-1
                    starting_conformation = 'fractal'
                    platform = 'CUDA'
                    save_decimals = 2
                    except_bonds_from_updater = True
                    steps = int(1000 * sim_info['steps_wo_saving'] * get_every_n_frames) * 8
                    first_molsteps_mult = 20
                    molstepsmul = 1.5

                    n_lifetimes = int(sim_info['steps'] * sim_info['steps_wo_saving'] / np.max(sim_info['LIFETIME'] + sim_info['LIFETIME2']))
                    with open(logs_path, 'a') as log_file:
                        log_file.write('n_lifetimes: '+ str(sim_info['steps'] * sim_info['steps_wo_saving'] / np.max(sim_info['LIFETIME'] + sim_info['LIFETIME2'])))
                        
                    print("="*30)
                    print(restart_simulation)
                    print("="*30)
                    
                    for par_sim_i in range(start_oo, restart_simulation, 1):
                        path_to = path_to_file + f'oocyte_{par_sim_i}'
                        if not os.path.exists(path_to):
                            os.mkdir(path_to)
                        path_to += '/'

                        from_frame = 0
                        to_frame = sim_info['steps']

                        frame_slice = slice(from_frame, to_frame, get_every_n_frames)
                        with open(logs_path, 'a') as log_file:
                            log_file.write(f'frame_slice: {frame_slice}')

                        N_raw = len(bead_sizes)
                        Nframes = (frame_slice.stop - frame_slice.start)//frame_slice.step
                        with open(logs_path, 'a') as log_file:
                            log_file.write(f'N_raw: {N_raw}, Nframes: {Nframes}')

                        rnaps_mover = bead_params_updater_from_file(gr_size = beads_gr_size, 
                                                                    file_h5_io = rnap_file,
                                                                    sim_i = par_sim_i,
                                                                    N_raw = N_raw, 
                                                                    base_bead_nm = 15., 
                                                                    a = bris_bead_size_nm,
                                                                   )
                        LEFpositions = np.concatenate(
                            (np.vectorize(lef_pos_calculator)(coh_file['positions'][par_sim_i, frame_slice], N_raw, beads_gr_size), 
                             np.vectorize(lef_pos_calculator)(cond_file['positions'][par_sim_i, frame_slice], N_raw, beads_gr_size)),
                            axis=1
                        )
                        N = rnaps_mover.N

                        restartSimulationEveryBlocks = int(np.max(sim_info['LIFETIME'] + sim_info['LIFETIME2']))//sim_info['steps_wo_saving']
                        saveEveryBlocks = restartSimulationEveryBlocks // 5

                        with open(logs_path, 'a') as log_file:
                            log_file.write(f'restartSimulationEveryBlocks: {restartSimulationEveryBlocks}, saveEveryBlocks: {saveEveryBlocks}\n')

                        # assertions for easy managing code below
                        assert (Nframes % restartSimulationEveryBlocks) == 0, f'Nframes = {Nframes}, restartSimulationEveryBlocks = {restartSimulationEveryBlocks}'
                        assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0, f'restartSimulationEveryBlocks = {restartSimulationEveryBlocks}, saveEveryBlocks = {saveEveryBlocks}'

                        savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
                        simInitsTotal  = (Nframes) // (restartSimulationEveryBlocks)

                        with open(logs_path, 'a') as log_file:
                            log_file.write('simInitsTotal: '+str(simInitsTotal)+ '\n')

                        reporter = HDF5Reporter(folder=path_to, max_data_length=savesPerSim, overwrite=overwrite, blocks_only=False)
                        time_hist = []
                        eK = []
                        eP = []
                        global_eneries = []

                        smcBondWiggleDist = smcBondDist * smc_robustness
                        BondWiggleDist = 1.

                        time_step = 1 / (collision_rate * n_timesteps_mult) # femtosecs, ~1000 steps per "one collision"

                        chains = [(0, N//2, False), (N//2, None, False)]

                        data = make_start_conf(N, BondDist, starting_conformation = starting_conformation, starting_conf_file = starting_conf_file)

                        milker = bondUpdater(LEFpositions, N = N)
                        brist_len, lambd, _ = rnaps_mover.setup(start_block = to_frame-1, block_step = get_every_n_frames)
                        nbCutOffDist=(bris_bead_size_nm**(5/4) * np.array(brist_len)**(3/4) * (np.pi * np.array(lambd))**(-1/4)).max()
                        brist_len, lambd, start_n_rnaps = rnaps_mover.setup(start_block = from_frame, block_step = get_every_n_frames)

                        line_forces = [0 for i in range(len(brist_len))]
                        line_forces[0] = line_forces[-1]= 1e-3
                        line_forces[len(brist_len)//2] = line_forces[len(brist_len)//2+1]= -1e-3


                        with open(logs_path, 'a') as log_file:
                            log_file.write('Start iterations\n')
                            log_file.write("="*100+'\n')
                        for iteration in range(simInitsTotal):   
                            t1 = time.time()
                            eK = []
                            eP = []
                            with open(logs_path, 'a') as log_file:
                                log_file.write(f'Iteration: {iteration}\n')
                            # simulation parameters are defined below 
                            a = Simulation(
                                    max_Ek=1e10,
                                    platform=platform,
                                    integrator="langevin",
                                    mass = mass,
                                    error_tol=1e-10,
                                    timestep = time_step,
                                    GPU = f"{gpu_id}", 
                                    collision_rate = start_col_rate if iteration==0 else collision_rate, # collision rate in inverse picoseconds
                                    N = len(data),
                                    reporters=[reporter],
                                    #PBCbox=[box, box, box],
                                    precision="mixed",
                                    save_decimals = save_decimals,
                            )  # timestep not necessary for variableLangevin

                            ############################## New code ##############################
                            a.set_data(data, center = True)  # loads a polymer, puts a center of mass at zero
                            a.add_force(
                                forcekits.polymer_chains(
                                    a,
                                    chains = chains,

                                    bond_force_func = forces.harmonic_bonds,
                                    bond_force_kwargs = {
                                        # Distance between histons
                                        'bondLength': BondDist,
                                        'bondWiggleDistance': BondWiggleDist, # Bond distance will fluctuate +- 0.05 on average
                                     },

                                    angle_force_func = forces.angle_force,
                                    angle_force_kwargs = {
                                        'k' : p_len_coef,
                                    },

                                    nonbonded_force_func = marko_excluded_volume_with_attraction,
                                    nonbonded_force_kwargs={
                                        'brist_len_beads' : brist_len.tolist(),
                                        'brist_sep_nm' : lambd.tolist(),
                                        'n_rnaps' : start_n_rnaps.tolist(),
                                        'brist_bead_size_nm' : bris_bead_size_nm,
                                        'Erep' : Erep,
                                        'Eattr' : Eattr,
                                        'min_R_nm' : min_R_nm,
                                        'dt_fs' : time_step,
                                        'base_mass_amu' : mass,
                                        'n_timesteps_mult' : n_timesteps_mult,
                                        'Attr_dist_nm': Attr_dist_nm,
                        #                     'Attr_n_sigmas' : Attr_n_sigmas,
                                        'nbCutOffDist': nbCutOffDist,
                                    },

                                    except_bonds=True,
                                )
                            )
                            a.add_force(line_force(a,line_forces, name="line"))
                            non_bond_force_name = "marko_excluded_volume_with_attraction"

                            # ------------ initializing milker; adding bonds ---------
                            # copied from addBond
                            kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)

                            activeParams = {"length": smcBondDist * a.length_scale,"k": kbond}
                            inactiveParams = {"length": smcBondDist * a.length_scale, "k": 0}
                            milker.setParams(activeParams, inactiveParams)

                            # this step actually puts all bonds in and sets first bonds to be what they should be
                            milker.setup(bondForce = a.force_dict['harmonic_bonds'],
                                         nonbondForce = a.force_dict[non_bond_force_name],
                                            blocks=restartSimulationEveryBlocks,
                                            except_bonds = except_bonds_from_updater)

                            with open(logs_path, 'a') as log_file:
                                log_file.write(f'Cutoffdist: {a.force_dict[non_bond_force_name].getCutoffDistance()}\n')

                            if iteration==0:
                                a.local_energy_minimization()
                                a.collisionRate = collision_rate * a.collisionRate.unit
                                a.integrator.setFriction(a.collisionRate)
                                a.integrator.step(steps*first_molsteps_mult)
                            else:
                                a._apply_forces()
                                a.integrator.step(steps*first_molsteps_mult)




                            for i in range(restartSimulationEveryBlocks):

                                if i % saveEveryBlocks == (saveEveryBlocks - 1):  
                                    a.do_block(steps=steps*molstepsmul)

                                else:
                                    a.integrator.step(steps*molstepsmul)  # do steps without getting the positions from the GPU (faster)
                                if i < restartSimulationEveryBlocks - 1:

                                    curBonds, pastBonds = milker.step(a.context, except_bonds = except_bonds_from_updater)  # this updates bonds. You can do something with bonds here
                                    rnaps_mover.step(a, force_name = non_bond_force_name)



                            data = a.get_data() 
                            del a

                            reporter.blocks_only = True  # Write output hdf5-files only for blocks

                            t = time.localtime() 
                            t = time.strftime("%H:%M:%S", t)
                            time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
                            t2 = time.time()
                            time_hist.append(t2-t1)
                            remTime = np.mean(time_hist)*(simInitsTotal-iteration-1)
                            with open(logs_path, 'a') as log_file:  
                                log_file.write('Compleated: ' + str(round(((1+iteration)/simInitsTotal)*100,1))+ '%'+  ' Remained time: '+ 
                                  str(round(remTime/3600,1))+ ' hours '+ 'Time for one iteration:'+ str((t2-t1)/60)+ 'min'+'\n' 
                                 )
                        with open(logs_path, 'a') as log_file:
                            log_file.write('\_(0_0)_/ '+'\_(-o-)_/ ' + '/_(0o0)_\ \n')
                        reporter.dump_data()
            except Exception:
                # tb = sys.exc_info()[2]
                # tbinfo = traceback.format_tb(tb)[0]
                work_dir_name = f'./{name_prefix}chroms_for_statistics'
                errs_path = f'{work_dir_name}/errors_{chosen_slice[0]}' + folder_add
                with open(errs_path, 'a') as err_file:
                    err_file.write(traceback.format_exc())
        
        if (chrom_for_modeling == 'chr1p100'):
            break
        
if __name__ == '__main__':
    main()