from cmath import inf
import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Circle
import matplotlib.animation as animation
import seaborn as sns
import re
import h5py 
                     
### Simulations

def slice_str_parser(chosen_slice_str):
    chosen_slice = chosen_slice_str.split(':')
    chosen_slice = [chosen_slice[0], *chosen_slice[1].split('-')]
    chosen_slice[1] = int(chosen_slice[1].replace(',', ''))
    chosen_slice[2] = int(chosen_slice[2].replace(',', ''))
    return chosen_slice

def bead_size_gene_calculator(rnap_sep, args, rnap_sep_threshold = 250.0):
    nucleosome_compactization = args['BEADSIZE_BASE'] / args['BEADSIZE_GENE']
    if rnap_sep > rnap_sep_threshold:
        return args['BEADSIZE_GENE'] * min([rnap_sep/rnap_sep_threshold, nucleosome_compactization])
    else:
        return args['BEADSIZE_GENE']

# Функция изменена, теперь есть учёт рзбиения на нуклеосомы в межгенном пространстве
def genes_to_beads_convert(chosen_genes, full_df, args, chosen_slice, rnap_sep_name = 'rnap_sep_from_max'): # chosen_slice, N было забыто
    chosen_genes_beads = []
    
    chosen_genes = chosen_genes[chosen_genes[rnap_sep_name].notna()].reset_index(drop=True)
    bead_sizes=[]
    end = chosen_slice[1]
    i = -1
    for i in chosen_genes.index:
        direction = int(chosen_genes.loc[i, 'direction']+'1')

        num_beads = int(round((chosen_genes.loc[i, 'start']-end)/args['BEADSIZE_BASE'],0))
        
        for _ in range(num_beads):
            bead_sizes.append(round((chosen_genes.loc[i, 'start']-end)/num_beads,2))
        
        beadsize_gene = bead_size_gene_calculator(chosen_genes.loc[i, rnap_sep_name], args) # args['BEADSIZE_GENE']

        if direction >0:    
            start_pos = len(bead_sizes)
            end_pos = start_pos+int(round((chosen_genes.loc[i, 'end']-chosen_genes.loc[i, 'start'])/beadsize_gene,0))-1
        elif direction <0:
            end_pos = len(bead_sizes)
            start_pos = end_pos+int(round((chosen_genes.loc[i, 'end']-chosen_genes.loc[i, 'start'])/beadsize_gene,0))-1
            
        end = chosen_genes.loc[i, 'end']
        
        for _ in range(int(round((chosen_genes.loc[i, 'end']-chosen_genes.loc[i, 'start'])/beadsize_gene,0))):
            bead_sizes.append(beadsize_gene)
        
        load_prob = args['TIMESTEP'] / (chosen_genes.loc[i, rnap_sep_name] / args['RNAP_VEL'])
        intron_list = full_df[(full_df['transcript_id'] == chosen_genes.loc[i, 'transcript_id']) & (full_df['type'] == 'intron')].copy()
        intron_list['len'] = intron_list['end'] - intron_list['start']
        intron_list = intron_list[intron_list['len'] > 0][['start', 'end', 'len']].values
        introns = []
        prev_cut_len = 0
        for *st_en, l in intron_list[::direction]:
            cut_pos = abs(st_en[(direction+1)//2] - chosen_genes.loc[i, ['start', 'end'][(direction-1)//2]]) - prev_cut_len
            prev_cut_len += l
            introns += [(cut_pos, l)]
        
        chosen_genes_beads += [[start_pos, direction, end_pos, load_prob, introns]]
    
    if i >= 0:
        last_gene_end = chosen_genes.loc[i, 'end']
    else:
        last_gene_end = chosen_slice[1]

    num_beads = int(round((chosen_slice[2] - last_gene_end)//args['BEADSIZE_BASE'],0))
    for _ in range(num_beads):
        bead_sizes.append(round((chosen_slice[2] - last_gene_end)/num_beads,0))

    bead_sizes = bead_sizes + bead_sizes[::-1]
    N = len(bead_sizes)
    
    for i in range(len(chosen_genes_beads)):
        start_pos, direction, end_pos, load_prob, introns = chosen_genes_beads[i]
        chosen_genes_beads += [[N - start_pos - 1, -1*direction, N - end_pos - 1, load_prob, introns]] # Make sister chromatid gene
    
    return chosen_genes_beads, N, bead_sizes

def make_one_step(
    rnaps = None,
    occupied_rnap = None,
    chosen_genes_beads = None,
    condensins = None,
    occupied = None,
    maxcondesins = None,
    cohesins = None,
):
    global args
    
    if (rnaps is not None) and (occupied_rnap is not None) and (chosen_genes_beads is not None):
        translocate_rnap(rnaps, occupied_rnap, chosen_genes_beads, args, occupied = occupied, cohesins = cohesins, condensins = condensins)
    if (condensins is not None) and (occupied is not None) and (maxcondesins is not None):
        translocate_condensin(condensins, occupied, args, maxcondesins, cohesins = cohesins, occupied_rnap = occupied_rnap)
    if (cohesins is not None) and (occupied is not None):
        translocate_cohesin(cohesins, occupied, args, occupied_rnap = occupied_rnap)
    
def run_one_simulation(rnaps, occupied_rnap, occupied, cohesins, condensins):
    global N, args, steps, chosen_genes_beads, steps_wo_saving, max_rnaps, maxcondesins, N_coh

    cur_rnap_pos = []
    cur_brists = []
    cur_cond_pos = []
    cur_coh_pos = []
    for i in range(steps):
        for _ in range(steps_wo_saving):
            make_one_step(rnaps, occupied_rnap, chosen_genes_beads, condensins, occupied, maxcondesins, cohesins)
            
        rnap_positions = [rnap.position for rnap in rnaps] + [-1 for _ in range(max_rnaps - len(rnaps))]
        bristles = [rnap.bristle for rnap in rnaps] + [-1 for _ in range(max_rnaps - len(rnaps))]
        cond_positions = [(cond.left.pos, cond.right.pos) for cond in condensins] + [(-1,-1)]*(maxcondesins - len(condensins))
        coh_positions = [(coh.left.pos, coh.right.pos) for coh in cohesins] + [(-1,-1)]*(N_coh - len(cohesins))

        cur_rnap_pos.append(rnap_positions)  # appending current positions to an array
        cur_brists.append(bristles)
        cur_cond_pos.append(cond_positions)
        cur_coh_pos.append(coh_positions)

    return cur_rnap_pos, cur_brists, cur_cond_pos, cur_coh_pos

def params_generator(params, n):
    for _ in range(n):
        yield params

        
        
        
### Draw arcs

def plot_interaction(
    l, r, n_lef=0,
    height_factor=1.0,
    lw_factor=1.0,
    lw_max=5.0,
    max_height = 150,
    height=None,
    y_base=30,
    plot_text=True,
    orientation = 1,
    color='tomato',
    **kwargs,
):
    """Visualize an individual loop with an arc diagram.
    """
    arc_center = ((l+r)/2, y_base)
    arc_height = (min(max_height, (r-l)/2.0*height_factor * orientation)
                  if (height is None) else height)

    arc = Arc(xy=arc_center,
            width=r-l,
            height=arc_height,
            theta1 = min(0, 180 * orientation),
            theta2 = max(0, 180 * orientation),
            alpha = 0.45,
            lw=min(lw_max, 1.0*n_lef*lw_factor),
            color=color,
            capstyle='round')
    plt.gca().add_artist(arc)

    if n_lef > 1 and arc_center[0] < plt.xlim()[1] and plot_text:
        plt.text(x=arc_center[0],
                 y=arc_center[1]+arc_height/2+30,
                 horizontalalignment='center',
                 verticalalignment='center',
                 s=str(int(n_lef)),
                 fontsize=20,
                 color=color
                )
        
        
def plot_cohesin(
    l, r, n_lef=0,
    y_base = [0, 0],
    lw_factor=1.0,
    lw_max=5.0,
    color='seagreen',
    **kwargs, 
):
    """Visualize an individual cohesin with a line diagram.
    """
    poly = Polygon(
        xy = [[x0,x1] for x0,x1 in zip([l, r], [y_base[0], y_base[1]])],
        closed=False,
        lw=min(lw_max, 1.0*n_lef*lw_factor),
        color=color,
        alpha = 0.45,
    )
    plt.gca().add_artist(poly)
    

def prepare_canvas(
    L,
    site_width_bp = None,
    single_strand = True,
    max_y = 200,
    ):
#     plt.figure(figsize=(8,8))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(
        which='both',
        bottom='on',
        right='off',
        left='off',
        direction='out',
        top='off')
    plt.yticks([])
    
    if not single_strand:
        assert L%2 == 0, f'Sites number must be even if two_strands is True (now {L} sites)'
        plt.xlim(-1, L//2+1)
        plt.ylim(-max_y*1.2, max_y*1.2)
        plt.axhline(int(max_y*0.1), color='gray', lw=2, zorder=-1)
        plt.axhline(int(-max_y*0.1), color='gray', lw=2, zorder=-1)
    else:
        plt.xlim(-1, L+1)
        plt.ylim(0, max_y)
        plt.axhline(int(max_y*0.1), color='gray', lw=2, zorder=-1)

    if site_width_bp:
        plt.xticks([100*i*1000/float(site_width_bp) for i in range(16)],
                   [100*i for i in range(16)],
                   fontsize=20)
        plt.xlabel('chromosomal position, kb', fontsize=20)
        
    
    
def plot_lefs_single_strand(
        l_sites, 
        r_sites,
        y = [20],
        colors='tomato',
        **kwargs):
    """Plot an arc diagram for a list of loops.
    """
    
    unique_connections = pd.DataFrame(np.vstack([l_sites,r_sites]).T, columns=['l_sites', 'r_sites']).value_counts().reset_index()
    for i, (ls, rs, nl) in enumerate(unique_connections.values):
        plot_interaction(
            ls,
            rs,
            nl,
            y_base = y[0],
            color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
            **kwargs) 
        
def plot_lefs_sister_strands(
        l_sites, 
        r_sites, 
        L = 0,
        y = [20, -20],
        colors='tomato',
        **kwargs):
    """Plot an arc diagram for a list of loops.
    """
    
    unique_connections = pd.DataFrame(np.vstack([l_sites,r_sites]).T, columns=['l_sites', 'r_sites']).value_counts().reset_index()
    
    for i, (ls, rs, nl) in enumerate(unique_connections.values):
        l_orient = np.sign(ls-L//2+1)
        r_orient = np.sign(rs-L//2)
        if l_orient == r_orient:
            if l_orient < 0:
                plot_interaction(
                    L//2 - abs(ls-L//2),
                    L//2 - abs(rs-L//2),
                    nl,
                    # color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
                    y_base = y[int(l_orient+1)//2],
                    orientation = -l_orient,
                    **kwargs)

            if l_orient > 0:
                plot_interaction(
                    L//2-1 - abs(ls-L//2),
                    L//2-1 - abs(rs-L//2),
                    nl,
                    # color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
                    y_base = y[int(l_orient+1)//2],
                    orientation = -l_orient,
                    **kwargs)
        else:
            plot_cohesin(
                L//2 - abs(ls-L//2),
                L//2-1 - abs(rs-L//2),
                nl,
                y_base = y,
#                 color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
                **kwargs,
            )
            
            
def plot_rnap(
    site, bristle, n_brist=0,
    height_factor=1.0,
    lw_factor=1.0,
    lw_max=5.0,
    max_height = 150,
    height=None,
    y_base=30,
    plot_text=True,
    orientation = 1,
    color='#000080',
    **kwargs,
):
    circle = Circle(
        (site, y_base), 
        radius = 1,
        color = color,
        alpha = 0.35,
    )
    plt.gca().add_artist(circle)
    
    if np.sign(bristle) > 0:
        y_fin = min(y_base + bristle*height_factor, y_base + max_height)
    else:
        y_fin = max(y_base - abs(bristle)*height_factor, y_base - max_height)
        
    poly = Polygon(
        xy = [[x0,x1] for x0,x1 in zip([site, site], [y_base, y_fin])],
        closed=False,
        lw = min(lw_max, 1.0*n_brist*lw_factor),
        color=color,
        alpha = 0.35,
    )
    plt.gca().add_artist(poly)

def max_brist_and_counts(df):
    return pd.Series({'bristles': df['bristles'].max(), 'counts': len(df)})
    
def plot_rnaps_single_strand(
    sites, 
    bristles,
    y = [20],
    colors='#000080',
    **kwargs
):
    rnaps_df = pd.DataFrame(np.vstack([sites, bristles]).T, columns=['sites', 'bristles']).groupby('sites').apply(max_brist_and_counts).reset_index()
    for i, (site, bristle, nl) in enumerate(rnaps_df.values):
        plot_rnap(
            site,
            bristle,
            nl,
            y_base = y[0],
            color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
            **kwargs)
        
def plot_rnaps_sister_strands(
        sites, 
        bristles, 
        L = 0,
        y = [20, -20],
        colors='#000080',
        **kwargs
):
    rnaps_df = pd.DataFrame(np.vstack([sites, bristles]).T, columns=['sites', 'bristles']).groupby('sites').apply(max_brist_and_counts).reset_index()
    for i, (site, bristle, nl) in enumerate(rnaps_df.values):
        if site < L//2:
            plot_rnap(
                site,
                bristle,
                nl,
                y_base = y[0],
                color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
                **kwargs)
        else:
            plot_rnap(
                L//2-1 - abs(site-L//2),
                -bristle,
                nl,
                y_base = y[1],
                color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
                **kwargs)
        

def draw_gif_by_coords(
    out_name,
    path_to_rnap_h5 = None,
    path_to_coh_h5 = None,
    path_to_cond_h5 = None,
    first_frame = 0,
    n_frames = None,
    save_each_frame = 1,
    time_interval = 30, 
    single_strand = True, 
    shift_k = 0.1,
    rnap_heigh_factor = 3.0,
    arc_heigh_factor = 6.0,
    base_resolution = 1,
    out_resolution = int(1e3),
    bristle_res = 1.0,
    simulation_i = 0,
    **kwargs,
):

    fig, ax = plt.subplots()
    
    L = 0
    if path_to_rnap_h5 is not None:
        rnap_h5 = h5py.File(path_to_rnap_h5, mode='r')
        L = rnap_h5.attrs['N']
        # max_brist_len = rnap_h5['bristles'][:].max()
    if path_to_coh_h5 is not None:
        coh_h5 = h5py.File(path_to_coh_h5, mode='r')
        L = coh_h5.attrs['N']
    if path_to_cond_h5 is not None:
        cond_h5 = h5py.File(path_to_cond_h5, mode='r')
        L = cond_h5.attrs['N']
    
    if (type(base_resolution) is int) or (type(base_resolution) is float):
        bead_sizes_array = np.cumsum([base_resolution]*L)
        if single_strand:
            L = int(np.ceil(L * base_resolution/out_resolution))
        else:
            L = int(np.ceil(L/2 * base_resolution/out_resolution)) * 2
    elif type(base_resolution) is list:
        if single_strand:
            bead_sizes_array = np.cumsum(base_resolution)
            L = int(np.ceil(bead_sizes_array[-1]/out_resolution))
        else:
            bead_sizes_array = np.cumsum(base_resolution)
            L = int(np.ceil(np.ceil(bead_sizes_array[-1])/2/out_resolution))*2
        

    def draw_frame(frame):
        print(' '*200, end = '\r')
        print(f'Frame: {frame}', end = '\r')
        plt.gca().clear()
        prepare_canvas(L, single_strand = single_strand, max_y = L)
        
        if path_to_rnap_h5 is not None:
            positions = rnap_h5['positions'][simulation_i, frame,:]
            bristles = rnap_h5['bristles'][simulation_i, frame,:]
            rnap_sites = np.round(bead_sizes_array[positions[positions > 0]] / out_resolution).astype(int)
            rnap_bristles = bristles[bristles > 0] * bristle_res / out_resolution

            if len(rnap_sites) > 0:
                if single_strand:
                    plot_rnaps_single_strand(
                        sites = rnap_sites, 
                        bristles = rnap_bristles,
                        y = [L*shift_k],
                        max_height = L,
                        plot_text = False,
                        height_factor = rnap_heigh_factor,
                        lw_factor = 1.0,
                        lw_max = 5.0,
                    )
                else:
                    plot_rnaps_sister_strands(
                        sites = rnap_sites, 
                        bristles = rnap_bristles,
                        L = L,
                        y = [(L/2)*shift_k, -(L/2)*shift_k],
                        max_height = L,
                        plot_text = False,
                        height_factor = rnap_heigh_factor,
                        lw_factor = 1.0,
                        lw_max = 5.0,
                    )
                    
        if path_to_coh_h5 is not None:
            l_sites = coh_h5['positions'][simulation_i, frame,:,0]
            r_sites = coh_h5['positions'][simulation_i, frame,:,1]
            l_sites = np.round(bead_sizes_array[l_sites[l_sites > 0]] / out_resolution).astype(int)
            r_sites = np.round(bead_sizes_array[r_sites[r_sites > 0]] / out_resolution).astype(int)
            assert l_sites.shape == r_sites.shape
            
            if len(l_sites) > 0:
                if single_strand:
                    plot_lefs_single_strand(
                        l_sites = l_sites, 
                        r_sites = r_sites,
                        y = [L*shift_k],
                        max_height=L,
                        plot_text=False,
                        height_factor=arc_heigh_factor,
                        lw_factor=1.0,
                        lw_max=5.0,
                    )
                else:
                    plot_lefs_sister_strands(
                        l_sites=l_sites, 
                        r_sites=r_sites,
                        L = L,
                        y = [L/2*shift_k, -L/2*shift_k],
                        max_height=L,
                        plot_text=False,
                        height_factor=arc_heigh_factor,
                        lw_factor=1.0,
                        lw_max=5.0,
                    )
            
        if path_to_cond_h5 is not None:
            l_sites = cond_h5['positions'][simulation_i, frame,:,0]
            r_sites = cond_h5['positions'][simulation_i, frame,:,1]
            l_sites = np.round(bead_sizes_array[l_sites[l_sites > 0]] / out_resolution).astype(int)
            r_sites = np.round(bead_sizes_array[r_sites[r_sites > 0]] / out_resolution).astype(int)
            assert l_sites.shape == r_sites.shape
            
            if len(l_sites) > 0:
                if single_strand:
                    plot_lefs_single_strand(
                        l_sites = l_sites, 
                        r_sites = r_sites,
                        y = [L*shift_k],
                        max_height=L,
                        plot_text=False,
                        height_factor=arc_heigh_factor,
                        lw_factor=1.0,
                        lw_max=5.0,
                    )
                else:
                    plot_lefs_sister_strands(
                        l_sites=l_sites, 
                        r_sites=r_sites,
                        L = L,
                        y = [L/2*shift_k, -L/2*shift_k],
                        max_height=L,
                        plot_text=False,
                        height_factor=arc_heigh_factor,
                        lw_factor=1.0,
                        lw_max=5.0,
                    )

    ani = animation.FuncAnimation(fig = fig, 
                                  func = draw_frame, 
                                  frames = range(first_frame, first_frame + (n_frames or rnap_h5['positions'].shape[1]), save_each_frame), 
                                  interval = time_interval,
                                 )
    ani.save(out_name)

def calculate_interaction_matrix(
    base_resolution = None,
    path_to_coh_h5 = None,
    path_to_cond_h5 = None,
    first_frame = 0,
    n_frames = None,
    save_each_frame = 1,
    out_resolution = int(1e3),
    interaction_presence_only = False,
    log_scale = False,
    sector = None,
    chunk = 1000,
    add_noise = False,
    **kwargs,
):
    assert base_resolution is not None, f'Base resolution is prohibitted'
    L = 0
    
    if path_to_coh_h5 is not None:
        coh_h5 = h5py.File(path_to_coh_h5, mode='r')
        L = coh_h5.attrs['N']
        n_frames = n_frames or coh_h5['positions'].shape[1]
        
    if path_to_cond_h5 is not None:
        cond_h5 = h5py.File(path_to_cond_h5, mode='r')
        L = cond_h5.attrs['N']
        n_frames = n_frames or cond_h5['positions'].shape[1]
    
    rows = []
    cols = []
    bins = np.linspace(first_frame, first_frame + n_frames, max(n_frames//chunk+1, 2), dtype=int) # chunks boundaries
    for st,end in zip(bins[:-1], bins[1:]):
        if path_to_coh_h5 is not None:
            mask = (coh_h5['positions'][:, st:end:save_each_frame,:,0] > 0)
            tmp_rows = coh_h5['positions'][:, st:end:save_each_frame][mask,0].reshape(-1).tolist() + (L - 1 - coh_h5['positions'][:, st:end:save_each_frame][mask,1]).reshape(-1).tolist()
            tmp_cols = (L - 1 - coh_h5['positions'][:, st:end:save_each_frame][mask,1]).reshape(-1).tolist() + coh_h5['positions'][:, st:end:save_each_frame][mask,0].reshape(-1).tolist()
            rows += tmp_rows
            cols += tmp_cols
            assert max(tmp_rows) < L/2, f'NOT {max(rows)} < {L/2}'
            assert max(tmp_cols) < L/2, f'NOT {max(rows)} < {L/2}'
            
        if path_to_cond_h5 is not None:
            mask_f = (cond_h5['positions'][:, st:end:save_each_frame,:,0] < L/2) & (cond_h5['positions'][:, st:end:save_each_frame,:,0] > 0)
            # mask_f = np.ones_like(mask_f)
            mask_s = (cond_h5['positions'][:, st:end:save_each_frame,:,0] >= L/2) & (cond_h5['positions'][:, st:end:save_each_frame,:,0] > 0)
            # mask_s = np.ones_like(mask_s)
            tmp_rows = cond_h5['positions'][:, st:end:save_each_frame][mask_f,0].reshape(-1).tolist() + (L - 1 - cond_h5['positions'][:, st:end:save_each_frame][mask_s,0]).reshape(-1).tolist() + \
                    cond_h5['positions'][:, st:end:save_each_frame][mask_f,1].reshape(-1).tolist() + (L - 1 - cond_h5['positions'][:, st:end:save_each_frame][mask_s,1]).reshape(-1).tolist()
            tmp_cols = cond_h5['positions'][:, st:end:save_each_frame][mask_f,1].reshape(-1).tolist() + (L - 1 - cond_h5['positions'][:, st:end:save_each_frame][mask_s,1]).reshape(-1).tolist() + \
                    cond_h5['positions'][:, st:end:save_each_frame][mask_f,0].reshape(-1).tolist() + (L - 1 - cond_h5['positions'][:, st:end:save_each_frame][mask_s,0]).reshape(-1).tolist()
            rows += tmp_rows
            cols += tmp_cols
            assert max(tmp_rows) < L/2, f'NOT {max(rows)} < {L/2}'
            assert max(tmp_cols) < L/2, f'NOT {max(rows)} < {L/2}'
    
    
    
    if (type(base_resolution) is int) or (type(base_resolution) is float):
        rows = np.array(rows) * base_resolution
        cols = np.array(cols) * base_resolution
    elif type(base_resolution) is list:
        true_resolution = np.cumsum(base_resolution)
        if add_noise:
            rows = true_resolution[np.array(rows)] + np.random.randint(np.array(base_resolution)[np.array(rows)])
            cols = true_resolution[np.array(cols)] + np.random.randint(np.array(base_resolution)[np.array(cols)])
        else:
            rows = true_resolution[np.array(rows)]
            cols = true_resolution[np.array(cols)]

        
    
    # print(max(rows))
    if sector is None:
        rows = np.round(rows / out_resolution).astype(int)
        cols = np.round(cols / out_resolution).astype(int)
        if type(base_resolution) == int:
            shape = (np.round(L/2 * base_resolution / out_resolution).astype(int)+1, 
                 np.round(L/2 * base_resolution / out_resolution).astype(int)+1)
        elif type(base_resolution)  ==  list:
            shape = (np.round(true_resolution.max()/out_resolution).astype(int)+1,
                     np.round(true_resolution.max()/out_resolution).astype(int)+1)
    else:
        assert len(sector) == 2
        assert (len(sector[0]) == len(sector[1])) and (len(sector[0]) == 2)
        mask = (rows >= sector[1][0]) & (rows < sector[1][1]) & (cols >= sector[0][0]) & (cols < sector[0][1])
        rows = rows[mask]
        cols = cols[mask]
        shape = (np.round((sector[1][1] - sector[1][0]) / out_resolution).astype(int)+1, 
                 np.round((sector[0][1] - sector[0][0]) / out_resolution).astype(int)+1)
        
        rows = np.round((rows - sector[1][0]) / out_resolution).astype(int)
        cols = np.round((cols - sector[0][0]) / out_resolution).astype(int)
    
    assert len(rows) == len(cols)    
    interaction_matrix = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape = shape)
    
    interaction_matrix.sum_duplicates()

    if interaction_presence_only: 
        tmp_mask = interaction_matrix.data > 0
        interaction_matrix.data = tmp_mask
        interaction_matrix.row = interaction_matrix.row[tmp_mask]
        interaction_matrix.col = interaction_matrix.col[tmp_mask]
    elif log_scale: 
        interaction_matrix.data = np.nan_to_num(np.log10(interaction_matrix.data+1), nan=0.0, posinf=0.0, neginf=0.0)

    return interaction_matrix

def save_contact_pairs(array, res, chromosome = 'chr1', start_bin = 0, name = 'default', chr_size_file = None):
    col_names = ["str1", "chr1", "pos1", "frag1", "str2", "chr2", "pos2", "frag2", "score"]
    coo_mat = sp.coo_matrix(sp.triu(array, format='coo'))

    if chr_size_file is not None:
        chr_size_dict = pd.read_csv(chr_size_file, sep='\t', header = None).set_index(0)[1].to_dict()
        max_coord = chr_size_dict[chromosome]
    else:
        max_coord = inf

    df_to_save = pd.DataFrame()
    df_to_save['pos1'] = start_bin + coo_mat.row[:]*res
    df_to_save['pos2'] = start_bin + coo_mat.col[:]*res
    df_to_save['score'] = coo_mat.data[:]
    df_to_save['str1'] = 0
    df_to_save['str2'] = 0
    df_to_save['frag1'] = 0
    df_to_save['frag2'] = 1
    df_to_save['chr1'] = chromosome
    df_to_save['chr2'] = chromosome
    
    df_to_save['pos1'][df_to_save['pos1'] >= max_coord] = max_coord - 1
    df_to_save['pos2'][df_to_save['pos2'] >= max_coord] = max_coord - 1
    df_to_save.sort_values(['chr1', 'pos1', 'chr2', 'pos2'], inplace=True)

    df_to_save[col_names].to_csv(f'{name}.short', sep=' ', index = False, header = False)

def draw_interaction_heatmap(
    out_name = None,
    figsize = (5,5),
    color_range = (None, None),
    **kwargs,
):
    fig, ax = plt.subplots(figsize = figsize)
    
    interaction_matrix = calculate_interaction_matrix(**kwargs).toarray()
  
    if ax != None:
        sns.heatmap(interaction_matrix, ax = ax, square = True, vmin = color_range[0], vmax = color_range[1])
    else:
        sns.heatmap(interaction_matrix, square = True, vmin = color_range[0], vmax = color_range[1])
    
    if out_name is not None:
        plt.savefig(out_name)

### Other
def gtf_reader(file_path, transcripts_only = False, stop_after = None):
    genes_list = []
    with open(file_path) as gtf_file:
        line_len = []
        black_list_genes = []
        all_lines = gtf_file.readlines()
        for line_n,line in enumerate(all_lines):
            if line_n == stop_after:
                break
            if '#' not in line:
                line = line.split('\n')[0]
                line = re.split('[;\t]', line)
                
                if (transcripts_only and line[2] == 'transcript') or not (transcripts_only):
                    gene_id = line[8].split(' ')[-1][1:-1]

                    black_list = False
    #                 black_list = sum([
    #                     y[0] <= int(line[3]) <= y[1] or \
    #                     y[0] <= int(line[4]) <= y[1] \
    #                     for y in tel_cen
    #                 ])
                    if black_list:
                        black_list_genes += [gene_id]
                    if not gene_id in black_list_genes:
                        if 'TPM' in line[-2]:
                            TPM = float(line[-2].split(' ')[-1][1:-1])
                        if 'FPKM' in line[-3]:
                            FPKM = float(line[-3].split(' ')[-1][1:-1])
                        if 'cov' in line[-4]:
                            cov = float(line[-4].split(' ')[-1][1:-1])
                        else:
                            cov = float(line[-2].split(' ')[-1][1:-1])
                        tmp = {
                            'type': line[2],
                            'chrom': line[0],
                            'start': int(line[3]),
                            'end': int(line[4]),
                            'direction': line[6],
                            'gene_id': line[8].split(' ')[-1][1:-1],
                            'transcript_id': line[9].split(' ')[-1][1:-1],
                            'TPM': TPM,
                            'FPKM': FPKM,
                            'cov': cov,
                        }
                        genes_list += [tmp]
            if line_n % 1000 == 0: print(f'{int(line_n/len(all_lines)*100)}% ({line_n} / {len(all_lines)}) are read' + ' '*200, end = '\r')
    return pd.DataFrame(genes_list, columns = ['type', 'chrom', 'start', 'end', 'direction', 'gene_id', 'transcript_id', 'TPM', 'FPKM', 'cov'])


def verbose_timedelta(delta):
    d, s = divmod(delta, 60*60*24)
    h, s = divmod(s, 60*60)
    m, s = divmod(s, 60)
    s, ms = divmod(s, 1)
    ms = ms*1000
    labels = ['d', 'h', 'm', 's', 'ms']   
    dhms = ['%s%s' % (i, lbl) for i, lbl in zip([int(d), int(h), int(m), int(s), int(ms)], labels)]
    for start in range(len(dhms)):
        if not dhms[start].startswith('0'):
            break
    for end in range(len(dhms)-1, -1, -1):
        if not dhms[end].startswith('0'):
            break  
    return ':'.join(dhms[start:end+1])

def overlap_fix(chosen_genes, args):
    for i in range(len(chosen_genes)):
        masking = (chosen_genes['start'][1:].reset_index(drop=True)-chosen_genes['end'][0:-1].reset_index(drop=True)) <0
        masking[len(masking)] = False #(chosen_genes['start'][-1:].reset_index(drop=True)-chosen_genes['end'][-2:-1].reset_index(drop=True) <0)[0]
        fpkmasking = (chosen_genes['FPKM'][1:].reset_index(drop=True)-chosen_genes['FPKM'][0:-1].reset_index(drop=True)) <0
        fpkmasking[len(fpkmasking)] = False
        if not chosen_genes[masking&fpkmasking].empty:
            chosen_genes.loc[(chosen_genes[masking&fpkmasking].index+1), 'start'] = chosen_genes.loc[(chosen_genes[masking&fpkmasking].index), 'end'].values +1
        if not chosen_genes[masking&~fpkmasking].empty:
            chosen_genes.loc[chosen_genes[masking&~fpkmasking].index, 'end'] = chosen_genes.loc[chosen_genes[masking&~fpkmasking].index+1, 'start'].values -1
        chosen_genes['len']=chosen_genes['end']-chosen_genes['start']
        if not sum(chosen_genes['len']<=args['BEADSIZE_GENE']):
            break
        chosen_genes = chosen_genes[chosen_genes['len']>=args['BEADSIZE_GENE']].reset_index(drop=True)
    return chosen_genes