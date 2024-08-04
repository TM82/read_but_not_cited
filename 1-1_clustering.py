import argparse
import configparser
import os
import sys
import time
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np
import pandas as pd
import igraph
import leidenalg

sys.path.append('/disks/qnap2/notebook/t-miura/modules/ToolBox')
from ToolBox.utils import start_logging,timer,do_job

config = configparser.ConfigParser()
config.read('./config.ini', encoding='utf-8')
SAVE_DIR = config.get('settings','SAVE_DIR')
WORK_DIR = config.get('settings','WORK_DIR')
SHARE_DIR = config.get('settings','SHARE_DIR')

settings = configparser.ConfigParser()
settings.read('./settings.ini', encoding='utf-8')
MIN_YEAR = int(settings.get('experiment','MIN_YEAR'))
MAX_YEAR = int(settings.get('experiment','MAX_YEAR'))
CPU_COUNT = int(settings.get('experiment','CPU_COUNT'))

LOGGER = start_logging(filename=WORK_DIR+"log/" + os.path.basename(__file__)[:-3] + '.log')


### ==========clustering==========
def do_leidenpartitioning(df_edges, resolution):
    """ Leiden partitioning with CPM model

    Args:
        df_edges (pd.DataFrame): edges(indirect)
        resolution (float): resolution parameter

    Returns:
        pd.DataFrame: index=eid, column=partition
    """    
    g = igraph.Graph.TupleList(df_edges.itertuples(index=False), directed=False) # Including not maximum component(extract as 1 cluster)
    g_nodenames = [g['name'] for g in g.vs]
    partition = leidenalg.find_partition(
        graph = g, 
        partition_type = leidenalg.CPMVertexPartition,
        resolution_parameter = resolution,
        seed=0
    )
    partition_df = pd.DataFrame({"node":g_nodenames,"partition":partition.membership}).set_index('node')
    partition_df.index.rename('eid', inplace=True)
    partition_df.rename(columns={'partition': f'partition_{resolution}'}, inplace=True)
    return partition_df

def get_frac_citation(df):
    """
    Calculation of association, (fractional weight)
    """
    frac_refs_dic = df['source'].value_counts().map(lambda x: 1/x).to_dict()
    df['frac_c'] = df['source'].map(frac_refs_dic)
    return df


def get_large(s_idx):
    if s_idx in not_merged_idx:
        return -1
    return large_cluster_idx[np.argmax([aij_sum.get((s_idx,l_idx),aij_sum.get((l_idx,s_idx),0)) / n_edge_sum.get((s_idx,l_idx),n_edge_sum.get((l_idx,s_idx),1)) for l_idx in large_cluster_idx])]


if __name__ == '__main__':
    LOGGER.info('')
    LOGGER.info('logging start')
    os.makedirs(SAVE_DIR+'paper_detail/',exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,required=True,choices=['field','topic'], help='choose "field" or "topic" to set RESOLUTION and NMIN')
    MODE = parser.parse_args().mode
    
    if MODE == 'field':
        LOGGER.info('Mode: field')
        RESOLUTION = float(settings.get('experiment','RESOLUTION'))
        NMIN = int(settings.get('experiment','NMIN'))
    elif MODE == 'topic':
        LOGGER.info('MODE: Topic')
        RESOLUTION = float(settings.get('experiment','RESOLUTION_TOPIC'))
        NMIN = int(settings.get('experiment','NMIN_TOPIC'))
    else:
        raise
    
    LOGGER.info(f'RESOLUTION: {RESOLUTION}, NMIN: {NMIN}')    
    
    with do_job(__file__ + ' get_partition', LOGGER):
        citations = pd.read_pickle(f'/disks/qnap2/data/t-miura/2022_fieldmerge/SIGMET/papers/{MAX_YEAR}/citations.pickle') 
        partition = do_leidenpartitioning(
            df_edges = citations,
            resolution = RESOLUTION
        )
        partition.to_pickle(SAVE_DIR + f'paper_detail/partition_in_{MAX_YEAR}_{RESOLUTION}.pickle')
    
    with do_job(__file__ + ' prepare', LOGGER):
        citations = pd.read_pickle(f'/disks/qnap2/data/t-miura/2022_fieldmerge/SIGMET/papers/{MAX_YEAR}/citations.pickle')
        partition = pd.read_pickle(SAVE_DIR + f'paper_detail/partition_in_{MAX_YEAR}_{RESOLUTION}.pickle')[f'partition_{RESOLUTION}']
        LOGGER.info(f"all parititon: {partition.nunique()}")
        LOGGER.info(f"small parititon (<{NMIN}): {(partition.value_counts() < NMIN).sum()}")
        partition_dic = partition.to_dict()

        citations = (citations
                    .assign(source_cluster=lambda df: df['source'].map(partition_dic))
                    .assign(target_cluster=lambda df: df['target'].map(partition_dic))
                    .pipe(get_frac_citation)
        )

        c_size_dic = partition.value_counts().to_dict()
        large_cluster_idx = [i for i in partition.unique() if c_size_dic[i] >= NMIN]
        small_cluster_idx = [i for i in partition.unique() if c_size_dic[i] < NMIN]
        
        # association(weight of edge)
        aij_sum = citations[['source_cluster','target_cluster','frac_c']].groupby(['source_cluster','target_cluster'])['frac_c'].agg(sum).to_dict()
        n_edge_sum = citations.value_counts(['source_cluster','target_cluster']).to_dict()

        _1 = citations[(citations['source_cluster'].isin(small_cluster_idx))&(citations['target_cluster'].isin(large_cluster_idx))]
        _2 = citations[(citations['source_cluster'].isin(large_cluster_idx))&(citations['target_cluster'].isin(small_cluster_idx))]
        not_merged_idx = set(small_cluster_idx) - set(_1['source_cluster']) - set(_2['target_cluster'])
        LOGGER.info(f"not merged parititon: {len(not_merged_idx)}")
        
    with do_job(__file__ + ' merge',LOGGER):
        print('start')
        with Pool(CPU_COUNT) as p:
            vals = list(tqdm(p.imap(get_large, small_cluster_idx),total=len(small_cluster_idx)))
        print('done', len(vals))
        small_to_large_dic = {s_idx:l_idx for s_idx,l_idx in zip(small_cluster_idx,vals)}

        partition = partition.map(lambda idx: small_to_large_dic.get(idx,idx))
        partition_name_dic = {v:k for k,v in enumerate(partition[partition!=-1].value_counts().index)}
        partition = pd.concat([partition[partition==-1],partition[partition!=-1].map(partition_name_dic)]).map(int).reindex(partition.index)
        
        _ = len(partition[partition==-1])
        LOGGER.info(f"# parititon=-1: {_}/{len(partition)}")
        LOGGER.info(f"unique parititon: {partition.nunique()-1}")

        partition.to_pickle(SAVE_DIR+f'paper_detail/partition_in_{MAX_YEAR}_{RESOLUTION}_{NMIN}_waltman_231115_raw.pickle')
