import configparser
import math
import numpy as np
import pandas as pd
import os

from ToolBox.utils import start_logging,timer,do_job
import pyhibernator

config = configparser.ConfigParser()
config.read('./config.ini', encoding='utf-8')
SAVE_DIR = config.get('settings','SAVE_DIR')
WORK_DIR = config.get('settings','WORK_DIR')
SHARE_DIR = config.get('settings','SHARE_DIR')

settings = configparser.ConfigParser()
settings.read('./settings.ini', encoding='utf-8')
MIN_YEAR = int(settings.get('experiment','MIN_YEAR'))
MAX_YEAR = int(settings.get('experiment','MAX_YEAR'))
RESOLUTION = float(settings.get('experiment','RESOLUTION'))
NMIN = int(settings.get('experiment','NMIN'))
CPU_COUNT = int(settings.get('experiment','CPU_COUNT'))


LOGGER = start_logging(filename=WORK_DIR+"log/" + os.path.basename(__file__)[:-3] + '.log')

def normalize(citation,year,subj,mean_dic): 
    mu = mean_dic[(year,subj)]
    if mu == 0:
        return .0
    else:
        return citation/mu
    
def exe_delay_2016(c_history: list) -> float:
    """get pyhibernator.CitationDealy.score(c_history)
    
    Args:
        c_history (list): citation list from pub_year to MAX_YEAR
 
    Returns:
        float: citation delay score
    
    """
    return pyhibernator.CitationDelay.score(c = c_history)


if __name__ == '__main__':
    LOGGER.info('')
    LOGGER.info('logging start')
    os.makedirs(SAVE_DIR+'paper_detail_2016/',exist_ok=True)

    with do_job(__file__ + ' target data', LOGGER):
        if os.path.exists(SAVE_DIR+'paper_detail_2016/eid_2016.pickle'):
            pass
        else:
            papers = pd.concat([
                pd.read_pickle(SHARE_DIR+'paper_detail/eid.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/year.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/authids.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/doctype.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/journal.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/subjs.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/doi.pickle'),
            ],axis=1)
            papers_2016 = papers[papers['year']==2016].copy().reset_index(drop=True)
            
            for c in papers_2016.columns:
                papers_2016[c].to_pickle(SAVE_DIR+f'paper_detail_2016/{c}_2016.pickle')
            
    
    with do_job(__file__ + ' citation history',LOGGER):
        if os.path.exists(SAVE_DIR+'paper_detail_2016/c_history_2016.pickle'):
            pass
        else:
            paper_year_dic = pd.concat([
                pd.read_pickle(SHARE_DIR+'paper_detail/eid.pickle'),
                pd.read_pickle(SHARE_DIR+'paper_detail/year.pickle'),
            ],axis=1).set_index('eid')['year'].to_dict()
            
            source_dic = pd.read_pickle(SHARE_DIR + 'citations_gb.pickle')['source'].dropna().to_dict()
            
            papers_2016 = pd.concat([
                pd.read_pickle(SAVE_DIR+'paper_detail_2016/eid_2016.pickle'),
                pd.read_pickle(SAVE_DIR+'paper_detail_2016/year_2016.pickle'),
            ],axis=1)

            source_years_2016 = [np.array([paper_year_dic.get(eid,-1) for eid in source_dic.get(idx,[])]) for idx in papers_2016['eid']]
            LOGGER.info('source_years done')
            
            tot_citations_2016 = [len(l) for l in source_years_2016]
            pd.Series(tot_citations_2016, index=papers_2016.index, name='tot_citations').to_pickle(SAVE_DIR+'paper_detail_2016/tot_citations_2016.pickle')
            LOGGER.info('tot done')
            
            start_years_2016 = papers_2016['year'].values
            c_2016 = [[np.count_nonzero(s_years == year) for year in range(start_year,MAX_YEAR+1)] for start_year,s_years in zip(start_years_2016,source_years_2016)]
            LOGGER.info('c done')
            pd.Series(c_2016, index=papers_2016.index, name='c_history').to_pickle(SAVE_DIR+'paper_detail_2016/c_history_2016.pickle')

            del paper_year_dic, source_dic, source_years_2016, tot_citations_2016, start_years_2016, c_2016

    with do_job(__file__ + ' c_normalized', LOGGER):
        if os.path.exists(SAVE_DIR + f'paper_detail_2016/c_normalized_in_{MAX_YEAR}_{RESOLUTION}_{NMIN}_2016_waltman.pickle'):
            LOGGER.info(f'paper_detail_2016/c_normalized_in_{MAX_YEAR}_{RESOLUTION}_{NMIN}_2016_waltman.pickle exist')
        else:
            partitions_2016 = (pd.merge(
                    pd.merge(
                        pd.read_pickle(SHARE_DIR + 'paper_detail/year.pickle'),
                        pd.concat([
                            pd.read_pickle(SAVE_DIR+'paper_detail_2016/eid_2016.pickle'),
                            pd.read_pickle(SAVE_DIR+'paper_detail_2016/c_history_2016.pickle')
                        ],axis=1),
                        left_index=True, right_on='eid', how='right'
                    ).assign(n_cited=lambda df:df['c_history'].map(sum)),
                    pd.read_pickle(SAVE_DIR+f'paper_detail/partition_in_{MAX_YEAR}_{RESOLUTION}_{NMIN}_waltman.pickle'),
                    left_on='eid', right_index=True, how='left'
            ))

            mean_dic = (partitions_2016
                        [partitions_2016[f'partition_{RESOLUTION}']!=-1] 
                        .groupby(f'partition_{RESOLUTION}')
                        .agg(np.mean) 
                        .fillna(np.inf).replace(0,np.inf) 
            )['n_cited'].to_dict()

            partitions_2016[f'c_normalized_{RESOLUTION}'] = [c/mean_dic.get(partition,c) if c!=0 else 0 for partition,c in partitions_2016[[f'partition_{RESOLUTION}','n_cited']].values] 
            partitions_2016[f'c_normalized_{RESOLUTION}'].to_pickle(SAVE_DIR + f'paper_detail_2016/c_normalized_in_{MAX_YEAR}_{RESOLUTION}_{NMIN}_2016_waltman.pickle')

            del partitions_2016, mean_dic


    with do_job(__file__ + ' citation delay',LOGGER):
        
        if os.path.exists(SAVE_DIR+'paper_detail_2016/CD_2016.pickle'):
            pass
        else:
            papers_2016 = pd.concat([
                pd.read_pickle(SAVE_DIR+'paper_detail_2016/eid_2016.pickle'),
                pd.read_pickle(SAVE_DIR+'paper_detail_2016/c_history_2016.pickle')
            ],axis=1)

            pd.Series([exe_delay_2016(c_history=l) for l in papers_2016['c_history']], index=papers_2016.index, name=f'CD').to_pickle(SAVE_DIR+f'paper_detail_2016/CD_2016.pickle')
