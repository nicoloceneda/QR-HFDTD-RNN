Database Structure

wrds
 |
 |_ _ _ nyse
 |        |
 |        |_ _ _ sasdata
 |        |         |
 |        |         |_ _ _ taqms
 |        |         |        |_ _ _ cq
 |        |         |        |_ _ _ ct
 |        |         |        |_ _ _ mast
 |        |         |        |_ _ _ nbbo [Daily NBBO]
 |        |         |        |_ _ _ nbbod2m
 |        |         |        
 |        |         |_ _ _ wrds_taqms_iid         
 |        |         |         
 |        |         |_ _ _ wrds_taqms_nbbo 
 |        |         |         
 |        |         |_ _ _ wrds_taqms_wct 
 |        |         |
 |        |         |_ _ _ taqs 
 |        |         |        
 |        |         |_ _ _ wrds_taqs_ct [Monthly WCT]
 |        |         |         
 |        |         |_ _ _ wrds_taqs_nbbo [Monthly NBBO]
 |        |         |         
 |        |         |_ _ _ wrds_taqs_iid_v1
 |        |         |         
 |        |         |_ _ _ wrds_taqs_iid_v2
 |        |
 |        |_ _ _ taq_msecYYYY
 |                  |         
 |                  |_ _ _ mYYYYMM
 |                  |         
 |                  |_ _ _ wrds_iid
 |
 |_ _ _ taq
 |        |
 |        |_ _ _ sasdata
 |        |         |
 |        |         |_ _ _ cq
 |        |         |_ _ _ ct
 |        |         |_ _ _ mast
 |        |         |_ _ _ div
 |        |         |_ _ _ names, rgsh, ... 
 |        |
 |        |_ _ _ ...
 | 
 |_ _ _ taq.YYYY
          |
          |_ _ _ ....




- light: physical files
- bold: symbolic link files
- blue: TAQ daily product
- red: TAQ monthly product

/wrds/nyse/taq_msec2019/m200908 (CyberDuck):

-cqm_20190501.sas7bdat
-cqm_20190501.sas7bndx

-ctm_20190501.sas7bdat
-ctm_20190501.sas7bndx

-nbbom_20190501.sas7bdat
-nbbom_20190501.sas7bndx

-ix_cqm_20190501.sas7bdat
-ix_ctm_20190501.sas7bdat
-ix_nbbom_20190501.sas7bdat

-mastm_20190501.sas7bdat (introduced January 2010)

-nbbod2m_20190501.sas7bdat
-nbbod2m_20190501.sas7bdat

-complete_nbbo_20190501.sas7bdat
-complete_nbbo_20190501.sas7bndx

-wct_20190501.sas7bdat
-wct_20190501.sas7bndx	'taqm_2019’:

-'cqm_20190102' ----------> 'cqm_20190503'
-'ctm_20190102' ----------> 'ctm_20190503'
-'nbbom_20190102' --------> 'nbbom_20190503'

-'ix_cqm_20190102' -------> 'ix_cqm_20190503'
-'ix_ctm_20190102' -------> 'ix_ctm_20190503'
-'ix_nbbom_20190102', ----> 'ix_nbbom_20190503'

'taqmsec':

-'cqm_20030910' ----------> 'cqm_20180613'
-'ctm_20030910' ----------> 'ctm_20180613'
-'nbbom_20030910', -------> 'nbbom_20180613'
-'mastm_20130102', -------> 'mastm_20180605'

-'ix_cqm_20030910' -------> 'ix_cqm_20180613'
-'ix_ctm_20030910' -------> 'ix_ctm_20180613'
-'ix_nbbom_20030910', ----> 'ix_nbbom_20180613'
