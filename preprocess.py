from datasets import Dataset
from datasets import load_dataset
import pandas as pd


# fineweb_2 = Dataset.to_pandas(load_dataset("HuggingFaceFW/fineweb-2", name="sin_Sinh", split="train"))
# fineweb_2_processed = fineweb_2[['text']]
# fineweb_2_processed['source'] = "fineweb_2"
# fineweb_2_processed.to_csv('fineweb_2_processed.tsv', index=False, sep="\t", encoding='utf-8',)
#
#
hplt_2 = Dataset.to_pandas(load_dataset("HPLT/HPLT2.0_cleaned", name="sin_Sinh", split="train"))
hplt_2_processed = hplt_2[['text']]
hplt_2_processed['source'] = "hplt_2"

CulturaX = Dataset.to_pandas(load_dataset("uonlp/CulturaX", "si", split="train",
                  use_auth_token=True))
CulturaX_processed = CulturaX[['text']]
CulturaX_processed['source'] = "CulturaX"

Nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
Nsina_processed = Nsina[['News Content']]
Nsina_processed = Nsina_processed.rename(columns={'News Content': 'text'})
Nsina_processed['source'] = "Nsina"
# Nsina_processed.to_csv('Nsina_processed.tsv', index=False, sep="\t", encoding='utf-8')

sinmin = Dataset.to_pandas(load_dataset('sinhala-nlp/sinmin', split='train'))
sinmin_processed = sinmin[['content']]
sinmin_processed = sinmin_processed.rename(columns={'content': 'text'})
sinmin_processed['source'] = "sinmin"


FacebookDecadeCorpora = Dataset.to_pandas(load_dataset('sinhala-nlp/FacebookDecadeCorpora', split='train'))
FacebookDecadeCorpora_processed = FacebookDecadeCorpora[['Message']]
FacebookDecadeCorpora_processed = FacebookDecadeCorpora_processed.rename(columns={'Message': 'text'})
FacebookDecadeCorpora_processed['source'] = "FacebookDecadeCorpora"

semisold = Dataset.to_pandas(load_dataset("sinhala-nlp/SemiSOLD", split="train"))
semisold = semisold[semisold['xlmr'] < 0.5]
semisold_processed = semisold[['text']]
semisold_processed['source'] = "semisold"


all_df = [hplt_2_processed, CulturaX_processed, Nsina_processed, sinmin_processed, FacebookDecadeCorpora_processed, semisold_processed]
final_df = pd.concat(all_df, ignore_index=True)
final_df.to_csv('train.tsv', index=False, sep="\t", encoding='utf-8')






