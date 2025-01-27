from datasets import Dataset
from datasets import load_dataset


# fineweb_2 = Dataset.to_pandas(load_dataset("HuggingFaceFW/fineweb-2", name="sin_Sinh", split="train"))
# fineweb_2_processed = fineweb_2[['text']]
# fineweb_2_processed['source'] = "fineweb_2"
# fineweb_2_processed.to_csv('fineweb_2_processed.tsv', index=False, sep="\t", encoding='utf-8',)
#
#
# hplt_2 = Dataset.to_pandas(load_dataset("HPLT/HPLT2.0_cleaned", name="sin_Sinh", split="train"))
# hplt_2_processed = hplt_2[['text']]
# hplt_2_processed['source'] = "hplt_2"
# hplt_2_processed.to_csv('hplt_2_processed.tsv', index=False, sep="\t", encoding='utf-8',)

# CulturaX = Dataset.to_pandas(load_dataset("uonlp/CulturaX", "si", split="train",
#                   use_auth_token=True))
# CulturaX_processed = CulturaX[['text']]
# CulturaX_processed['source'] = "CulturaX"
# CulturaX_processed.to_csv('CulturaX_processed.tsv', index=False, sep="\t", encoding='utf-8',)

Nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
Nsina_processed = Nsina[['News Content']]
Nsina_processed = Nsina_processed.rename(columns={'News Content': 'text'})
Nsina_processed['source'] = "Nsina"
Nsina_processed.to_csv('Nsina_processed.tsv', index=False, sep="\t", encoding='utf-8',)



# hplt = Dataset.to_pandas(load_dataset("HPLT/HPLT2.0_cleaned", name="sin_Sinh", split="train"))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/sinmin', split='train'))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/sinmin', split='train'))

