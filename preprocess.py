from datasets import Dataset
from datasets import load_dataset


fineweb_2 = Dataset.to_pandas(load_dataset("HuggingFaceFW/fineweb-2", name="sin_Sinh", split="train"))
fineweb_2_processed = fineweb_2[['text']]
fineweb_2_processed['source'] = "fineweb_2"
fineweb_2_processed.to_csv('fineweb_2_processed.tsv', index=False, sep="\t", encoding='utf-8',)






# hplt = Dataset.to_pandas(load_dataset("HPLT/HPLT2.0_cleaned", name="sin_Sinh", split="train"))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/sinmin', split='train'))
#
# nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/sinmin', split='train'))

