import sys; sys.path.insert(0, '.')
from data.datasets.iemocap import IEMOCAPDataset
ds = IEMOCAPDataset('data/raw/IEMOCAP/IEMOCAP_full_release', num_classes=4)
ds.load()
utts = ds.get_flat_utterances()
empty = sum(1 for u in utts if not u.text)
print(f'Total: {len(utts)} utterances')
print(f'With text: {len(utts) - empty}, Empty text: {empty}')
if utts:
    print('\nSample utterances:')
    for u in utts[:5]:
        print(f'  [{u.utterance_id}] emotion={u.emotion} speaker={u.speaker}')
        print(f'    text="{u.text[:100]}"')
        print(f'    wav={u.wav_path}')
print()
ds.print_stats()
