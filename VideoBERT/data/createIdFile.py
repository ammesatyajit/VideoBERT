from tqdm import tqdm

global_ids_path = '/Users/ammesatyajit/Downloads/download_instructions/howto100m_videos.txt'
specific_ids_path = '/Users/ammesatyajit/Documents/GitHub/VideoBERT/VideoBERT/data/ids.txt'
save_path = '/Users/ammesatyajit/Downloads/filtered_ids.txt'

all_ids = {}
with open(global_ids_path, 'rt') as f:
    data = f.readlines()
    for line in tqdm(data):
        all_ids[line[31:42]] = line
print('global ids done')

with open(save_path, 'a') as w:
    with open(specific_ids_path, 'rt') as f:
        data = f.readlines()
        for line in tqdm(data):
            try:
                out = all_ids[line[:-1]]
                w.write(out)
            except:
                print('corrupted')

print('write done')
