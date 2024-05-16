# %%
import pickle
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.flickr8k import Flickr8kDataset
from datasets.glove import embedding_matrix_creator
from metrics import *
from utils_torch import *

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
# %%

DATASET_BASE_PATH = 'data/flickr8k/'

sentence_bleu_score_fn = bleu_score_fn(4, 'sentence')
corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

# %%

train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                            return_type='tensor',
                            load_img_to_memory=False)
vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()
val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device,
                          return_type='corpus',
                          load_img_to_memory=False)
test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device,
                           return_type='corpus',
                           load_img_to_memory=False)
train_eval_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', vocab_set=vocab_set, device=device,
                                 return_type='corpus',
                                 load_img_to_memory=False)

model = torch.save(vocab, 'data/flickr8k/vocab.pt')

def evaluate_model(data_loader, model, loss_fn, vocab_size, bleu_score_fn, tensor_to_word_fn, desc=''):
    running_bleu = [0.0] * 5
    model.eval()
    t = tqdm(iter(data_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch
        outputs = tensor_to_word_fn(model.sample(images).cpu().numpy())

        for i in (1, 2, 3, 4):
            running_bleu[i] += bleu_score_fn(reference_corpus=captions, candidate_corpus=outputs, n=i)
        t.set_postfix({
            'bleu1': running_bleu[1] / (batch_idx + 1),
            'bleu4': running_bleu[4] / (batch_idx + 1),
        }, refresh=True)
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(data_loader)
    return running_bleu

# %%
t_i = 1003
dset = train_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(dset.get_image_captions(t_i)[1])

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
t_i = 500
dset = val_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(cp)

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
t_i = 500
dset = test_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(cp)

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
with torch.no_grad():
    model.eval()
    train_bleu = evaluate_model(desc=f'Train: ', model=final_model,
                                loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                tensor_to_word_fn=tensor_to_word_fn,
                                data_loader=train_eval_loader, vocab_size=vocab_size)
    val_bleu = evaluate_model(desc=f'Val: ', model=final_model,
                              loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                              tensor_to_word_fn=tensor_to_word_fn,
                              data_loader=val_loader, vocab_size=vocab_size)
    test_bleu = evaluate_model(desc=f'Test: ', model=final_model,
                               loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                               tensor_to_word_fn=tensor_to_word_fn,
                               data_loader=test_loader, vocab_size=vocab_size)
    for setname, result in zip(('train', 'val', 'test'), (train_bleu, val_bleu, test_bleu)):
        print(setname, end=' ')
        for ngram in (1, 2, 3, 4):
            print(f'Bleu-{ngram}: {result[ngram]}', end=' ')
            wandb.run.summary[f"{setname}_bleu{ngram}"] = result[ngram]
        print()
