# TODO: does this file make sense?


@torch.no_grad()
def get_embedding(nb, cell_id):
    start = time.time()
    cell = nb.loc[cell_id]
    tokens = unixcoder_model.tokenize([cell['source']],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens).to(device)
    _,embeddings = unixcoder_model(source_ids)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)[0].cpu()

def get_text_tokens(text):
    tokens = unixcoder_model.tokenize([text],max_length=512,mode="<encoder-only>")
    return torch.tensor(tokens).to(device)

def get_texts_tokens(texts):
    tokens = unixcoder_model.tokenize(texts,max_length=512,mode="<encoder-only>", padding=True)
    return torch.tensor(tokens).to(device)
    

def get_text_embedding(text):
    source_ids = get_text_tokens(text)
    _,embeddings = unixcoder_model(source_ids)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()[0]
    
@torch.no_grad()
def get_nb_embeddings(nb):
    start = time.time()

    res = {}

#     TODO: maybe different size?
    batch_size = 8
    n_chunks = len(nb) / min(len(nb), batch_size)

    nb = nb.sort_values(by="source", key=lambda x: x.str.len())
    for nb in np.array_split(nb, n_chunks):
        # TODO: different max_length?
        tokens = unixcoder_model.tokenize(nb['source'].to_numpy(),max_length=512,mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens).to(device)
        _,embeddings = unixcoder_model(source_ids)
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()

        
        for key, val in zip(nb['source'].index, normalized):
            res[key] = val
    
    return res


def get_source(cell_id):
    return nb.loc[cell_id]['source']

def sim(emb1, emb2):
    return torch.einsum("i,i->", emb1, emb2).detach().numpy()
