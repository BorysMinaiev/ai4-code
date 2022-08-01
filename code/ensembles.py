from dataclasses import dataclass
import torch
from common import get_probs_by_embeddings, get_best_pos_by_probs
from state import State
from graph_model import MyGraphModel
import graph_model
import unixcoder


@dataclass
class Sample:
    text: str
    md_cell_id: str
    graph3_pos: float
    unix_pos: float
    total_cells: int
    md_cells: int
    code_cells: int
    part_code_cells: float
    target_pos: float


def gen_nb_samples(nb, graph3_embeddings, unix_embeddings, correct_order):
    code_cells = nb[nb['cell_type'] == 'code'].reset_index(level='cell_id')
    markdown_cells = nb[nb['cell_type'] != 'code'].reset_index(level='cell_id')

    code_cell_ids = code_cells['cell_id'].values.tolist()
    code_cell_ids.append('END')

    samples = []

    md_cells = len(markdown_cells)
    code_cells = len(code_cells)
    total_cells = md_cells + code_cells
    part_code_cells = code_cells / total_cells

    for m_cell_id in markdown_cells['cell_id'].values:
        text = nb.loc[m_cell_id]['source']
        graph_sims_probs = get_probs_by_embeddings(
            graph3_embeddings, m_cell_id, code_cell_ids, 25.0)
        unix_sims_probs = get_probs_by_embeddings(
            unix_embeddings, m_cell_id, code_cell_ids, 1000.0)

        graph3_pos = get_best_pos_by_probs(graph_sims_probs)
        unix_pos = get_best_pos_by_probs(unix_sims_probs)

        target_pos = 0 if correct_order is None else correct_order.index(
            m_cell_id) / total_cells * code_cells
        samples.append(Sample(md_cell_id=m_cell_id, text=text, graph3_pos=graph3_pos, unix_pos=unix_pos, total_cells=total_cells,
                       md_cells=md_cells, code_cells=code_cells, part_code_cells=part_code_cells, target_pos=target_pos))

    return samples


@torch.no_grad()
def gen_samples(state: State, nb, graph3_model: MyGraphModel, unixcoder_model, correct_order):
    graph3_embeddings = graph_model.get_nb_embeddings(state, graph3_model, nb)
    unix_embeddings = unixcoder.get_nb_embeddings(state, unixcoder_model, nb)
    return gen_nb_samples(nb, graph3_embeddings, unix_embeddings, correct_order)


def predict(state: State, ensemble_model, samples):
    texts = list(map(lambda s: s.text, samples))
    additional_features = list(map(lambda s: torch.FloatTensor(
        [s.graph3_pos, s.unix_pos, s.total_cells, s.md_cells, s.code_cells, s.part_code_cells]), samples))
    additional_features = torch.stack(additional_features).to(state.device)

    to_mul = list(map(lambda s: torch.FloatTensor(
        [s.graph3_pos, s.unix_pos]), samples))
    to_mul = torch.stack(to_mul).to(state.device)

    coefs = None
    if state.config.use_simple_ensemble_model:
        coefs = ensemble_model(additional_features)
    else:
        text_tokens = ensemble_model.encoder.tokenize(
            texts, max_length=512, mode="<encoder-only>", padding=True)
        text_tokens = torch.tensor(text_tokens).to(state.device)
        coefs = ensemble_model(text_tokens, additional_features, state.device)
    preds = torch.einsum("ab,ab->a", coefs, to_mul)

    return {'coefs': coefs, 'preds': preds}
