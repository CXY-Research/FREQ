import torch
import torch.nn as nn
import math
from typing import List, Optional, Union


class Attn(nn.Module):

    def __init__(self, dim: int, num_head: int, dropout: float = 0):
        super(Attn, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.dim_per_head = dim // num_head
        self.drop = nn.Dropout(p=dropout)
        self.P = nn.Linear(dim, dim, bias=True)
        self.V = nn.Linear(dim, dim, bias=True)
        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=True)
        self.a = nn.Parameter(torch.zeros((1, num_head, self.dim_per_head)))
        self.act = nn.PReLU()
        self.emb_ln = nn.LayerNorm(dim)
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.P.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.V.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.Q.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.K.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.a, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.P.bias)
        nn.init.zeros_(self.V.bias)
        nn.init.zeros_(self.Q.bias)
        nn.init.zeros_(self.K.bias)

    def forward(
        self,
        query: torch.FloatTensor,
        keys: List[torch.FloatTensor],
        values: List[torch.FloatTensor],
        query_len: Union[int, torch.LongTensor],
        self_attn: bool = False,
        attn_idxs: Optional[List[torch.LongTensor]] = None,
        query_idxs: Optional[List[torch.LongTensor]] = None,
        key_idxs: Optional[List[torch.LongTensor]] = None,
        value_idxs: Optional[List[torch.LongTensor]] = None,
        pair_weights: Optional[List[torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
        if query_idxs is None:
            query_idxs = [None for _ in range(len(keys))]
        if key_idxs is None:
            key_idxs = [None for _ in range(len(keys))]
        if value_idxs is None:
            value_idxs = [None for _ in range(len(values))]
        if pair_weights is None:
            pair_weights = [None for _ in range(len(keys))]
        if attn_idxs is None:
            attn_idxs = [None for _ in range(len(keys))]
        if self_attn:
            attn_query_per_query = (
                self.a
                * self.act(self.Q(query) + self.K(query)).reshape(
                    -1, self.num_head, self.dim_per_head
                )
            ).sum(dim=-1, keepdim=True)
        attn_query_per_key = []
        for key, query_idx, key_idx, attn_idx in zip(
            keys, query_idxs, key_idxs, attn_idxs
        ):
            if (query_idx is not None and len(query) > 1) and key_idx is not None:
                if len(query) < len(query_idx) and len(key) < len(key_idx):
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                torch.index_select(self.Q(query), 0, query_idx)
                                + torch.index_select(self.K(key), 0, key_idx)
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
                elif len(query) < len(query_idx) and len(key) >= len(key_idx):
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                torch.index_select(self.Q(query), 0, query_idx)
                                + self.K(torch.index_select(key, 0, key_idx))
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
                elif len(query) >= len(query_idx) and len(key) < len(key_idx):
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                self.Q(torch.index_select(query, 0, query_idx))
                                + torch.index_select(self.K(key), 0, key_idx)
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
                else:
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                self.Q(torch.index_select(query, 0, query_idx))
                                + self.K(torch.index_select(key, 0, key_idx))
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
            elif (query_idx is None or len(query) == 1) and key_idx is not None:
                if len(key) < len(key_idx):
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                self.Q(query)
                                + torch.index_select(self.K(key), 0, key_idx)
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
                else:
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                self.Q(query)
                                + self.K(torch.index_select(key, 0, key_idx))
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
            elif (query_idx is not None and len(query) > 1) and key_idx is None:
                if len(query) < len(query_idx):
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                torch.index_select(self.Q(query), 0, query_idx)
                                + self.K(key)
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
                else:
                    attn_query_per_key.append(
                        (
                            self.a
                            * self.act(
                                self.Q(torch.index_select(query, 0, query_idx))
                                + self.K(key)
                            ).reshape(-1, self.num_head, self.dim_per_head)
                        ).sum(dim=-1, keepdim=True)
                    )
            else:
                attn_query_per_key.append(
                    (
                        self.a
                        * self.act(self.Q(query) + self.K(key)).reshape(
                            -1, self.num_head, self.dim_per_head
                        )
                    ).sum(dim=-1, keepdim=True)
                )
            if attn_idx is not None:
                attn_query_per_key[-1] = attn_query_per_key[-1][attn_idx]
        if self_attn:
            max_attn_per_query = attn_query_per_query
        else:
            min_attn = min(
                [torch.min(attn) for attn in attn_query_per_key if len(attn) > 0]
            )
            max_attn_per_query = (
                min_attn * torch.ones((query_len, self.num_head, 1)).cuda()
            )
        for attn, query_idx in zip(attn_query_per_key, query_idxs):
            if query_idx is not None:
                max_attn_per_query = max_attn_per_query.index_reduce(
                    dim=0,
                    index=query_idx,
                    source=attn,
                    reduce="amax",
                    include_self=True,
                ).detach()
            else:
                max_attn_per_query = torch.maximum(max_attn_per_query, attn).detach()
        if self_attn:
            exp_attn_query_per_query = torch.exp(
                attn_query_per_query - max_attn_per_query
            )
            exp_attn_per_query = exp_attn_query_per_query
        else:
            exp_attn_per_query = torch.zeros((query_len, self.num_head, 1)).cuda()
        exp_attn_query_per_key = []
        for attn, query_idx, pair_weight in zip(
            attn_query_per_key, query_idxs, pair_weights
        ):
            if pair_weight is None:
                pair_weight = torch.ones_like(attn)
            if query_idx is not None:
                exp_attn_query_per_key.append(
                    torch.exp(attn - max_attn_per_query[query_idx])
                )
                exp_attn_per_query = exp_attn_per_query.index_add(
                    dim=0,
                    index=query_idx,
                    source=pair_weight * exp_attn_query_per_key[-1],
                )
            else:
                exp_attn_query_per_key.append(torch.exp(attn - max_attn_per_query))
                exp_attn_per_query = (
                    exp_attn_per_query + pair_weight * exp_attn_query_per_key[-1]
                )
        exp_attn_per_query = torch.where(exp_attn_per_query == 0, 1, exp_attn_per_query)
        if self_attn:
            msg_to_query = exp_attn_query_per_query * self.V(query).reshape(
                -1, self.num_head, self.dim_per_head
            )
        else:
            msg_to_query = torch.zeros(
                (query_len, self.num_head, self.dim_per_head)
            ).cuda()
        for exp_attn, query_idx, value, value_idx, pair_weight in zip(
            exp_attn_query_per_key, query_idxs, values, value_idxs, pair_weights
        ):
            if pair_weight is None:
                pair_weight = torch.ones_like(exp_attn)
            if query_idx is not None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = msg_to_query.index_add(
                        dim=0,
                        index=query_idx,
                        source=pair_weight
                        * exp_attn
                        * torch.index_select(self.V(value), 0, value_idx).reshape(
                            -1, self.num_head, self.dim_per_head
                        ),
                    )
                else:
                    msg_to_query = msg_to_query.index_add(
                        dim=0,
                        index=query_idx,
                        source=pair_weight
                        * exp_attn
                        * self.V(torch.index_select(value, 0, value_idx)).reshape(
                            -1, self.num_head, self.dim_per_head
                        ),
                    )
            elif query_idx is None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = (
                        msg_to_query
                        + pair_weight
                        * exp_attn
                        * torch.index_select(self.V(value), 0, value_idx).reshape(
                            -1, self.num_head, self.dim_per_head
                        )
                    )
                else:
                    msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(
                        torch.index_select(value, 0, value_idx)
                    ).reshape(-1, self.num_head, self.dim_per_head)
            elif query_idx is not None and value_idx is None:
                msg_to_query = msg_to_query.index_add(
                    dim=0,
                    index=query_idx,
                    source=pair_weight
                    * exp_attn
                    * self.V(value).reshape(-1, self.num_head, self.dim_per_head),
                )
            else:
                msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(
                    value
                ).reshape(-1, self.num_head, self.dim_per_head)
        new_query = self.emb_ln(
            query
            + self.drop(
                self.P((msg_to_query / exp_attn_per_query).flatten(start_dim=1))
            )
        )
        return new_query


class Init_Layer(nn.Module):

    def __init__(self, dim, logger, dropout=0.1):
        super(Init_Layer, self).__init__()
        self.dim = dim
        self.drop = nn.Dropout(p=dropout)
        self.logger = logger
        self.proj_he2e = nn.Linear(dim, dim, bias=True)
        self.proj_te2e = nn.Linear(dim, dim, bias=True)
        self.proj_qe2e = nn.Linear(dim, dim, bias=True)
        self.proj_he2pr = nn.Linear(dim, dim, bias=True)
        self.proj_te2pr = nn.Linear(dim, dim, bias=True)
        self.proj_qe2qr = nn.Linear(dim, dim, bias=True)
        self.proj_pr2he = nn.Linear(dim, dim, bias=True)
        self.proj_pr2te = nn.Linear(dim, dim, bias=True)
        self.proj_qr2qe = nn.Linear(dim, dim, bias=True)
        self.proj_pr2r = nn.Linear(dim, dim, bias=True)
        self.proj_qr2r = nn.Linear(dim, dim, bias=True)
        self.proj_fe2he = nn.Linear(dim, dim, bias=True)
        self.proj_fe2te = nn.Linear(dim, dim, bias=True)
        self.proj_fe2qe = nn.Linear(dim, dim, bias=True)
        self.proj_fr2pr = nn.Linear(dim, dim, bias=True)
        self.proj_fr2qr = nn.Linear(dim, dim, bias=True)
        self.emb_ent_ln = nn.LayerNorm(dim)
        self.emb_rel_ln = nn.LayerNorm(dim)
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(
            self.proj_he2e.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_te2e.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qe2e.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_he2pr.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_te2pr.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qe2qr.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_pr2he.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_pr2te.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qr2qe.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_pr2r.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qr2r.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fe2he.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fe2te.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fe2qe.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fr2pr.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fr2qr.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.zeros_(self.proj_he2e.bias)
        nn.init.zeros_(self.proj_te2e.bias)
        nn.init.zeros_(self.proj_qe2e.bias)
        nn.init.zeros_(self.proj_he2pr.bias)
        nn.init.zeros_(self.proj_te2pr.bias)
        nn.init.zeros_(self.proj_qe2qr.bias)
        nn.init.zeros_(self.proj_pr2he.bias)
        nn.init.zeros_(self.proj_pr2te.bias)
        nn.init.zeros_(self.proj_qr2qe.bias)
        nn.init.zeros_(self.proj_pr2r.bias)
        nn.init.zeros_(self.proj_qr2r.bias)
        nn.init.zeros_(self.proj_fe2he.bias)
        nn.init.zeros_(self.proj_fe2te.bias)
        nn.init.zeros_(self.proj_fe2qe.bias)
        nn.init.zeros_(self.proj_fr2pr.bias)
        nn.init.zeros_(self.proj_fr2qr.bias)

    def forward(self, emb_ent, emb_rel, pri, qual2fact, qual):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()
        msg_he2e = torch.index_select(self.proj_he2e(emb_ent), 0, heads)
        msg_te2e = torch.index_select(self.proj_te2e(emb_ent), 0, tails)
        msg_qe2e = torch.index_select(self.proj_qe2e(emb_ent), 0, qual_ents)
        msg_fe2e = (msg_he2e + msg_te2e).index_add(
            dim=0, index=qual2fact, source=msg_qe2e
        )
        fe_cnt = torch.bincount(qual2fact, minlength=len(pri)).float() + 2
        msg_he2pr = torch.index_select(self.proj_he2pr(emb_ent), 0, heads)
        msg_te2pr = torch.index_select(self.proj_te2pr(emb_ent), 0, tails)
        msg_qe2qr = torch.index_select(self.proj_qe2qr(emb_ent), 0, qual_ents)
        msg_pr2he = torch.index_select(self.proj_pr2he(emb_rel), 0, rels)
        msg_pr2te = torch.index_select(self.proj_pr2te(emb_rel), 0, rels)
        msg_qr2qe = torch.index_select(self.proj_qr2qe(emb_rel), 0, qual_rels)
        msg_pr2r = torch.index_select(self.proj_pr2r(emb_rel), 0, rels)
        msg_qr2r = torch.index_select(self.proj_qr2r(emb_rel), 0, qual_rels)
        msg_fr2r = msg_pr2r.index_add(dim=0, index=qual2fact, source=msg_qr2r)
        fr_cnt = fe_cnt - 1
        zero4ent = torch.zeros_like(emb_ent)
        idx4e = torch.cat([heads, tails], dim=0)
        src4e2e = torch.cat(
            [
                self.proj_fe2he(msg_fe2e - msg_he2e),
                self.proj_fe2te(msg_fe2e - msg_te2e),
            ],
            dim=0,
        )
        src4e2ecnt = torch.cat([fe_cnt - 1, fe_cnt - 1], dim=0)
        src4r2e = torch.cat([msg_pr2he, msg_pr2te], dim=0)
        if len(qual2fact) > 0:
            idx4e = torch.cat([idx4e, qual_ents], dim=0)
            src4e2e = torch.cat(
                [
                    src4e2e,
                    self.proj_fe2qe(
                        torch.index_select(msg_fe2e, 0, qual2fact) - msg_qe2e
                    ),
                ],
                dim=0,
            )
            src4e2ecnt = torch.cat(
                [src4e2ecnt, torch.index_select(fe_cnt - 1, 0, qual2fact)], dim=0
            )
            src4r2e = torch.cat([src4r2e, msg_qr2qe], dim=0)
        e2e = zero4ent.index_add(dim=0, index=idx4e, source=src4e2e)
        e2e_cnt = (
            torch.zeros(len(emb_ent))
            .cuda()
            .index_add(dim=0, index=idx4e, source=src4e2ecnt)
            .unsqueeze(dim=1)
        )
        e2e_cnt = torch.where(e2e_cnt == 0, 1, e2e_cnt)
        r2e = zero4ent.index_reduce(
            dim=0, index=idx4e, source=src4r2e, reduce="mean", include_self=False
        )
        new_emb_ent = self.emb_ent_ln(
            emb_ent + self.drop(e2e / e2e_cnt) + self.drop(r2e)
        )
        zero4rel = torch.zeros_like(emb_rel)
        if len(qual2fact) > 0:
            r2r = zero4rel.index_add(
                dim=0,
                index=torch.cat([rels, qual_rels], dim=0),
                source=torch.cat(
                    [
                        self.proj_fr2pr(msg_fr2r - msg_pr2r),
                        self.proj_fr2qr(
                            torch.index_select(msg_fr2r, 0, qual2fact) - msg_qr2r
                        ),
                    ],
                    dim=0,
                ),
            )
            r2r_cnt = (
                torch.zeros(len(emb_rel))
                .cuda()
                .index_add(
                    dim=0,
                    index=torch.cat([rels, qual_rels], dim=0),
                    source=torch.cat(
                        [fr_cnt - 1, torch.index_select(fr_cnt - 1, 0, qual2fact)],
                        dim=0,
                    ),
                )
                .unsqueeze(dim=1)
            )
            r2r_cnt = torch.where(r2r_cnt == 0, 1, r2r_cnt)
            e2r = zero4rel.index_reduce(
                dim=0,
                index=torch.cat([rels, rels, qual_rels], dim=0),
                source=torch.cat([msg_he2pr, msg_te2pr, msg_qe2qr], dim=0),
                reduce="mean",
                include_self=False,
            )
            new_emb_rel = self.emb_rel_ln(
                emb_rel + self.drop(r2r / r2r_cnt) + self.drop(e2r)
            )
        else:
            e2r = zero4rel.index_reduce(
                dim=0,
                index=torch.cat([rels, rels], dim=0),
                source=torch.cat([msg_he2pr, msg_te2pr], dim=0),
                reduce="mean",
                include_self=False,
            )
            new_emb_rel = self.emb_rel_ln(emb_rel + self.drop(e2r))
        return (new_emb_ent, new_emb_rel)


class ANMP_Layer(nn.Module):

    def __init__(
        self,
        dim,
        num_head,
        logger,
        dropout=0.2,
        use_rcqg=False,
        use_daqg=False,
        use_rcqg_bilinear=False,
        use_random_pair_weight=False,
        use_idf=True,
        use_idf_hpair=True,
        use_idf_tpair=True,
        use_idf_qpair=True,
        idf_weight_form="log",
    ):
        super(ANMP_Layer, self).__init__()
        self.dim = dim
        self.dim_per_head = dim // num_head
        assert self.dim_per_head * num_head == dim
        self.num_head = num_head
        self.drop = nn.Dropout(p=dropout)
        self.use_rcqg = use_rcqg
        self.use_daqg = use_daqg
        self.use_rcqg_bilinear = use_rcqg_bilinear
        self.use_random_pair_weight = use_random_pair_weight
        self.use_idf = use_idf
        self.use_idf_hpair = use_idf_hpair
        self.use_idf_tpair = use_idf_tpair
        self.use_idf_qpair = use_idf_qpair
        self.idf_weight_form = idf_weight_form
        if self.idf_weight_form not in {"log", "linear", "sqrt"}:
            raise ValueError(
                "idf_weight_form must be one of {'log', 'linear', 'sqrt'}."
            )
        self.init_emb_fact = nn.Parameter(torch.zeros(1, dim))
        self.init_emb_fact_ln = nn.LayerNorm(dim)
        self.fact_ln = nn.LayerNorm(dim)
        self.proj_head_ent = nn.Linear(dim, dim, bias=True)
        self.proj_tail_ent = nn.Linear(dim, dim, bias=True)
        self.proj_qual_ent = nn.Linear(dim, dim, bias=True)
        self.proj_head_rel = nn.Linear(dim, dim, bias=True)
        self.proj_tail_rel = nn.Linear(dim, dim, bias=True)
        self.proj_qual_rel = nn.Linear(dim, dim, bias=True)
        self.proj_hpair_to_fact = nn.Linear(dim, dim, bias=True)
        self.proj_tpair_to_fact = nn.Linear(dim, dim, bias=True)
        self.proj_qpair_to_fact = nn.Linear(dim, dim, bias=True)
        self.proj_head_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_tail_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_qual_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_head_ent2 = nn.Linear(dim, dim, bias=True)
        self.proj_tail_ent2 = nn.Linear(dim, dim, bias=True)
        self.proj_qual_ent2 = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_head_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_tail_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_pri_rel = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_qual_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_qual_rel = nn.Linear(dim, dim, bias=True)
        self.rcqg_compat_W = (
            nn.Linear(dim, dim, bias=False) if self.use_rcqg_bilinear else None
        )
        self.ent_attn = Attn(dim, num_head, dropout=dropout)
        self.rel_attn = Attn(dim, num_head, dropout=dropout)
        self.to_fact_ln = nn.LayerNorm(dim)
        self.to_ent_ln = nn.LayerNorm(dim)
        self.to_rel_ln = nn.LayerNorm(dim)
        self.logger = logger
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_fact, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(
            self.proj_head_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_tail_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qual_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_head_rel.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_tail_rel.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qual_rel.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_head_ent2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_tail_ent2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qual_ent2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_head_rel2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_tail_rel2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qual_rel2.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_hpair_to_fact.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_tpair_to_fact.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_qpair_to_fact.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fact_to_head_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fact_to_tail_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fact_to_pri_rel.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fact_to_qual_ent.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.proj_fact_to_qual_rel.weight, gain=nn.init.calculate_gain("relu")
        )
        if self.rcqg_compat_W is not None:
            nn.init.xavier_normal_(
                self.rcqg_compat_W.weight, gain=nn.init.calculate_gain("relu")
            )
        nn.init.zeros_(self.proj_head_ent.bias)
        nn.init.zeros_(self.proj_tail_ent.bias)
        nn.init.zeros_(self.proj_qual_ent.bias)
        nn.init.zeros_(self.proj_head_rel.bias)
        nn.init.zeros_(self.proj_tail_rel.bias)
        nn.init.zeros_(self.proj_qual_rel.bias)
        nn.init.zeros_(self.proj_head_ent2.bias)
        nn.init.zeros_(self.proj_tail_ent2.bias)
        nn.init.zeros_(self.proj_qual_ent2.bias)
        nn.init.zeros_(self.proj_head_rel2.bias)
        nn.init.zeros_(self.proj_tail_rel2.bias)
        nn.init.zeros_(self.proj_qual_rel2.bias)
        nn.init.zeros_(self.proj_hpair_to_fact.bias)
        nn.init.zeros_(self.proj_tpair_to_fact.bias)
        nn.init.zeros_(self.proj_qpair_to_fact.bias)
        nn.init.zeros_(self.proj_fact_to_head_ent.bias)
        nn.init.zeros_(self.proj_fact_to_tail_ent.bias)
        nn.init.zeros_(self.proj_fact_to_pri_rel.bias)
        nn.init.zeros_(self.proj_fact_to_qual_ent.bias)
        nn.init.zeros_(self.proj_fact_to_qual_rel.bias)

    def _compute_pair_weights(self, hpair_freq, tpair_freq, qpair_freq):
        """IDF scales precede pair LayerNorm; random controls resample per call."""

        def idf_weight(freq):
            if self.idf_weight_form == "linear":
                return 1.0 / (freq + 1.0)
            if self.idf_weight_form == "sqrt":
                return 1.0 / (torch.sqrt(freq) + 1.0)
            return 1.0 / (torch.log(freq + 1.0) + 1.0)

        hpair_freq = hpair_freq.float()
        tpair_freq = tpair_freq.float()
        qpair_freq = qpair_freq.float()
        if self.use_random_pair_weight:
            hpair_w = 0.5 + torch.rand_like(hpair_freq)
            tpair_w = 0.5 + torch.rand_like(tpair_freq)
            qpair_w = 0.5 + torch.rand_like(qpair_freq)
            return (hpair_w, tpair_w, qpair_w)
        if self.use_idf and self.use_idf_hpair:
            hpair_w = idf_weight(hpair_freq)
        else:
            hpair_w = torch.ones_like(hpair_freq)
        if self.use_idf and self.use_idf_tpair:
            tpair_w = idf_weight(tpair_freq)
        else:
            tpair_w = torch.ones_like(tpair_freq)
        if self.use_idf and self.use_idf_qpair:
            qpair_w = idf_weight(qpair_freq)
        else:
            qpair_w = torch.ones_like(qpair_freq)
        return (hpair_w, tpair_w, qpair_w)

    def _get_fact_count_term(self, qual2fact, num_fact):
        fact_count = torch.bincount(qual2fact, minlength=num_fact).float()
        return fact_count

    def _segment_softmax(self, logits, segment_ids, num_segments):
        if len(logits) == 0:
            return logits
        logits = logits.squeeze(dim=1)
        seg_max = torch.full(
            (num_segments,), -float("inf"), dtype=logits.dtype, device=logits.device
        )
        seg_max = seg_max.index_reduce(
            dim=0, index=segment_ids, source=logits, reduce="amax", include_self=True
        )
        stabilized = logits - seg_max[segment_ids]
        exp_logits = torch.exp(stabilized)
        seg_denom = torch.zeros(
            (num_segments,), dtype=exp_logits.dtype, device=exp_logits.device
        )
        seg_denom = seg_denom.index_add(dim=0, index=segment_ids, source=exp_logits)
        return (exp_logits / seg_denom[segment_ids].clamp(min=1e-12)).unsqueeze(dim=1)

    def _compute_rcqg_gate(
        self, emb_rel, emb_rel_qual, rels, qual2fact, qpair_rel, qual2qpair
    ):
        """RCQW uses existing qualifier-role projections and fact-wise softmax."""
        if len(qual2fact) == 0:
            return torch.empty((0, 1), dtype=emb_rel.dtype, device=emb_rel.device)
        main_rel_per_qual = torch.index_select(emb_rel, dim=0, index=rels[qual2fact])
        qual_rel_per_qual = torch.index_select(
            emb_rel_qual,
            dim=0,
            index=torch.index_select(qpair_rel, dim=0, index=qual2qpair),
        )
        if self.rcqg_compat_W is not None:
            main_rel_per_qual = self.rcqg_compat_W(main_rel_per_qual)
        compat = (main_rel_per_qual * qual_rel_per_qual).sum(
            dim=1, keepdim=True
        ) / math.sqrt(float(self.dim))
        softmax_gate = self._segment_softmax(compat, qual2fact, num_segments=len(rels))
        if not self.use_daqg:
            return softmax_gate
        qual_count_per_fact = torch.bincount(qual2fact, minlength=len(rels))
        is_single = (qual_count_per_fact == 1)[qual2fact].unsqueeze(dim=1)
        sigmoid_gate = torch.sigmoid(compat)
        return torch.where(is_single, sigmoid_gate, softmax_gate)

    def forward(
        self,
        emb_ent,
        emb_rel,
        pri,
        qual2fact,
        qual2loc,
        qual,
        hpair_ent,
        hpair_rel,
        hpair_freq,
        fact2hpair,
        tpair_ent,
        tpair_rel,
        tpair_freq,
        fact2tpair,
        qpair_ent,
        qpair_rel,
        qpair_freq,
        qual2qpair,
    ):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()
        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact))
        emb_ent_head = self.proj_head_ent(emb_ent)
        emb_ent_tail = self.proj_tail_ent(emb_ent)
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent)
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda()
        emb_rel_head = self.proj_head_rel(emb_rel)
        emb_rel_tail = self.proj_tail_rel(emb_rel)
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel)
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda()
        hpair_w, tpair_w, qpair_w = self._compute_pair_weights(
            hpair_freq.squeeze(2).float(),
            tpair_freq.squeeze(2).float(),
            qpair_freq.squeeze(2).float(),
        )
        emb_hpair_to_fact = self.to_fact_ln(
            self.proj_hpair_to_fact(
                torch.index_select(emb_rel_head, 0, hpair_rel)
                * torch.index_select(emb_ent_head, 0, hpair_ent)
            )
            * hpair_w
        )
        emb_tpair_to_fact = self.to_fact_ln(
            self.proj_tpair_to_fact(
                torch.index_select(emb_rel_tail, 0, tpair_rel)
                * torch.index_select(emb_ent_tail, 0, tpair_ent)
            )
            * tpair_w
        )
        emb_qpair_to_fact = self.to_fact_ln(
            self.proj_qpair_to_fact(
                torch.index_select(emb_rel_qual, 0, qpair_rel)
                * torch.index_select(emb_ent_qual, 0, qpair_ent)
            )
            * qpair_w
        )
        new_emb_fact = torch.index_select(
            emb_hpair_to_fact, 0, fact2hpair
        ) + torch.index_select(emb_tpair_to_fact, 0, fact2tpair)
        qual_emb = torch.index_select(emb_qpair_to_fact, 0, qual2qpair)
        if len(qual2fact) > 0 and self.use_rcqg:
            qual_gate = self._compute_rcqg_gate(
                emb_rel, emb_rel_qual, rels, qual2fact, qpair_rel, qual2qpair
            )
            qual_emb = qual_emb * qual_gate
        new_emb_fact = new_emb_fact.index_add(dim=0, index=qual2fact, source=qual_emb)
        fact_count_term = self._get_fact_count_term(qual2fact, len(pri))
        new_emb_fact = new_emb_fact / (fact_count_term + 2.0).unsqueeze(dim=1)
        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact))
        emb_fact_to_head = self.to_ent_ln(
            self.proj_fact_to_head_ent(
                torch.index_select(self.proj_head_rel2(emb_rel), 0, rels) * new_emb_fact
            )
        )
        emb_fact_to_tail = self.to_ent_ln(
            self.proj_fact_to_tail_ent(
                torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels) * new_emb_fact
            )
        )
        emb_fact_to_qual = self.to_ent_ln(
            self.proj_fact_to_qual_ent(
                torch.index_select(self.proj_qual_rel2(emb_rel), 0, qual_rels)
                * new_emb_fact[qual2fact]
            )
        )
        new_emb_ent = self.ent_attn(
            query=emb_ent,
            keys=[emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
            values=[emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
            query_len=len(emb_ent),
            self_attn=True,
            query_idxs=[heads, tails, qual_ents],
            pair_weights=None,
        )
        emb_fact_to_pri_rel = self.to_rel_ln(
            self.proj_fact_to_pri_rel(
                (
                    torch.index_select(self.proj_head_ent2(emb_ent), 0, heads)
                    + torch.index_select(self.proj_tail_ent2(emb_ent), 0, tails)
                )
                * new_emb_fact
            )
        )
        emb_fact_to_qual_rel = self.to_rel_ln(
            self.proj_fact_to_qual_rel(
                torch.index_select(self.proj_qual_ent2(emb_ent), 0, qual_ents)
                * torch.index_select(new_emb_fact, 0, qual2fact)
            )
        )
        new_emb_rel = self.rel_attn(
            query=emb_rel,
            keys=[emb_fact_to_pri_rel, emb_fact_to_qual_rel],
            values=[emb_fact_to_pri_rel, emb_fact_to_qual_rel],
            query_len=len(emb_rel),
            self_attn=True,
            query_idxs=[rels, qual_rels],
        )
        return (new_emb_ent, new_emb_rel)

    def pred(
        self,
        num_ent,
        num_rel,
        emb_ent,
        emb_rel,
        pri,
        qual2fact,
        qual2loc,
        qual,
        hpair_ent,
        hpair_rel,
        hpair_freq,
        fact2hpair,
        tpair_ent,
        tpair_rel,
        tpair_freq,
        fact2tpair,
        qpair_ent,
        qpair_rel,
        qpair_freq,
        qual2qpair,
    ):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()
        pred_heads = (heads >= num_ent).nonzero(as_tuple=True)[0]
        pred_tails = (tails >= num_ent).nonzero(as_tuple=True)[0]
        if len(qual) > 0:
            pred_qual_ents = (qual_ents >= num_ent).nonzero(as_tuple=True)[0]
        else:
            qual = qual.long()
            qual2fact = qual2fact.long()
            pred_qual_ents = torch.tensor([]).long().cuda()
        pred_ent = torch.zeros_like(heads)
        pred_ent[pred_heads] = 1
        pred_ent[pred_tails] = 1
        pred_ent[qual2fact[pred_qual_ents]] = 1
        pred_ent = pred_ent.nonzero(as_tuple=True)[0]
        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact))
        emb_ent_head = self.proj_head_ent(emb_ent)
        emb_ent_tail = self.proj_tail_ent(emb_ent)
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent)
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda()
        emb_rel_head = self.proj_head_rel(emb_rel)
        emb_rel_tail = self.proj_tail_rel(emb_rel)
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel)
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda()
        hpair_w, tpair_w, qpair_w = self._compute_pair_weights(
            hpair_freq.squeeze(2).float(),
            tpair_freq.squeeze(2).float(),
            qpair_freq.squeeze(2).float(),
        )
        emb_hpair_to_fact = self.to_fact_ln(
            self.proj_hpair_to_fact(
                torch.index_select(emb_rel_head, 0, hpair_rel)
                * torch.index_select(emb_ent_head, 0, hpair_ent)
            )
            * hpair_w
        )
        emb_tpair_to_fact = self.to_fact_ln(
            self.proj_tpair_to_fact(
                torch.index_select(emb_rel_tail, 0, tpair_rel)
                * torch.index_select(emb_ent_tail, 0, tpair_ent)
            )
            * tpair_w
        )
        emb_qpair_to_fact = self.to_fact_ln(
            self.proj_qpair_to_fact(
                torch.index_select(emb_rel_qual, 0, qpair_rel)
                * torch.index_select(emb_ent_qual, 0, qpair_ent)
            )
            * qpair_w
        )
        new_emb_fact = torch.index_select(
            emb_hpair_to_fact, 0, fact2hpair
        ) + torch.index_select(emb_tpair_to_fact, 0, fact2tpair)
        qual_emb = torch.index_select(emb_qpair_to_fact, 0, qual2qpair)
        if len(qual2fact) > 0 and self.use_rcqg:
            qual_gate = self._compute_rcqg_gate(
                emb_rel, emb_rel_qual, rels, qual2fact, qpair_rel, qual2qpair
            )
            qual_emb = qual_emb * qual_gate
        new_emb_fact = new_emb_fact.index_add(dim=0, index=qual2fact, source=qual_emb)
        fact_count_term = self._get_fact_count_term(qual2fact, len(pri))
        new_emb_fact = new_emb_fact / (fact_count_term + 2.0).unsqueeze(dim=1)
        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact))
        emb_fact_to_head = self.to_ent_ln(
            self.proj_fact_to_head_ent(
                torch.index_select(self.proj_head_rel2(emb_rel), 0, rels[pred_heads])
                * torch.index_select(new_emb_fact, 0, pred_heads)
            )
        )
        emb_fact_to_tail = self.to_ent_ln(
            self.proj_fact_to_tail_ent(
                torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels[pred_tails])
                * torch.index_select(new_emb_fact, 0, pred_tails)
            )
        )
        emb_fact_to_qual = self.to_ent_ln(
            self.proj_fact_to_qual_ent(
                torch.index_select(
                    self.proj_qual_rel2(emb_rel), 0, qual_rels[pred_qual_ents]
                )
                * torch.index_select(new_emb_fact, 0, qual2fact[pred_qual_ents])
            )
        )
        pred_ent_idxs = torch.arange(num_ent, len(emb_ent)).cuda()
        new_emb_ent = self.ent_attn(
            query=emb_ent,
            keys=[emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
            values=[emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
            query_len=len(emb_ent),
            self_attn=False,
            query_idxs=[
                pred_ent_idxs,
                heads[pred_heads],
                tails[pred_tails],
                qual_ents[pred_qual_ents],
            ],
            key_idxs=[pred_ent_idxs, None, None, None],
            value_idxs=[pred_ent_idxs, None, None, None],
            pair_weights=None,
        )
        return new_emb_ent


class MAYPL(nn.Module):

    def __init__(
        self,
        dim,
        num_init_layer,
        num_layer,
        num_head,
        logger,
        model_dropout=0.2,
        use_rcqg=False,
        use_daqg=False,
        use_rcqg_bilinear=False,
        use_random_pair_weight=False,
        use_idf=True,
        use_idf_hpair=True,
        use_idf_tpair=True,
        use_idf_qpair=True,
        idf_weight_form="log",
    ):
        super(MAYPL, self).__init__()
        init_layers = []
        layers = []
        for _ in range(num_init_layer):
            init_layers.append(Init_Layer(dim, logger, dropout=model_dropout))
        for _ in range(num_layer):
            layers.append(
                ANMP_Layer(
                    dim,
                    num_head,
                    logger,
                    dropout=model_dropout,
                    use_rcqg=use_rcqg,
                    use_daqg=use_daqg,
                    use_rcqg_bilinear=use_rcqg_bilinear,
                    use_random_pair_weight=use_random_pair_weight,
                    use_idf=use_idf,
                    use_idf_hpair=use_idf_hpair,
                    use_idf_tpair=use_idf_tpair,
                    use_idf_qpair=use_idf_qpair,
                    idf_weight_form=idf_weight_form,
                )
            )
        self.init_layers = nn.ModuleList(init_layers)
        self.layers = nn.ModuleList(layers)
        self.dim = dim
        self.drop = nn.Dropout(p=model_dropout)
        self.init_emb_ent = nn.Parameter(torch.zeros(1, dim))
        self.init_emb_rel = nn.Parameter(torch.zeros(1, dim))
        self.logger = logger
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_ent, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.init_emb_rel, gain=nn.init.calculate_gain("relu"))

    def forward(
        self,
        pri,
        qual,
        qual2fact,
        qual2loc,
        num_ent,
        num_rel,
        hpair,
        hpair_freq,
        fact2hpair,
        tpair,
        tpair_freq,
        fact2tpair,
        qpair,
        qpair_freq,
        qual2qpair,
    ):
        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]
        hpair_freq = hpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]
        tpair_freq = tpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        if len(qual) > 0:
            qpair_ent = qpair[:, 1]
            qpair_rel = qpair[:, 0]
        else:
            qpair_ent = torch.tensor([]).long().cuda()
            qpair_rel = torch.tensor([]).long().cuda()
        qpair_freq = qpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        emb_ent = self.init_emb_ent.repeat(num_ent, 1)
        emb_rel = self.init_emb_rel.repeat(num_rel, 1)
        init_embs_ent = [emb_ent]
        init_embs_rel = [emb_rel]
        for init_layer in self.init_layers:
            emb_ent, emb_rel = init_layer(emb_ent, emb_rel, pri, qual2fact, qual)
            init_embs_ent.append(emb_ent)
            init_embs_rel.append(emb_rel)
        init_emb_ent = emb_ent
        init_emb_rel = emb_rel
        emb_ents = [init_emb_ent]
        emb_rels = [init_emb_rel]
        for layer in self.layers:
            emb_ent, emb_rel = layer(
                emb_ent,
                emb_rel,
                pri,
                qual2fact,
                qual2loc,
                qual,
                hpair_ent,
                hpair_rel,
                hpair_freq,
                fact2hpair,
                tpair_ent,
                tpair_rel,
                tpair_freq,
                fact2tpair,
                qpair_ent,
                qpair_rel,
                qpair_freq,
                qual2qpair,
            )
            emb_ents.append(emb_ent)
            emb_rels.append(emb_rel)
        return (emb_ents, emb_rels, init_embs_ent, init_embs_rel)

    def pred(
        self,
        pri,
        qual,
        qual2fact,
        qual2loc,
        hpair,
        hpair_freq,
        fact2hpair,
        tpair,
        tpair_freq,
        fact2tpair,
        qpair,
        qpair_freq,
        qual2qpair,
        emb_ents,
        emb_rels,
        init_embs_ent,
        init_embs_rel,
    ):
        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]
        hpair_freq = hpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]
        tpair_freq = tpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        if len(qual) > 0:
            qpair_ent = qpair[:, 1]
            qpair_rel = qpair[:, 0]
        else:
            qpair_ent = torch.tensor([]).long().cuda()
            qpair_rel = torch.tensor([]).long().cuda()
            qual2qpair = qual2qpair.long()
        qpair_freq = qpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)
        num_ent = len(emb_ents[0])
        num_rel = len(emb_rels[0])
        max_ent_cands = [max(pri[:, 0]), num_ent - 1]
        max_ent_cands.append(max(pri[:, 2]))
        if len(qual) > 0:
            max_ent_cands.append(max(qual[:, 1]))
        num_ent_preds = max(max_ent_cands) - num_ent + 1
        emb_ent = torch.cat(
            [init_embs_ent[0], self.init_emb_ent.repeat(num_ent_preds, 1)], dim=0
        )
        for l, init_layer in enumerate(self.init_layers):
            emb_ent, _ = init_layer(emb_ent, init_embs_rel[l], pri, qual2fact, qual)
            emb_ent = torch.cat([init_embs_ent[l + 1], emb_ent[num_ent:]], dim=0)
        input_emb_ent = emb_ent
        input_emb_rel = init_embs_rel[-1]
        for emb_ent, emb_rel, layer in zip(emb_ents[1:], emb_rels[1:], self.layers):
            output_emb_ent = layer.pred(
                num_ent,
                num_rel,
                input_emb_ent,
                input_emb_rel,
                pri,
                qual2fact,
                qual2loc,
                qual,
                hpair_ent,
                hpair_rel,
                hpair_freq,
                fact2hpair,
                tpair_ent,
                tpair_rel,
                tpair_freq,
                fact2tpair,
                qpair_ent,
                qpair_rel,
                qpair_freq,
                qual2qpair,
            )
            input_emb_ent = torch.cat([emb_ent, output_emb_ent[num_ent:]], dim=0)
            input_emb_rel = emb_rel
        final_unseen_emb = input_emb_ent[num_ent:]
        ent_preds = torch.inner(final_unseen_emb, input_emb_ent[:num_ent])
        return ent_preds
