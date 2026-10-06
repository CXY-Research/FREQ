from dataloader import HKG
from tqdm import tqdm
from utils import calculate_rank, metrics, load_model_state, parse_idf_component_flags
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import os
import random
import logging
from model import MAYPL

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.cuda.empty_cache()


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(args, logger):
    for arg_name in vars(args).keys():
        logger.info(f"{arg_name}:{vars(args)[arg_name]}")
    logger.info("Args Listed!")
    use_idf_hpair, use_idf_tpair, use_idf_qpair, idf_component_desc = (
        parse_idf_component_flags(args.idf_components, args.disable_idf)
    )
    logger.info(f"IDF components enabled: {idf_component_desc}")
    logger.info(f"IDF weight form: {args.idf_weight_form}")
    logger.info(f"Evaluation pair-frequency source: {args.eval_pair_freq_source}")
    if args.use_rcqg:
        logger.info(
            "RCQG mode: relation-conditioned qualifier gating is enabled on forward/pred qualifier aggregation."
        )
    if args.use_daqg:
        logger.info(
            "DAQG mode: single-qualifier facts use sigmoid gate, multi-qualifier facts keep softmax gate."
        )
    if args.use_rcqg_bilinear:
        logger.info(
            "RCQG bilinear mode: compatibility uses learnable bilinear transform W before dot-product."
        )
    if args.use_daqg and (not args.use_rcqg):
        raise ValueError("use_daqg requires use_rcqg to be enabled.")
    if args.use_rcqg_bilinear and (not args.use_rcqg):
        raise ValueError("use_rcqg_bilinear requires use_rcqg to be enabled.")
    KG = HKG(
        args.data_dir,
        args.dataset_name,
        logger,
        setting=args.setting,
        msg_add_tr=args.msg_add_tr,
        eval_pair_freq_source=args.eval_pair_freq_source,
    )
    default_answer = []
    if args.msg_add_tr:
        orig_KG = HKG(args.data_dir, args.dataset_name, logger, setting=args.setting)
        for ent in orig_KG.ent2id_train:
            default_answer.append(KG.ent2id_inf[ent])
    model = MAYPL(
        dim=args.dim,
        num_head=args.num_head,
        num_init_layer=args.num_init_layer,
        num_layer=args.num_layer,
        logger=logger,
        model_dropout=args.model_dropout,
        use_rcqg=args.use_rcqg,
        use_daqg=args.use_daqg,
        use_rcqg_bilinear=args.use_rcqg_bilinear,
        use_random_pair_weight=args.use_random_pair_weight,
        use_idf=not args.disable_idf,
        use_idf_hpair=use_idf_hpair,
        use_idf_tpair=use_idf_tpair,
        use_idf_qpair=use_idf_qpair,
        idf_weight_form=args.idf_weight_form,
    ).cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing, reduction="sum")
    optim_params = list(model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)
    if args.start_epoch != 0:
        ckpt_path = f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.start_epoch}.ckpt"
        ckpt = torch.load(ckpt_path)
        legacy = load_model_state(model, ckpt["model_state_dict"])
        if legacy:
            logger.warning(
                "Legacy inactive parameters removed; optimizer starts fresh."
            )
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_valid_mrr = 0
    best_valid_epoch = 0
    best_ckpt_path = None
    last_ckpt_path = None
    for epoch in range(args.start_epoch, args.num_epoch):
        epoch_total_loss = 0.0
        train_graph_idxs = (torch.rand(KG.num_train) < args.train_graph_ratio).nonzero(
            as_tuple=True
        )[0]
        (
            base_pri,
            base_qual,
            base_qual2fact,
            base_qual2loc,
            num_base_ents,
            num_base_rels,
            base_hpair,
            base_hpair_freq,
            base_fact2hpair,
            base_tpair,
            base_tpair_freq,
            base_fact2tpair,
            base_qpair,
            base_qpair_freq,
            base_qual2qpair,
            conv_ent,
            conv_rel,
            query_idxs,
        ) = KG.train_split(train_graph_idxs)
        num_batch = args.batch_num
        base_pri_input = base_pri.clone().detach()
        base_qual_input = base_qual.clone().detach()
        base_hpair_input = base_hpair.clone().detach()
        base_tpair_input = base_tpair.clone().detach()
        base_qpair_input = base_qpair.clone().detach()
        for rand_idxs in tqdm(
            torch.tensor_split(torch.randperm(len(query_idxs)), args.batch_num)
        ):
            batch = query_idxs[rand_idxs]
            (
                query_pri,
                query_qual,
                query_qual2fact,
                query_qual2loc,
                query_hpair,
                query_hpair_freq,
                query_fact2hpair,
                query_tpair,
                query_tpair_freq,
                query_fact2tpair,
                query_qpair,
                query_qpair_freq,
                query_qual2qpair,
                answers,
            ) = KG.train_preds(conv_ent, conv_rel, batch)
            optimizer.zero_grad()
            emb_ents, emb_rels, init_embs_ent, init_embs_rel = model(
                base_pri_input,
                base_qual_input,
                base_qual2fact,
                base_qual2loc,
                num_base_ents,
                num_base_rels,
                base_hpair_input,
                base_hpair_freq,
                base_fact2hpair,
                base_tpair_input,
                base_tpair_freq,
                base_fact2tpair,
                base_qpair_input,
                base_qpair_freq,
                base_qual2qpair,
            )
            predictions = model.pred(
                query_pri,
                query_qual,
                query_qual2fact,
                query_qual2loc,
                query_hpair,
                query_hpair_freq,
                query_fact2hpair,
                query_tpair,
                query_tpair_freq,
                query_fact2tpair,
                query_qpair,
                query_qpair_freq,
                query_qual2qpair,
                emb_ents,
                emb_rels,
                init_embs_ent,
                init_embs_rel,
            )
            lp_loss = criterion(predictions, answers) / len(batch)
            total_loss = lp_loss
            total_loss.backward()
            optimizer.step()
            epoch_total_loss += total_loss.item()
        logger.info(
            f"Epoch {epoch + 1} GPU:{torch.cuda.max_memory_allocated()} Loss:{epoch_total_loss / num_batch:.6f}"
        )
        if (epoch + 1) % args.val_dur == 0:
            model.eval()
            lp_head_list_rank = []
            lp_tail_list_rank = []
            lp_pri_list_rank = []
            lp_qual_list_rank = []
            lp_all_list_rank = []
            with torch.no_grad():
                emb_ents, emb_rels, init_embs_ent, init_embs_rel = model(
                    KG.pri_inf.clone().detach(),
                    KG.qual_inf.clone().detach(),
                    KG.qual2fact_inf,
                    KG.qual2loc_inf,
                    KG.num_ent_inf,
                    KG.num_rel_inf,
                    KG.hpair_inf.clone().detach(),
                    KG.hpair_freq_inf,
                    KG.fact2hpair_inf,
                    KG.tpair_inf.clone().detach(),
                    KG.tpair_freq_inf,
                    KG.fact2tpair_inf,
                    KG.qpair_inf.clone().detach(),
                    KG.qpair_freq_inf,
                    KG.qual2qpair_inf,
                )
                for idxs in tqdm(
                    torch.split(torch.arange(len(KG.valid_query)), args.val_size)
                ):
                    (
                        query_pri,
                        query_qual,
                        query_qual2fact,
                        query_qual2loc,
                        query_hpair,
                        query_hpair_freq,
                        query_fact2hpair,
                        query_tpair,
                        query_tpair_freq,
                        query_fact2tpair,
                        query_qpair,
                        query_qpair_freq,
                        query_qual2qpair,
                        answers,
                        pred_locs,
                    ) = KG.valid_inputs(idxs)
                    preds = model.pred(
                        query_pri,
                        query_qual,
                        query_qual2fact,
                        query_qual2loc,
                        query_hpair,
                        query_hpair_freq,
                        query_fact2hpair,
                        query_tpair,
                        query_tpair_freq,
                        query_fact2tpair,
                        query_qpair,
                        query_qpair_freq,
                        query_qual2qpair,
                        emb_ents,
                        emb_rels,
                        init_embs_ent,
                        init_embs_rel,
                    )
                    for i, idx in enumerate(idxs):
                        pred_loc = pred_locs[i]
                        answer = answers[i] + default_answer
                        for valid_answer in KG.valid_answer[idx]:
                            rank = calculate_rank(
                                preds.detach().cpu().numpy()[i], valid_answer, answer
                            )
                            if pred_loc <= 2:
                                lp_pri_list_rank.append(rank)
                            if pred_loc == 0:
                                lp_head_list_rank.append(rank)
                            elif pred_loc == 2:
                                lp_tail_list_rank.append(rank)
                            else:
                                lp_qual_list_rank.append(rank)
                            lp_all_list_rank.append(rank)
                head_mr, head_mrr, head_hit10, head_hit3, head_hit1 = metrics(
                    np.array(lp_head_list_rank)
                )
                tail_mr, tail_mrr, tail_hit10, tail_hit3, tail_hit1 = metrics(
                    np.array(lp_tail_list_rank)
                )
                if len(lp_qual_list_rank) > 0:
                    (
                        qual_ent_mr,
                        qual_ent_mrr,
                        qual_ent_hit10,
                        qual_ent_hit3,
                        qual_ent_hit1,
                    ) = metrics(np.array(lp_qual_list_rank))
                pri_ent_mr, pri_ent_mrr, pri_ent_hit10, pri_ent_hit3, pri_ent_hit1 = (
                    metrics(np.array(lp_pri_list_rank))
                )
                all_ent_mr, all_ent_mrr, all_ent_hit10, all_ent_hit3, all_ent_hit1 = (
                    metrics(np.array(lp_all_list_rank))
                )
                logger.info(
                    f"Link Prediction (Head, {len(lp_head_list_rank)})\nMR:{head_mr}\nMRR:{head_mrr}\nHit1:{head_hit1}\nHit3:{head_hit3}\nHit10:{head_hit10}"
                )
                logger.info(
                    f"Link Prediction (Tail, {len(lp_tail_list_rank)})\nMR:{tail_mr}\nMRR:{tail_mrr}\nHit1:{tail_hit1}\nHit3:{tail_hit3}\nHit10:{tail_hit10}"
                )
                if len(lp_qual_list_rank) > 0:
                    logger.info(
                        f"Link Prediction (Qual, {len(lp_qual_list_rank)})\nMR:{qual_ent_mr}\nMRR:{qual_ent_mrr}\nHit1:{qual_ent_hit1}\nHit3:{qual_ent_hit3}\nHit10:{qual_ent_hit10}"
                    )
                logger.info(
                    f"Link Prediction (Pri, {len(lp_pri_list_rank)})\nMR:{pri_ent_mr}\nMRR:{pri_ent_mrr}\nHit1:{pri_ent_hit1}\nHit3:{pri_ent_hit3}\nHit10:{pri_ent_hit10}"
                )
                if len(lp_qual_list_rank) > 0:
                    logger.info(
                        f"Link Prediction (All, {len(lp_all_list_rank)})\nMR:{all_ent_mr}\nMRR:{all_ent_mrr}\nHit1:{all_ent_hit1}\nHit3:{all_ent_hit3}\nHit10:{all_ent_hit10}"
                    )
                if not args.no_write:
                    ckpt_obj = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    cur_ckpt_path = f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{epoch + 1}.ckpt"
                    torch.save(ckpt_obj, cur_ckpt_path)
                    if (
                        last_ckpt_path is not None
                        and last_ckpt_path != best_ckpt_path
                        and os.path.exists(last_ckpt_path)
                    ):
                        os.remove(last_ckpt_path)
                    last_ckpt_path = cur_ckpt_path
                if all_ent_mrr > best_valid_mrr:
                    if (
                        best_ckpt_path is not None
                        and best_ckpt_path != last_ckpt_path
                        and os.path.exists(best_ckpt_path)
                    ):
                        os.remove(best_ckpt_path)
                    best_valid_mrr = all_ent_mrr
                    best_valid_epoch = epoch
                    if not args.no_write:
                        best_ckpt_path = last_ckpt_path
                        logger.info(
                            f"New best valid MRR {best_valid_mrr:.4f} at epoch {epoch + 1}; kept {best_ckpt_path}"
                        )
                elif args.early_stop > 0:
                    if epoch - best_valid_epoch >= args.early_stop:
                        break
            model.train()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="WikiPeople--eval", type=str)
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--exp", default="ICML2025", type=str)
    parser.add_argument("--setting", default="Transductive", type=str)
    parser.add_argument("--log_name", default=None, type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_epoch", default=2900, type=int)
    parser.add_argument("--val_dur", default=2900, type=int)
    parser.add_argument("--val_size", default=100, type=int)
    parser.add_argument("--num_init_layer", default=3, type=int)
    parser.add_argument("--num_layer", default=4, type=int)
    parser.add_argument("--num_head", default=32, type=int)
    parser.add_argument("--early_stop", default=0, type=int)
    parser.add_argument("--batch_num", default=20, type=int)
    parser.add_argument("--train_graph_ratio", default=0.7, type=float)
    parser.add_argument("--model_dropout", default=0.1, type=float)
    parser.add_argument("--smoothing", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--msg_add_tr", action="store_true")
    parser.add_argument(
        "--use_rcqg",
        action="store_true",
        help="Enable relation-conditioned qualifier gating (RCQG) before qualifier-to-fact aggregation.",
    )
    parser.add_argument(
        "--use_daqg",
        action="store_true",
        help="Enable density-adaptive qualifier gating: sigmoid for single-qualifier facts, softmax for multi-qualifier facts (requires --use_rcqg).",
    )
    parser.add_argument(
        "--use_rcqg_bilinear",
        action="store_true",
        help="Enable bilinear RCQG compatibility score with learnable W (requires --use_rcqg).",
    )
    parser.add_argument(
        "--use_random_pair_weight",
        action="store_true",
        help="Reviewer-1 baseline: apply random weights (uniform in [0.5, 1.5), resampled each aggregation call from the run-seeded RNG) to h/t/q pairs instead of any frequency-based weighting.",
    )
    parser.add_argument("--disable_idf", action="store_true")
    parser.add_argument(
        "--idf_components",
        default="htq",
        type=str,
        help="Subset of {h,t,q} to apply IDF on hpair/tpair/qpair routes. Ignored when --disable_idf is set.",
    )
    parser.add_argument(
        "--idf_weight_form",
        default="log",
        choices=["log", "linear", "sqrt"],
        type=str,
        help="IDF weighting form: 'log' (1/(log(f+1)+1)), 'linear' (1/(f+1)), or 'sqrt' (1/(sqrt(f)+1)).",
    )
    parser.add_argument(
        "--eval_pair_freq_source",
        default="inference",
        choices=["inference", "train"],
        type=str,
        help="Pair-frequency source for valid/test query construction: 'inference' (default) or 'train'.",
    )
    args = parser.parse_args()
    set_random_seed(args.seed)
    if args.log_name is None:
        file_format = datetime.datetime.now()
    else:
        file_format = args.log_name
    if not args.no_write:
        os.makedirs(f"./ckpt/{args.exp}/{args.dataset_name}", exist_ok=True)
        os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok=True)
    else:
        file_format = None
    if not args.no_write:
        file_handler = logging.FileHandler(
            f"./logs/{args.exp}/{args.dataset_name}/{file_format}.log"
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    logger.info(f"{os.getpid()}")
    try:
        train(args, logger)
    except Exception as e:
        logging.critical(e, exc_info=True)
        raise
    logger.info("END")
