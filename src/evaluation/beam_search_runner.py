import pandas as pd
import numpy as np
import math
import copy
import argparse
from pathlib import Path

from src.utils.logger import get_logger, setup_logging
from src.utils.config import get_data_dir
from src.discourse.discourse_state import MinimalDiscourseState
from src.discourse.conversation_state import ConversationStateModule
from src.evaluation.dynamic_features import extract_dynamic_features
from src.evaluation.runner import load_frozen_exp014_dataset, train_exp014_model, parse_stringified_list, flatten_mentions

logger = get_logger("BeamSearchRunner")

class BeamHypothesis:
    def __init__(self, state, conv_state, log_prob, predictions, predicted_speakers, is_gold_path):
        self.state = state
        self.conv_state = conv_state
        self.log_prob = log_prob
        self.predictions = predictions
        self.predicted_speakers = predicted_speakers
        self.is_gold_path = is_gold_path
        self.last_prediction_confidence = 0.0

def run_beam_search_evaluation(df, model, feature_names, beam_size=5):
    """
    Runs evaluation on the test set using Beam Search over quotes.
    Re-computes dynamic features live for each hypothesis branch.
    """
    test_df = df[df['split'] == 'test'].copy()
    logger.info(f"Running Beam Search with K={beam_size}")
    
    all_predictions = []
    oracle_logs = []
    oracle_survival_stats = {"total_quotes": 0, "gold_path_survived": 0}
    
    test_novels = df[df['split'] == 'test']['novel'].unique()
    for novel in test_novels:
        logger.info(f"Running Beam Search for {novel}")
        
        initial_state = MinimalDiscourseState()
        initial_conv_state = ConversationStateModule(scene_id=f"{novel}_0")
        
        hypotheses = [BeamHypothesis(
            state=initial_state,
            conv_state=initial_conv_state,
            log_prob=0.0,
            predictions=[],
            predicted_speakers=[],
            is_gold_path=True
        )]
        
        # Load quotation_info for ground-truth mentions and candidates
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        q_info = pd.read_csv(q_info_path)
        
        gold_path_alive = True
        total_quotes_in_novel = len(q_info)
        
        for idx_in_novel, q_row in q_info.iterrows():
            q_id = q_row.get("quote_id")
            if not q_id: q_id = f"{novel}_{idx_in_novel}"
            
            gold_speaker = str(q_row.get("speaker", "Unknown")).strip()
            
            mentions_raw = parse_stringified_list(q_row.get("mentionEntitiesList", "[]"))
            addressees_raw = parse_stringified_list(q_row.get("addressees", "[]"))
            explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
            
            novel_df = df[df['novel'] == novel]
            q_df = novel_df[novel_df['quote_id'] == q_id].copy()
            if not q_df.empty:
                candidates_set = set(q_df['candidate'].unique())
            else:
                candidates_set = set()
                
            quote_spans_raw = parse_stringified_list(q_row.get("quoteByteSpans", "[]"))
            q_start, q_end = -1, -1
            if quote_spans_raw and len(quote_spans_raw) > 0:
                try:
                    q_start = int(quote_spans_raw[0][0])
                    q_end = int(quote_spans_raw[-1][1])
                except:
                    pass
            
            df_dp = q_df.iloc[0]['discourse_dialogue_position'] if not q_df.empty else None
            df_q_start = int(q_df['quote_start_byte'].iloc[0]) if not q_df.empty else -1
            
            new_hypotheses = []
            
            # Loop Order:
            # 1. Existing state -> 2. Extract features -> 3. Predict -> 4. Branch -> 5. Update state -> 6. Prune
            for hyp in hypotheses:
                # Scene reset logic
                if df_dp == 1.0:
                    hyp.conv_state.reset(novel)
                
                # 1. Update state with current quote's mentions and previous quote's predicted speaker
                speaker_for_update = hyp.predicted_speakers[-1] if hyp.predicted_speakers else None
                hyp.state.update(speaker_for_update, explicit_mentions, candidates_set)
                
                # If there are candidates to predict
                if not q_df.empty:
                    # 2. Extract candidate features based on existing state
                    q_df_hyp = q_df.copy()
                    for idx, row in q_df_hyp.iterrows():
                        candidate = row['candidate']
                        dyn_feats = extract_dynamic_features(candidate, hyp.state, hyp.conv_state, df_q_start)
                        for k, v in dyn_feats.items():
                            q_df_hyp.at[idx, k] = v
                    
                    # 3. Predict probabilities
                    X = q_df_hyp[feature_names]
                    scores = model.predict_proba(X)[:, 1]
                    q_df_hyp['score'] = scores
                    
                    # 4. Branch hypotheses
                    for cand_idx, cand_row in q_df_hyp.iterrows():
                        cand_prob = float(cand_row['score'])
                        cand_name = cand_row['candidate']
                        
                        # Use 1e-12 as suggested by user
                        new_log_prob = hyp.log_prob + np.log(cand_prob + 1e-12)
                        
                        # Copy state BEFORE mutating it for the branch
                        new_state = copy.deepcopy(hyp.state)
                        new_conv_state = copy.deepcopy(hyp.conv_state)
                        
                        # 5. Update conv state (after feature extraction)
                        new_conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, cand_name)
                        
                        is_gold_branch = hyp.is_gold_path and (cand_name == gold_speaker)
                        
                        new_hyp = BeamHypothesis(
                            state=new_state,
                            conv_state=new_conv_state,
                            log_prob=new_log_prob,
                            predictions=hyp.predictions + [cand_row],
                            predicted_speakers=hyp.predicted_speakers + [cand_name],
                            is_gold_path=is_gold_branch
                        )
                        new_hyp.last_prediction_confidence = cand_prob
                        new_hypotheses.append(new_hyp)
                else:
                    # Quote was filtered before evaluation.
                    # Advance the oracle state with the gold speaker
                    # so discourse history stays faithful.
                    cand_name = gold_speaker
                    new_state = copy.deepcopy(hyp.state)
                    new_conv_state = copy.deepcopy(hyp.conv_state)
                    new_conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, cand_name)
                    
                    new_hyp = BeamHypothesis(
                        state=new_state,
                        conv_state=new_conv_state,
                        log_prob=hyp.log_prob, # unchanged log prob since no branch
                        predictions=hyp.predictions,
                        predicted_speakers=hyp.predicted_speakers + [cand_name],
                        is_gold_path=hyp.is_gold_path # Do NOT kill oracle
                    )
                    new_hypotheses.append(new_hyp)
            
            # 6. Keep top K
            new_hypotheses.sort(key=lambda x: x.log_prob, reverse=True)
            hypotheses = new_hypotheses[:beam_size]
            
            # Oracle tracking: did the gold path survive in ANY of the beams for this quote?
            gold_survived = any(h.is_gold_path for h in hypotheses)
            
            death_reason = None
            if q_df.empty:
                death_reason = "not_evaluated"
            else:
                if gold_path_alive and not gold_survived:
                    if gold_speaker not in candidates_set:
                        death_reason = "candidate_missing"
                    else:
                        death_reason = "beam_pruned"
                    gold_path_alive = False
                
            oracle_logs.append({
                "novel": novel,
                "quote_id": q_id,
                "idx_in_novel": idx_in_novel,
                "total_quotes_in_novel": total_quotes_in_novel,
                "gold_survived": gold_survived,
                "death_reason": death_reason
            })
            
            if not q_df.empty:
                oracle_survival_stats["total_quotes"] += 1
                if gold_survived:
                    oracle_survival_stats["gold_path_survived"] += 1
        
        # At the end of the novel, take the best hypothesis
        best_hyp = hypotheses[0]
        if best_hyp.predictions:
            all_predictions.append(pd.DataFrame(best_hyp.predictions))
            
    final_df = pd.concat(all_predictions, ignore_index=True)
    oracle_logs_df = pd.DataFrame(oracle_logs)
    return final_df, oracle_survival_stats, oracle_logs_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Beam size")
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Loading base EXP012 dataset...")
    df = load_frozen_exp014_dataset()
    
    logger.info(f"Training frozen model...")
    model, feature_names = train_exp014_model(df)
    
    logger.info(f"Starting Beam Search Evaluation with K={args.k}...")
    pred_df, oracle_stats, oracle_logs_df = run_beam_search_evaluation(df, model, feature_names, beam_size=args.k)

    # Save predictions
    out_dir = Path(f"results/EXP018A/beam_K{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Save oracle stats and logs
    import json
    with open(out_dir / "oracle_survival.json", "w") as f:
        json.dump(oracle_stats, f, indent=2)
        
    oracle_logs_df.to_csv(out_dir / "oracle_logs.csv", index=False)

    logger.info(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
