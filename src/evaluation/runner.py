import os
import ast
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils.config import get_data_dir
from src.coreference.parser import BookNLPParser
from src.coreference.mapping import MentionToEntityMapper
from src.attribution.pipeline import AttributionFeatureProvider
from src.discourse.discourse_state import MinimalDiscourseState
from src.discourse.conversation_state import ConversationStateModule
from src.evaluation.dynamic_features import extract_dynamic_features
from src.evaluation.discourse_mode import TeacherForcedMode, FullyAutoregressiveMode, OneStepAutoregressiveMode
from src.style.pipeline import StyleFeatureProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_stringified_list(val):
    try: return ast.literal_eval(val)
    except: return []

def flatten_mentions(mentions):
    flat = []
    if isinstance(mentions, list):
        for item in mentions:
            if isinstance(item, list): flat.extend(flatten_mentions(item))
            else: flat.append(item)
    return flat

def get_novel_text(novel: str) -> str:
    novel_txt_path = Path(f"data/raw/pdnc/data/{novel}/{novel}.txt")
    if not novel_txt_path.exists():
        txt_files = list(Path(f"data/raw/pdnc/data/{novel}").glob("*.txt"))
        if txt_files:
            novel_txt_path = txt_files[0]
    with open(novel_txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_frozen_exp014_dataset():
    """Reconstructs the exact dataset from EXP014 including coref and attribution features."""
    logger.info("Loading base EXP012 dataset...")
    exp012_cache_file = get_data_dir() / "phase2" / "candidate_features_exp012.csv"
    df = pd.read_csv(exp012_cache_file)
    
    logger.info("Extracting static attribution features...")
    novel_features_list = []
    for novel, novel_df in df.groupby('novel'):
        content = get_novel_text(novel)
        novel_dir = os.path.join("data/raw/pdnc/booknlp_out", novel)
        
        parser = BookNLPParser()
        entities = parser.parse_entities(os.path.join(novel_dir, f"{novel}.entities"))
        aliases = parser.parse_book_aliases(os.path.join(novel_dir, f"{novel}.book"))
        mapper = MentionToEntityMapper(entities, aliases)
        attr_provider = AttributionFeatureProvider(mapper, enabled=True)
        
        logger.info(f"Extracting static attribution features for {novel}...")
        unique_quotes = sorted(novel_df['quote_id'].unique(), key=lambda x: int(x.split('_')[-1]))
        
        for q_id in unique_quotes:
            q_df = novel_df[novel_df['quote_id'] == q_id]
            q_start = int(q_df['quote_start_byte'].iloc[0])
            q_end = int(q_df['quote_end_byte'].iloc[0])
            
            for _, row in q_df.iterrows():
                candidate = row['candidate']
                candidate_chain_id = mapper.resolve_string_to_chain_id(candidate)
                if candidate_chain_id is None: candidate_chain_id = -1
                
                attr_feats = attr_provider.get_features(
                    candidate_chain_id=int(candidate_chain_id),
                    quote_id=q_id,
                    quote_start=q_start,
                    quote_end=q_end,
                    content=content
                )
                attr_feats['quote_id'] = q_id
                attr_feats['candidate'] = candidate
                novel_features_list.append(attr_feats)
                
    attr_feat_df = pd.DataFrame(novel_features_list)
    df = df.merge(attr_feat_df, on=['quote_id', 'candidate'], how='left')
    return df

def train_exp014_model(df):
    """Trains the HistGBM on the frozen training set exactly as EXP014 did."""
    train_df = df[df['split'] == 'train'].copy()
    
    # We must construct exp_feats identically to run_exp014.py
    base_feats = [c for c in df.columns if c not in [
        "quote_id", "novel", "candidate", "gold_speaker", "split", "label",
        "quote_start_byte", "quote_end_byte", "quoteByteSpans",
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ] and not c.startswith("symbolic_")]
    
    exp_feats = base_feats + [
        "candidate_in_quote_chain", "nearest_coref_dist", "recent_mention_count", "chain_recency",
        "candidate_is_attributed_speaker"
    ]
    
    # Sort them to guarantee stable order
    exp_feats = sorted(exp_feats)
    
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(train_df[exp_feats], train_df['label'])
    
    return model, exp_feats

def run_evaluation(mode, df, model, feature_names, use_soft_state=False, integrity_stats=None, style_update_mode="real", min_quotes=5, control_mode=None, force_style_zero=False):
    """
    Runs evaluation on the test set using the specified discourse tracking mode.
    Re-computes dynamic features live while keeping static features frozen.
    
    Args:
        mode: DiscourseMode instance controlling state updates
        df: Full dataset (train+test) with static features
        model: Frozen HistGBM classifier
        feature_names: Ordered list of feature columns
        use_soft_state: If True, replaces binary candidate_is_last_speaker with
                        the model's probability from the previous turn (EXP017B)
    """
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"Running evaluation mode: {mode.name}")
    predictions = []
    
    for novel, novel_df in test_df.groupby('novel'):
        state = MinimalDiscourseState()
        conv_state = ConversationStateModule(scene_id=f"{novel}_0")
        
        # Load quotation_info to extract ground-truth mentions and candidates
        q_info_path = get_data_dir() / "data" / novel / "quotation_info.csv"
        q_info = pd.read_csv(q_info_path)
        
        style_provider = None
        if 'character_lexical_similarity' in feature_names:
            style_provider = StyleFeatureProvider(min_quotes=min_quotes, control_mode=control_mode, q_info=q_info)
        
        # For tracking state
        gold_previous_speakers = []
        predicted_previous_speakers = []
        last_prediction_confidence = 0.0
        
        for idx_in_novel, q_row in q_info.iterrows():
            q_id = q_row.get("quote_id")
            if not q_id: q_id = f"{novel}_{idx_in_novel}"
            
            gold_speaker = str(q_row.get("speaker", "Unknown")).strip()
            
            mentions_raw = parse_stringified_list(q_row.get("mentionEntitiesList", "[]"))
            addressees_raw = parse_stringified_list(q_row.get("addressees", "[]"))
            explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
            
            # Determine candidates_set from the frozen df if available
            q_df = novel_df[novel_df['quote_id'] == q_id].copy()
            if not q_df.empty:
                candidates_set = set(q_df['candidate'].unique())
            else:
                candidates_set = set()
            
            if gold_previous_speakers:
                gold_prev = gold_previous_speakers[-1]
            else:
                gold_prev = None
                
            speaker_for_update = mode.resolve_speaker(
                gold_speaker=gold_prev, 
                predicted_speaker=predicted_previous_speakers[-1] if predicted_previous_speakers else None,
                confidence=last_prediction_confidence
            )
            
            # Parse quote bytes for ConversationStateModule
            quote_spans_raw = parse_stringified_list(q_row.get("quoteByteSpans", "[]"))
            q_start = -1
            q_end = -1
            if quote_spans_raw and len(quote_spans_raw) > 0:
                try:
                    q_start = int(quote_spans_raw[0][0])
                    q_end = int(quote_spans_raw[-1][1])
                except:
                    pass
            
            # Evaluate ConversationStateModule reset (simulating run_exp011a.py)
            if not q_df.empty:
                df_dp = q_df.iloc[0]['discourse_dialogue_position']
                if df_dp == 1.0:
                    conv_state.reset(novel)
                    
            if mode.name == "one_step_autoregressive":
                state.update(gold_prev, explicit_mentions, candidates_set)
                if predicted_previous_speakers:
                    state.last_speaker = predicted_previous_speakers[-1]
            elif mode.name == "reverse_one_step_autoregressive":
                # corrupt complete state
                state.update(predicted_previous_speakers[-1] if predicted_previous_speakers else None, explicit_mentions, candidates_set)
                # restore only target variable
                state.last_speaker = gold_prev
            else:
                state.update(speaker_for_update, explicit_mentions, candidates_set)
            
            # If this quote is in our frozen evaluation set, we evaluate it!
            if not q_df.empty:
                df_q_start = int(q_df['quote_start_byte'].iloc[0])
                
                style_scores = None
                if style_provider:
                    if force_style_zero:
                        style_scores = {c: 0.0 for c in candidates_set}
                    else:
                        quote_text = str(q_row.get("quoteText", ""))
                        q_type = str(q_row.get("quoteType", ""))
                        style_scores = style_provider.extract_features(
                            quote_text,
                            list(candidates_set),
                            quote_id=q_id,
                            quote_type=q_type,
                            gold_speaker=gold_speaker
                        )
                
                # Recompute dynamic features for all candidates for this quote
                for idx, row in q_df.iterrows():
                    candidate = row['candidate']
                    dyn_feats = extract_dynamic_features(candidate, state, conv_state, df_q_start, style_scores)
                    
                    # EXP017B: Replace binary last_speaker with soft probability
                    if use_soft_state and state.last_speaker_probs:
                        dyn_feats['candidate_is_last_speaker'] = state.last_speaker_probs.get(candidate, 0.0)
                    
                    for k, v in dyn_feats.items():
                        q_df.at[idx, k] = v
                
                # Predict
                X = q_df[feature_names]
                scores = model.predict_proba(X)[:, 1]
                q_df['score'] = scores
                
                # Pick highest scoring candidate
                best_idx = scores.argmax()
                predicted_speaker = q_df.iloc[best_idx]['candidate']
                last_prediction_confidence = float(scores[best_idx])
                
                if mode.name == "explicit_anchor_reset":
                    anchor_candidate = None
                    for ci, crow in q_df.iterrows():
                        if crow.get('candidate_is_attributed_speaker', 0) == 1:
                            anchor_candidate = crow['candidate']
                            break
                    
                    q_df['raw_prediction'] = predicted_speaker
                    q_df['is_anchor_fired'] = 1 if anchor_candidate is not None else 0
                    
                    # Log anchor analysis stats on all rows of this quote
                    q_df['persisted_last_speaker'] = state.last_speaker
                    q_df['gold_prev_speaker'] = gold_prev
                    q_df['state_drifted'] = state.last_speaker != gold_prev
                    q_df['anchor_attributed_speaker'] = anchor_candidate
                    
                    if anchor_candidate:
                        # state_reset_applied: Does anchor override change what would have been written?
                        # It changes it if raw_prediction != anchor_candidate.
                        # Wait, the prompt says "whether anchor override changed what gets written to state.last_speaker for quote N+1"
                        # That means raw_prediction != anchor_candidate.
                        q_df['state_reset_applied'] = (predicted_speaker != anchor_candidate)
                        predicted_speaker = anchor_candidate
                    else:
                        q_df['state_reset_applied'] = False
                
                # Store soft probabilities in state for EXP017B
                if use_soft_state:
                    prob_dict = {}
                    for ci, crow in q_df.iterrows():
                        prob_dict[crow['candidate']] = float(q_df.at[ci, 'score'])
                    state.last_speaker_probs = prob_dict
                
                # Update Style State (Real Mode)
                if style_provider and style_update_mode == "real":
                    quote_text = str(q_row.get("quoteText", ""))
                    predicted_row = q_df[q_df['candidate'] == predicted_speaker]
                    if not predicted_row.empty:
                        is_attributed = predicted_row.iloc[0].get('candidate_is_attributed_speaker', 0) == 1
                        if is_attributed:
                            style_provider.update_state(predicted_speaker, quote_text)
                            
                predictions.append(q_df)
            else:
                predicted_speaker = "Unknown"
                last_prediction_confidence = 0.0
                
            # Update Style State (Diagnostic Mode, updates even for filtered quotes)
            if style_provider and style_update_mode == "gold":
                quote_text = str(q_row.get("quoteText", ""))
                if q_row.get("quoteType") == "Explicit" and gold_speaker != "Unknown":
                    style_provider.update_state(gold_speaker, quote_text)
                
            # ConversationStateModule update (AFTER feature extraction, matching run_exp011a.py)
            if mode.name == "one_step_autoregressive":
                conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, gold_speaker)
            elif mode.name == "reverse_one_step_autoregressive":
                conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, predicted_speaker)
            else:
                speaker_for_conv_update = mode.resolve_speaker(
                    gold_speaker=gold_speaker,
                    predicted_speaker=predicted_speaker,
                    confidence=last_prediction_confidence
                )
                conv_state.update({"quote_start_byte": q_start, "quote_end_byte": q_end}, speaker_for_conv_update)
            
            if mode.name == "reverse_one_step_autoregressive" and integrity_stats is not None:
                if gold_prev is not None:
                    if state.last_speaker == gold_prev:
                        integrity_stats["last_speaker_gold_match"] += 1
                    integrity_stats["last_speaker_total"] += 1
                    
                    # check history (previous speaker). Note state.previous_speaker is the one from 2 turns ago 
                    # due to update. But conv_state top should be predicted_speaker
                    if len(conv_state.participant_stack) > 0 and conv_state.participant_stack[-1] == predicted_speaker:
                        integrity_stats["history_predicted_match"] += 1
                    integrity_stats["history_total"] += 1

            if gold_speaker != "Unknown":
                gold_previous_speakers.append(gold_speaker)
                predicted_previous_speakers.append(predicted_speaker)
                
    return pd.concat(predictions)

def main():
    integrity_stats = {
        "last_speaker_gold_match": 0,
        "last_speaker_total": 0,
        "history_predicted_match": 0,
        "history_total": 0
    }
    
    df = load_frozen_exp014_dataset()
    model, feature_names = train_exp014_model(df)
    
    # Save the original EXP014 test evaluations for bit-for-bit verification
    test_df_original = df[df['split'] == 'test'].copy()
    test_df_original['score'] = model.predict_proba(test_df_original[feature_names])[:, 1]
    
    # 1. Teacher Forced Verification
    # tf_mode = TeacherForcedMode()
    # tf_preds = run_evaluation(tf_mode, df, model, feature_names)
    
    # Sort both identically for comparison
    test_df_original = test_df_original.sort_values(by=['quote_id', 'candidate']).reset_index(drop=True)
    # tf_preds = tf_preds.sort_values(by=['quote_id', 'candidate']).reset_index(drop=True)
    
    # Bit-for-Bit Verification Tests
    logger.info("Running Bit-for-Bit Verification Tests...")
    
    test_df_original['rank'] = test_df_original.groupby('quote_id')['score'].rank(ascending=False, method='first')
    acc_orig = test_df_original[test_df_original['rank'] == 1]['label'].mean()
    
    # tf_preds['rank'] = tf_preds.groupby('quote_id')['score'].rank(ascending=False, method='first')
    # acc_tf = tf_preds[tf_preds['rank'] == 1]['label'].mean()
    
    logger.info(f"EXP014 Original Accuracy: {acc_orig:.4f}")
    # logger.info(f"Teacher Forced Accuracy:  {acc_tf:.4f}")
    # if not np.isclose(acc_orig, acc_tf):
    #     logger.warning(f"Verification Level 1 Mismatch: {acc_orig:.4f} vs {acc_tf:.4f}")
    
    logger.info("Verification Complete.")
    
    # Save TF predictions
    # os.makedirs("results/EXP016/teacher_forced", exist_ok=True)
    # tf_preds.to_csv("results/EXP016/teacher_forced/predictions.csv", index=False)
    
    # 2. One Step Autoregressive
    # logger.info("Running One-Step Autoregressive Mode...")
    # os_mode = OneStepAutoregressiveMode()
    # os_preds = run_evaluation(os_mode, df, model, feature_names)
    # os.makedirs("results/EXP016/one_step_autoregressive", exist_ok=True)
    # os_preds.to_csv("results/EXP016/one_step_autoregressive/predictions.csv", index=False)
    
    # 3. Fully Autoregressive
    # logger.info("Running Fully Autoregressive Mode...")
    # fa_mode = FullyAutoregressiveMode()
    # fa_preds = run_evaluation(fa_mode, df, model, feature_names)
    # os.makedirs("results/EXP016/fully_autoregressive", exist_ok=True)
    # fa_preds.to_csv("results/EXP016/fully_autoregressive/predictions.csv", index=False)
    
    # 4. EXP017A: Confidence-Gated Sweep
    # from src.evaluation.discourse_mode import ConfidenceGatedMode
    # thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    # for t in thresholds:
    #     logger.info(f"Running Confidence-Gated Mode (threshold={t:.2f})...")
    #     cg_mode = ConfidenceGatedMode(threshold=t)
    #     cg_preds = run_evaluation(cg_mode, df, model, feature_names)
    #     out_dir = f"results/EXP017/confidence_gated_{t:.2f}"
    #     os.makedirs(out_dir, exist_ok=True)
    #     cg_preds.to_csv(f"{out_dir}/predictions.csv", index=False)
    
    # 5. EXP017B: Soft State (Top-K)
    # logger.info("Running Soft State (Top-K) Mode...")
    # fa_soft_mode = FullyAutoregressiveMode()
    # fa_soft_mode.name = "soft_state_autoregressive"
    # fa_soft_preds = run_evaluation(fa_soft_mode, df, model, feature_names, use_soft_state=True)
    # os.makedirs("results/EXP017/soft_state", exist_ok=True)
    # fa_soft_preds.to_csv("results/EXP017/soft_state/predictions.csv", index=False)
    
    # 6. EXP016C: Reverse One-Step Autoregressive
    # logger.info("Running Reverse One-Step Autoregressive Mode...")
    # from src.evaluation.discourse_mode import ReverseOneStepAutoregressiveMode
    # ros_mode = ReverseOneStepAutoregressiveMode()
    # ros_preds = run_evaluation(ros_mode, df, model, feature_names, integrity_stats=integrity_stats)
    # os.makedirs("results/EXP016C", exist_ok=True)
    # ros_preds.to_csv("results/EXP016C/predictions.csv", index=False)
    
    # with open("results/EXP016C/state_integrity_check.json", "w") as f:
    #     json.dump(integrity_stats, f, indent=2)
        
    # 7. EXP017C: Explicit Anchor Reset
    logger.info("Running Explicit Anchor Reset Mode...")
    from src.evaluation.discourse_mode import ExplicitAnchorResetMode
    ear_mode = ExplicitAnchorResetMode()
    ear_preds = run_evaluation(ear_mode, df, model, feature_names)
    os.makedirs("results/EXP017C", exist_ok=True)
    ear_preds.to_csv("results/EXP017C/predictions.csv", index=False)
    
    logger.info("All inference runs completed successfully!")

if __name__ == "__main__":
    main()

