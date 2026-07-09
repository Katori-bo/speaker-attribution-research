import json
import csv
import ast
import time
from pathlib import Path
from collections import defaultdict, Counter
import itertools

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_data_dir, get_reports_dir
from src.experiment.recorder import ExperimentRecorder
from src.baseline.candidate_generator import CandidateGenerator
from src.baseline.attribution_rules import ATTRIBUTION_RULES
from src.baseline.rule_engine import RuleEvaluator, RuleEngine
from src.discourse.discourse_state import MinimalDiscourseState

setup_logging()
logger = get_logger("exp003")

def parse_stringified_list(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def flatten_mentions(mentions):
    flat = []
    if isinstance(mentions, list):
        for item in mentions:
            if isinstance(item, list):
                flat.extend(flatten_mentions(item))
            else:
                flat.append(item)
    return flat

def run_exp003():
    logger.info("Starting EXP003: Symbolic Attribution Rules Evaluation...")
    
    with ExperimentRecorder("EXP003"):
        pdnc_dir = get_data_dir() / "data"
        generator = CandidateGenerator()
        evaluator = RuleEvaluator(ATTRIBUTION_RULES)
        engine = RuleEngine(ATTRIBUTION_RULES)
        state = MinimalDiscourseState()
        
        novels = [d for d in pdnc_dir.iterdir() if d.is_dir()]
        
        total_quotes = 0
        correct_predictions = 0
        oracle_correct = 0
        abstentions = 0
        
        # rule_name -> {applicable: 0, fired: 0, fired_correct: 0, won: 0, won_correct: 0}
        rule_stats = defaultdict(lambda: {"applicable": 0, "fired": 0, "fired_correct": 0, "won": 0, "won_correct": 0})
        
        # tuple(rule_a, rule_b) -> count
        conflicts = Counter()
        
        # quote_type -> {total: 0, correct: 0}
        quote_type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for novel in novels:
            quote_file = novel / "quotation_info.csv"
            if not quote_file.exists(): continue
                
            state.reset_conversation()
            previous_speakers = []
            
            with open(quote_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gold_speaker = row.get("speaker", "Unknown").strip()
                    quote_text = row.get("quoteText", "")
                    quote_type = row.get("quoteType", "Unknown").strip()
                    context_text = row.get("referringExpression", "").strip()
                    
                    mentions_raw = parse_stringified_list(row.get("mentionEntitiesList", "[]"))
                    addressees_raw = parse_stringified_list(row.get("addressees", "[]"))
                    explicit_mentions = flatten_mentions(mentions_raw) + flatten_mentions(addressees_raw)
                    
                    # Generate candidates
                    candidates = set()
                    for m in explicit_mentions:
                        if m: candidates.add(m)
                    for s in set(previous_speakers[-15:]):
                        if s: candidates.add(s)
                        
                    state.update(
                        previous_speakers[-1] if previous_speakers else None, 
                        explicit_mentions, 
                        candidates
                    )
                    
                    # Evaluate Rules Independently
                    evaluations = evaluator.evaluate(quote_text, context_text, state, explicit_mentions)
                    
                    any_correct = False
                    fired_rules = []
                    
                    for rule_name, (applicable, pred) in evaluations.items():
                        if applicable:
                            rule_stats[rule_name]["applicable"] += 1
                        if applicable and pred:
                            fired_rules.append((rule_name, pred))
                            rule_stats[rule_name]["fired"] += 1
                            if pred == gold_speaker:
                                rule_stats[rule_name]["fired_correct"] += 1
                                any_correct = True
                                
                    if any_correct:
                        oracle_correct += 1
                        
                    # Track Conflicts
                    if len(fired_rules) > 1:
                        for (ra, pa), (rb, pb) in itertools.combinations(fired_rules, 2):
                            if pa != pb:
                                pair = tuple(sorted([ra, rb]))
                                conflicts[pair] += 1
                                
                    # Engine Decision
                    final_pred, winning_rule = engine.decide(evaluations)
                    
                    if winning_rule == "No Match":
                        abstentions += 1
                    else:
                        rule_stats[winning_rule]["won"] += 1
                        if final_pred == gold_speaker:
                            rule_stats[winning_rule]["won_correct"] += 1
                            correct_predictions += 1
                            quote_type_stats[quote_type]["correct"] += 1
                            
                    total_quotes += 1
                    quote_type_stats[quote_type]["total"] += 1
                    
                    # Update discourse history
                    if gold_speaker != "Unknown":
                        previous_speakers.append(gold_speaker)
        
        EXP_DIR = get_reports_dir() / "EXP003"
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(EXP_DIR / "rule_statistics.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule", "Applicable", "Fired", "Fired_Correct", "Precision", "Won", "Won_Correct", "Contribution"])
            for rule_name in [r[0] for r in ATTRIBUTION_RULES]:
                st = rule_stats[rule_name]
                precision = st["fired_correct"] / st["fired"] if st["fired"] else 0
                contribution = st["won_correct"] / st["won"] if st["won"] else 0
                writer.writerow([
                    rule_name, st["applicable"], st["fired"], st["fired_correct"],
                    f"{precision*100:.2f}%", st["won"], st["won_correct"], f"{contribution*100:.2f}%"
                ])
                
        with open(EXP_DIR / "rule_conflicts.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule A", "Rule B", "Conflict Count"])
            for pair, count in conflicts.most_common():
                writer.writerow([pair[0], pair[1], count])
                
        overall_accuracy = correct_predictions / total_quotes if total_quotes else 0
        oracle_accuracy = oracle_correct / total_quotes if total_quotes else 0
        abstention_rate = abstentions / total_quotes if total_quotes else 0
        
        with open(EXP_DIR / "symbolic_report.md", 'w') as f:
            f.write("# EXP003: Symbolic Attribution Rules Report\n\n")
            f.write(f"- **Total Quotes:** {total_quotes}\n")
            f.write(f"- **Candidate Generator Ceiling:** 92.76%\n")
            f.write(f"- **Rule Oracle Accuracy:** {oracle_accuracy*100:.2f}%\n")
            f.write(f"- **Engine Final Accuracy:** {overall_accuracy*100:.2f}%\n")
            f.write(f"- **Abstentions (No Rule Fired):** {abstentions} ({abstention_rate*100:.2f}%)\n\n")
            
            f.write("## Quote-Type Error Breakdown\n")
            f.write("| Quote Type | Total | Correct | Accuracy |\n")
            f.write("|------------|-------|---------|----------|\n")
            for qt, stats in sorted(quote_type_stats.items()):
                acc = stats["correct"] / stats["total"] if stats["total"] else 0
                f.write(f"| {qt} | {stats['total']} | {stats['correct']} | {acc*100:.2f}% |\n")
                
            f.write("\n## Rule Quality & Contribution\n")
            f.write("| Rule | Applicable | Fired | Precision (Fired) | Won | Contribution (Won) |\n")
            f.write("|------|------------|-------|-------------------|-----|--------------------|\n")
            for rule_name in [r[0] for r in ATTRIBUTION_RULES]:
                st = rule_stats[rule_name]
                precision = st["fired_correct"] / st["fired"] if st["fired"] else 0
                contribution = st["won_correct"] / st["won"] if st["won"] else 0
                f.write(f"| {rule_name} | {st['applicable']} | {st['fired']} | {precision*100:.2f}% ({st['fired_correct']}/{st['fired']}) | {st['won']} | {contribution*100:.2f}% ({st['won_correct']}/{st['won']}) |\n")
                
            f.write("\n## Major Rule Conflicts\n")
            f.write("| Rule A | Rule B | Count |\n")
            f.write("|--------|--------|-------|\n")
            for pair, count in conflicts.most_common(10):
                f.write(f"| {pair[0]} | {pair[1]} | {count} |\n")
                
        logger.info(f"EXP003 Engine Accuracy: {overall_accuracy*100:.2f}% (Oracle: {oracle_accuracy*100:.2f}%)")
        logger.info("EXP003 completed successfully.")

if __name__ == "__main__":
    run_exp003()
