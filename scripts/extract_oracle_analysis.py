import os
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features.conversation_extractor import ConversationFeatureExtractor
from src.discourse.conversation_state import ConversationStateModule
from src.coreference.pipeline import SemanticFeatureProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_oracle():
    input_file = Path("data/raw/pdnc/phase2/candidate_features.csv")
    df = pd.read_csv(input_file)
    
    # ONLY KEEP TEST SET NOVELS to speed things up immensely, but wait, the model was trained on the train set.
    # We NEED the train set to train the model, otherwise the predictions won't match!
    # Ah, training on 37k quotes takes like 2 seconds for HistGradientBoostingClassifier.
    # The slow part is extracting EXP011 and EXP012 features for the train set.
    # If the train set is slow, we can just load the cached candidate features?
    # Wait, we didn't cache the exp011 and exp012 features! 
    pass

if __name__ == "__main__":
    extract_oracle()
