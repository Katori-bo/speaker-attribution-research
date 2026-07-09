from typing import Dict, Any, Optional

from src.coreference.mapping import MentionToEntityMapper
from src.attribution.extractor import AttributionExtractor
from src.attribution.resolver import AttributionResolver
from src.attribution.features import extract_attribution_features

class AttributionFeatureProvider:
    """
    Integrates the explicit attribution extractors into the ranking pipeline.
    Caches the extraction result per quote to avoid re-parsing the text 
    for every candidate.
    """
    def __init__(self, mapper: MentionToEntityMapper, enabled: bool = True):
        self.enabled = enabled
        self.extractor = AttributionExtractor()
        self.resolver = AttributionResolver(mapper)
        
        # Cache state
        self._last_quote_id: Optional[str] = None
        self._last_resolved_chain_id: Optional[int] = None
        
    def get_features(self, candidate_chain_id: int, quote_id: str, quote_start: int, quote_end: int, content: str) -> Dict[str, float]:
        """
        Extracts attribution features for a given candidate chain ID.
        """
        if not self.enabled:
            return {"candidate_is_attributed_speaker": 0.0}
            
        # Parse quote attribution only once per quote
        if quote_id != self._last_quote_id:
            self._last_quote_id = quote_id
            self._last_resolved_chain_id = None
            
            attribution = self.extractor.extract(quote_id, quote_start, quote_end, content)
            if attribution is not None:
                self._last_resolved_chain_id = self.resolver.resolve(attribution.speaker_mention)
                
        return extract_attribution_features(candidate_chain_id, self._last_resolved_chain_id)
