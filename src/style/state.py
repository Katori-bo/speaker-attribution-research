from src.style.schemas import StyleState, CharacterFingerprint

class CharacterStyleState:
    def __init__(self):
        self.state = StyleState(fingerprints={})
        
    def update(self, speaker: str, quote_text: str):
        if speaker is None:
            return
        speaker = str(speaker).strip()
        if not speaker or speaker.lower() in ("unknown", "none", "nan"):
            return
            
        quote_text = str(quote_text).strip()
        if not quote_text:
            return
            
        tokens = len(quote_text.split())
        
        if speaker not in self.state.fingerprints:
            self.state.fingerprints[speaker] = CharacterFingerprint(
                character_id=speaker,
                quotes_seen=0,
                total_tokens=0,
                texts=[]
            )
            
        fp = self.state.fingerprints[speaker]
        fp.quotes_seen += 1
        fp.total_tokens += tokens
        fp.texts.append(quote_text)
