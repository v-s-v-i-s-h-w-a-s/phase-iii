"""
Basketball Play Parser - Natural Language Processing Module
Converts natural language play descriptions into structured play data
"""

import re
import json
import spacy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Basketball action types"""
    PASS = "pass"
    DRIBBLE = "dribble"
    SCREEN = "screen"
    CUT = "cut"
    SHOT = "shot"
    REBOUND = "rebound"
    STEAL = "steal"
    BLOCK = "block"
    MOVEMENT = "movement"
    FORMATION = "formation"

class PlayerPosition(Enum):
    """Basketball positions"""
    PG = "point_guard"
    SG = "shooting_guard" 
    SF = "small_forward"
    PF = "power_forward"
    C = "center"
    G = "guard"
    F = "forward"

@dataclass
class Player:
    """Player representation"""
    id: str
    position: PlayerPosition
    team: str
    location: Tuple[float, float] = (0.0, 0.0)
    
@dataclass 
class Action:
    """Basketball action representation"""
    type: ActionType
    player: str
    target_player: Optional[str] = None
    location: Optional[Tuple[float, float]] = None
    timestamp: float = 0.0
    duration: float = 1.0
    description: str = ""
    confidence: float = 1.0

@dataclass
class ParsedPlay:
    """Complete parsed play structure"""
    name: str
    description: str
    actions: List[Action]
    players: List[Player]
    formation: str
    duration: float
    confidence: float
    alternative_interpretations: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'actions': [asdict(action) for action in self.actions],
            'players': [asdict(player) for player in self.players],
            'formation': self.formation,
            'duration': self.duration,
            'confidence': self.confidence,
            'alternative_interpretations': self.alternative_interpretations or []
        }

class PlayParser:
    """Natural language basketball play parser"""
    
    def __init__(self):
        """Initialize the parser with NLP models and basketball knowledge"""
        self.nlp = self._load_spacy_model()
        self.basketball_vocab = self._load_basketball_vocabulary()
        self.action_patterns = self._create_action_patterns()
        self.position_mappings = self._create_position_mappings()
        self.formation_templates = self._load_formation_templates()
        
    def _load_spacy_model(self):
        """Load spaCy NLP model"""
        try:
            # Try to load the model
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
            return nlp
        except OSError:
            logger.warning("spaCy model not found, using blank model")
            # Fallback to blank model if not installed
            nlp = spacy.blank("en")
            # Add basic components
            nlp.add_pipe("sentencizer")
            return nlp
    
    def _load_basketball_vocabulary(self) -> Dict[str, List[str]]:
        """Load basketball-specific vocabulary and synonyms"""
        return {
            'actions': {
                'pass': ['pass', 'throw', 'dish', 'feed', 'deliver', 'swing', 'dump'],
                'dribble': ['dribble', 'drive', 'penetrate', 'attack', 'handle'],
                'screen': ['screen', 'pick', 'set pick', 'block', 'set screen'],
                'cut': ['cut', 'slash', 'run', 'move', 'backdoor', 'baseline'],
                'shot': ['shoot', 'shot', 'score', 'fire', 'pull up', 'fadeaway'],
                'rebound': ['rebound', 'board', 'grab', 'secure'],
            },
            'positions': {
                'pg': ['point guard', 'pg', 'point', 'one', '1'],
                'sg': ['shooting guard', 'sg', 'two', '2', 'shooter'],
                'sf': ['small forward', 'sf', 'three', '3', 'wing'],
                'pf': ['power forward', 'pf', 'four', '4', 'big'],
                'c': ['center', 'c', 'five', '5', 'post', 'big man']
            },
            'locations': {
                'paint': ['paint', 'lane', 'key', 'inside'],
                'perimeter': ['perimeter', 'outside', 'arc', '3-point line'],
                'corner': ['corner', 'baseline corner'],
                'wing': ['wing', 'side', 'elbow'],
                'top': ['top', 'top of key', 'point']
            }
        }
    
    def _create_action_patterns(self) -> List[Dict[str, Any]]:
        """Create regex patterns for common basketball actions"""
        return [
            {
                'pattern': r'(\w+)\s+(pass|passes|throw|throws|dish|dishes)\s+to\s+(\w+)',
                'action': ActionType.PASS,
                'groups': ['player', 'action_verb', 'target']
            },
            {
                'pattern': r'(\w+)\s+(screen|screens|pick|picks|sets?\s+(?:a\s+)?(?:screen|pick))\s+(?:for\s+)?(\w+)?',
                'action': ActionType.SCREEN,
                'groups': ['player', 'action_verb', 'target']
            },
            {
                'pattern': r'(\w+)\s+(cut|cuts|slash|slashes|run|runs)\s+(?:to\s+)?(?:the\s+)?(\w+)?',
                'action': ActionType.CUT,
                'groups': ['player', 'action_verb', 'location']
            },
            {
                'pattern': r'(\w+)\s+(dribble|dribbles|drive|drives)\s+(?:to\s+)?(?:the\s+)?(\w+)?',
                'action': ActionType.DRIBBLE,
                'groups': ['player', 'action_verb', 'location']
            },
            {
                'pattern': r'(\w+)\s+(shoot|shoots|score|scores|fire|fires)\s*(?:(?:a|an)\s+)?(\w+)?',
                'action': ActionType.SHOT,
                'groups': ['player', 'action_verb', 'shot_type']
            }
        ]
    
    def _create_position_mappings(self) -> Dict[str, PlayerPosition]:
        """Create mappings from text to player positions"""
        mappings = {}
        for pos, variations in self.basketball_vocab['positions'].items():
            for variation in variations:
                mappings[variation.lower()] = PlayerPosition[pos.upper()]
        return mappings
    
    def _load_formation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load standard basketball formation templates"""
        return {
            '5-out': {
                'description': 'Five players around the perimeter',
                'positions': {
                    'PG': (470, 400),  # Top of key
                    'SG': (600, 350),  # Right wing
                    'SF': (340, 350),  # Left wing
                    'PF': (600, 250),  # Right baseline
                    'C': (340, 250)    # Left baseline
                }
            },
            '4-out-1-in': {
                'description': 'Four perimeter players, one in post',
                'positions': {
                    'PG': (470, 400),
                    'SG': (600, 350),
                    'SF': (340, 350),
                    'PF': (570, 300),
                    'C': (470, 200)    # Post position
                }
            },
            '1-4-high': {
                'description': 'Point guard and four players at elbows/wings',
                'positions': {
                    'PG': (470, 450),
                    'SG': (600, 350),
                    'SF': (340, 350),
                    'PF': (600, 320),
                    'C': (340, 320)
                }
            }
        }
    
    def parse(self, description: str, formation: str = "5-out", context: Dict[str, Any] = None) -> ParsedPlay:
        """
        Parse natural language play description
        
        Args:
            description: Natural language play description
            formation: Starting formation (default: "5-out")
            context: Game context (score, time, etc.)
            
        Returns:
            ParsedPlay object with structured play data
        """
        logger.info(f"Parsing play: '{description}' with formation: {formation}")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(description)
        
        # Extract players and positions
        players = self._extract_players(cleaned_text, formation)
        
        # Extract actions sequence
        actions = self._extract_actions(cleaned_text, players)
        
        # Calculate timing and sequencing
        actions = self._calculate_timing(actions)
        
        # Estimate confidence
        confidence = self._calculate_confidence(cleaned_text, actions, players)
        
        # Generate alternative interpretations
        alternatives = self._generate_alternatives(actions)
        
        # Create play name
        play_name = self._generate_play_name(actions, formation)
        
        parsed_play = ParsedPlay(
            name=play_name,
            description=description,
            actions=actions,
            players=players,
            formation=formation,
            duration=max([a.timestamp + a.duration for a in actions]) if actions else 5.0,
            confidence=confidence,
            alternative_interpretations=alternatives
        )
        
        logger.info(f"Parsed {len(actions)} actions with confidence {confidence:.2f}")
        return parsed_play
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        # Convert to lowercase
        text = text.lower()
        
        # Expand common abbreviations
        abbreviations = {
            'pg': 'point guard',
            'sg': 'shooting guard',
            'sf': 'small forward',
            'pf': 'power forward',
            'c': 'center',
            '3pt': 'three point',
            '3-pt': 'three point',
            'ft': 'free throw',
            'p&r': 'pick and roll',
            'pnr': 'pick and roll'
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_players(self, text: str, formation: str) -> List[Player]:
        """Extract player information from text"""
        players = []
        
        # Get formation template
        formation_template = self.formation_templates.get(formation, self.formation_templates['5-out'])
        
        # Extract mentioned positions
        mentioned_positions = set()
        doc = self.nlp(text)
        
        for token in doc:
            token_text = token.text.lower()
            if token_text in self.position_mappings:
                mentioned_positions.add(self.position_mappings[token_text])
        
        # If no positions mentioned, use all positions from formation
        if not mentioned_positions:
            mentioned_positions = {PlayerPosition.PG, PlayerPosition.SG, PlayerPosition.SF, PlayerPosition.PF, PlayerPosition.C}
        
        # Create player objects
        for i, position in enumerate(mentioned_positions):
            pos_name = position.name
            location = formation_template['positions'].get(pos_name, (470, 300))
            
            players.append(Player(
                id=f"player_{i+1}",
                position=position,
                team="offense",
                location=location
            ))
        
        return players
    
    def _extract_actions(self, text: str, players: List[Player]) -> List[Action]:
        """Extract sequence of actions from text"""
        actions = []
        
        # Split text into sentences/clauses
        sentences = re.split(r'[,.;]|then|and then|after|next', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Try to match action patterns
            for pattern_info in self.action_patterns:
                pattern = pattern_info['pattern']
                action_type = pattern_info['action']
                groups = pattern_info['groups']
                
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    # Extract action details
                    action_data = {}
                    for j, group_name in enumerate(groups):
                        if j < len(match.groups()) and match.group(j+1):
                            action_data[group_name] = match.group(j+1).lower()
                    
                    # Find player
                    player_id = self._find_player_by_position(action_data.get('player', ''), players)
                    target_player = self._find_player_by_position(action_data.get('target', ''), players)
                    
                    # Create action
                    action = Action(
                        type=action_type,
                        player=player_id or f"player_{i+1}",
                        target_player=target_player,
                        timestamp=float(i),
                        duration=1.0,
                        description=sentence.strip(),
                        confidence=0.8
                    )
                    
                    actions.append(action)
                    break
        
        # If no actions found, create default action
        if not actions:
            actions.append(Action(
                type=ActionType.MOVEMENT,
                player="player_1",
                timestamp=0.0,
                duration=2.0,
                description="General play movement",
                confidence=0.5
            ))
        
        return actions
    
    def _find_player_by_position(self, position_text: str, players: List[Player]) -> Optional[str]:
        """Find player ID by position text"""
        if not position_text:
            return None
            
        position_text = position_text.lower()
        
        # Check direct position mappings
        if position_text in self.position_mappings:
            target_position = self.position_mappings[position_text]
            for player in players:
                if player.position == target_position:
                    return player.id
        
        # Check for numbered players (1, 2, 3, 4, 5)
        position_numbers = {'1': PlayerPosition.PG, '2': PlayerPosition.SG, '3': PlayerPosition.SF, '4': PlayerPosition.PF, '5': PlayerPosition.C}
        if position_text in position_numbers:
            target_position = position_numbers[position_text]
            for player in players:
                if player.position == target_position:
                    return player.id
        
        return None
    
    def _calculate_timing(self, actions: List[Action]) -> List[Action]:
        """Calculate timing and sequencing for actions"""
        for i, action in enumerate(actions):
            # Space actions 1-2 seconds apart
            action.timestamp = float(i * 1.5)
            
            # Adjust duration based on action type
            if action.type == ActionType.PASS:
                action.duration = 0.5
            elif action.type == ActionType.DRIBBLE:
                action.duration = 2.0
            elif action.type == ActionType.SCREEN:
                action.duration = 3.0
            elif action.type == ActionType.CUT:
                action.duration = 1.5
            elif action.type == ActionType.SHOT:
                action.duration = 1.0
            else:
                action.duration = 1.0
        
        return actions
    
    def _calculate_confidence(self, text: str, actions: List[Action], players: List[Player]) -> float:
        """Calculate parsing confidence score"""
        confidence_factors = []
        
        # Factor 1: Number of recognized actions
        if actions:
            action_confidence = min(len(actions) / 3.0, 1.0)  # Up to 3 actions is optimal
            confidence_factors.append(action_confidence)
        else:
            confidence_factors.append(0.2)
        
        # Factor 2: Basketball vocabulary coverage
        basketball_words = []
        for word_list in self.basketball_vocab['actions'].values():
            basketball_words.extend(word_list)
        
        text_words = text.lower().split()
        basketball_word_count = sum(1 for word in text_words if word in basketball_words)
        vocab_confidence = min(basketball_word_count / max(len(text_words), 1), 1.0)
        confidence_factors.append(vocab_confidence)
        
        # Factor 3: Player position recognition
        if players:
            player_confidence = min(len(players) / 5.0, 1.0)  # Standard 5 players
            confidence_factors.append(player_confidence)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        return round(overall_confidence, 2)
    
    def _generate_alternatives(self, actions: List[Action]) -> List[str]:
        """Generate alternative interpretations of the play"""
        alternatives = []
        
        # Alternative action interpretations
        if any(action.type == ActionType.SCREEN for action in actions):
            alternatives.append("Could be interpreted as a pick-and-roll variation")
        
        if any(action.type == ActionType.CUT for action in actions):
            alternatives.append("Cutting action could be a backdoor or baseline cut")
        
        if any(action.type == ActionType.PASS for action in actions):
            alternatives.append("Pass could include a fake or pump fake")
        
        # Add maximum 3 alternatives
        return alternatives[:3]
    
    def _generate_play_name(self, actions: List[Action], formation: str) -> str:
        """Generate a descriptive name for the play"""
        if not actions:
            return f"{formation} Formation"
        
        # Common play patterns
        action_types = [action.type for action in actions]
        
        if ActionType.SCREEN in action_types and ActionType.PASS in action_types:
            return "Pick and Pass"
        elif ActionType.SCREEN in action_types and ActionType.CUT in action_types:
            return "Screen and Cut"
        elif ActionType.PASS in action_types and ActionType.CUT in action_types:
            return "Pass and Cut"
        elif ActionType.DRIBBLE in action_types and ActionType.SCREEN in action_types:
            return "Drive and Screen"
        elif ActionType.CUT in action_types:
            return "Cutting Play"
        elif ActionType.SCREEN in action_types:
            return "Screen Play"
        else:
            return f"{formation} Movement"

# Example usage and testing
if __name__ == "__main__":
    # Test the parser
    parser = PlayParser()
    
    test_plays = [
        "PG passes to SG, center sets screen, cut to basket",
        "Point guard dribbles left, power forward sets pick, shooting guard cuts to corner",
        "Run a pick and roll with center, then pass to wing for three-pointer",
        "1 passes to 2, 5 screens for 3, backdoor cut"
    ]
    
    for play_desc in test_plays:
        print(f"\n--- Testing: '{play_desc}' ---")
        result = parser.parse(play_desc)
        print(f"Play Name: {result.name}")
        print(f"Confidence: {result.confidence}")
        print(f"Actions: {len(result.actions)}")
        for action in result.actions:
            print(f"  - {action.type.value}: {action.description}")
        print(f"Players: {len(result.players)}")
        print("="*50)
