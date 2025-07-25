# ==== Q2E PATCH IMPORTS ====
import numpy as np
from collections import deque
from dataclasses import dataclass
import time  # For performance tracking
# ===========================

import base64
import io
import json
import logging
import textwrap
import random  # Add this to imports at the top of the file
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..structs import FrameData, GameAction
from ..agent import Agent

# Import or define BeamSearchPlanner
# If BeamSearchPlanner is not available, define a dummy class as fallback
class BeamSearchPlanner:
    def __init__(self, agent, beam_width=3, search_depth=3):
        self.agent = agent
        self.beam_width = beam_width
        self.search_depth = search_depth
    
    def plan(self, frame):
        """Plan a sequence of actions using beam search"""
        # Start with the current state
        state_id = self.agent._calculate_state_id(frame)
        
        # Get all available actions
        available_actions = self.agent._get_available_actions(state_id)
        if not available_actions:
            return GameAction.RESET  # No valid actions
            
        # If we only have one action, just return it
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Initialize beam with starting state
        beam = [([], 0.0, frame)]  # (action_path, cumulative_value, frame)
        
        # Run beam search for specified depth
        for _ in range(self.search_depth):
            candidates = []
            
            # Generate next states from current beam
            for action_path, value, curr_frame in beam:
                # Try each possible action
                for action in available_actions:
                    # We can't actually simulate actions without environment
                    # So we'll use heuristics to estimate the value
                    action_value = self._estimate_action_value(curr_frame, action)
                    
                    # Create new candidate
                    new_path = action_path + [action]
                    new_value = value + action_value
                    
                    # For simplicity, we just use the same frame
                    # In a real implementation, we would simulate the effect
                    candidates.append((new_path, new_value, curr_frame))
            
            # Sort by value and keep top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]
        
        # Return the first action of the best path
        if beam and beam[0][0]:
            return beam[0][0][0]
        
        # Fallback
        return self.agent._select_heuristic_action(state_id, frame)
    
    def _estimate_action_value(self, frame, action):
        """Estimate the value of taking an action with caching"""
        # Compute a cache key for this estimation
        state_id = self.agent._calculate_state_id(frame)
        cache_key = (state_id, action.name)
        
        # Check if we have a cached value
        if hasattr(self, '_value_cache') and cache_key in self._value_cache:
            return self._value_cache[cache_key]
        
        # Calculate the value
        grid = self.agent._get_grid_array(frame)
        player_pos = self.agent._find_player_position(grid)
        
        # Use the agent's directional preference if possible
        if hasattr(self.agent, '_directional_preference'):
            value = self.agent._directional_preference(grid, action, player_pos)
        else:
            # Simple random value as fallback
            value = self.agent.rng.random()
        
        # Cache the result
        if not hasattr(self, '_value_cache'):
            self._value_cache = {}
        self._value_cache[cache_key] = value
        
        return value

# Define our own ReasoningActionResponse class
class ReasoningActionResponse(BaseModel):
    """Response structure for reasoning agent actions"""
    name: str = Field(description="Name of the action")
    reason: str = Field(description="Reason for selecting this action")
    short_description: str = Field(description="Short description of the action")
    hypothesis: str = Field(description="Current hypothesis about game mechanics")
    aggregated_findings: str = Field(description="Aggregated findings from observations")

# Q2E core components
class QuotientField:
    """Core Q2E component tracking information-energy quotients"""
    def __init__(self):
        self.values = {}  # state_id -> quotient value
        self.info_gains = []
        self.energy_costs = []
        self.entropy_log = []
        self.stability = 1.0  # Initial stability
        
    def update(self, state_id, info_gain, energy_cost, entropy):
        """Update quotient field according to Q2E law"""
        q_prev = self.values.get(state_id, 0.0)
        delta_q = info_gain - energy_cost
        self.values[state_id] = q_prev + delta_q
        
        # Log metrics
        self.info_gains.append(info_gain)
        self.energy_costs.append(energy_cost)
        self.entropy_log.append(entropy)
        
        # Update stability based on recent metrics
        self.update_stability()
        
        return self.values[state_id], delta_q
    
    def update_stability(self):
        """Track system stability based on recent entropy and info gain trends"""
        if len(self.info_gains) < 3:
            return
            
        # Stability decreases if entropy rises while info gain falls
        recent_entropy_trend = self.entropy_log[-1] - self.entropy_log[-3]
        recent_info_trend = self.info_gains[-1] - self.info_gains[-3]
        
        if recent_entropy_trend > 0 and recent_info_trend < 0:
            self.stability = max(0.1, self.stability * 0.8)  # Decrease stability
        else:
            self.stability = min(1.0, self.stability * 1.2)  # Increase stability


class Q2EConfig:
    """Configuration for Q2E controller"""
    def __init__(self):
        self.alpha = 0.6      # weight for reward delta
        self.beta = 0.3       # weight for entropy delta
        self.gamma = 0.1      # weight for novelty
        self.stagnation_k = 5  # steps before stagnation
        self.min_improve = 1e-4  # minimum improvement threshold


def grid_entropy(grid):
    """Calculate Shannon entropy of grid values"""
    if grid.size == 0:
        return 0.0
    
    # Get value counts
    _, counts = np.unique(grid, return_counts=True)
    probs = counts / grid.size
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy

class Q2EController:
    """Manages Q2E metrics and updates"""
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.prev_entropy = None
        self.Q = 0.0
        self.prev_score = None
        self.stagnation = 0
        self.seen_states = set()
        self.rng = rng if rng is not None else np.random.default_rng(seed=42)

        # Add these after other initializations
        self.start_pos = (0, 0)  # Used in _strategy_return_to_start
        self.high_level_pr = 1.0  # Used in dimensionality analysis
        self.low_level_pr = 1.0   # Used in dimensionality analysis
        self.reward_history = []  # For feature correlation

    def update(self, grid, score):
        """Update Q2E metrics using real calculations"""
        # Calculate Shannon entropy and normalize
        entropy = grid_entropy(grid)
        hmax = np.log2(grid.size) if grid.size > 0 else 1.0
        h_norm = entropy / hmax
        
        # Calculate deltas
        entropy_delta = h_norm - self.prev_entropy if self.prev_entropy is not None else 0
        reward_delta = score - self.prev_score if self.prev_score is not None else 0
        
        # Update previous values
        self.prev_entropy = h_norm
        self.prev_score = score
        
        # Calculate novelty (binary: 1 for new, -0.5 for seen)
        grid_hash = hash(grid.tobytes())
        novelty = -0.5 if grid_hash in self.seen_states else 1.0
        self.seen_states.add(grid_hash)
        
        # Calculate ΔQ with proper signs: α·ΔR - β·ΔH + γ·N
        dq_raw = self.cfg.alpha * reward_delta - self.cfg.beta * entropy_delta + self.cfg.gamma * novelty
        delta_q = float(np.tanh(dq_raw))  # Keep in [-1, 1]
        self.Q += delta_q
        
        # Check for improvement to reset stagnation counter
        improved = (abs(delta_q) > self.cfg.min_improve) or (reward_delta != 0) or (abs(entropy_delta) > 1e-3)
        if improved:
            self.stagnation = 0
        else:
            self.stagnation += 1
        
        return {
            "entropy": h_norm,
            "entropy_delta": entropy_delta,
            "delta_q": delta_q,
            "Q": self.Q,
            "novelty": novelty,
            "stagnation": self.stagnation,
            "reward_delta": reward_delta
        }
        
    def _object_in_direction_score(self, grid, y, x, obj_value, action):
        """Check for objects in the specified direction"""
        h, w = grid.shape
        
        # Direction vectors
        if action == GameAction.ACTION1:  # UP
            dy, dx = -1, 0
        elif action == GameAction.ACTION2:  # DOWN
            dy, dx = 1, 0
        elif action == GameAction.ACTION3:  # LEFT
            dy, dx = 0, -1
        elif action == GameAction.ACTION4:  # RIGHT
            dy, dx = 0, 1
        else:
            return 0.0
        
        # Look ahead up to 3 cells
        for i in range(1, 4):
            ny, nx = y + i*dy, x + i*dx
            if 0 <= ny < h and 0 <= nx < w:
                if grid[ny, nx] == obj_value:
                    return 1.0 / i  # Closer objects have higher score
                elif grid[ny, nx] == 10:  # Wall blocks view
                    break
        
        return 0.0

# Q2EReasoningAgent class
class Q2EReasoningAgent(Agent):
    """Q2E agent that combines pattern detection with exploration strategies"""
    
    REASONING_EFFORT = "medium"  # Define the constant that was missing

    def _get_grid_array(self, frame: FrameData) -> np.ndarray:
        """
        Convert the frame data to a NumPy array for grid processing.
        Assumes frame.frame is a list of lists or similar structure.
        """
        if hasattr(frame, "frame") and frame.frame:
            return np.array(frame.frame[0])
        return np.array([])

    def _detect_stagnation(self, state_id: str) -> bool:
        """
        Detects stagnation based on reward plateau, repeated state visits, and low recent info gain.
        Returns True if stagnation is detected, otherwise False.
        """
        reward_plateau = self.reward_plateau if hasattr(self, "reward_plateau") else False
        state_cycling = self.visits.get(state_id, 0) > 3
        recent_info_gains = getattr(self, "recent_info_gains", [0.0])
        low_info_gain = recent_info_gains[-1] < 0.05 if recent_info_gains else False
        return reward_plateau or state_cycling or low_info_gain
    
    def __init__(self, *args, **kwargs):
        root_url = kwargs.get('ROOT_URL')
        super().__init__(*args, **kwargs)
        self.root_url = root_url
        self._init_q2e_fields()
        self._init_rng()
        self._init_llm()
        self._init_dynamic_params()
        self._init_thresholds()
        
        # Hierarchical control initialization
        self._init_hierarchical_control()
        
        # Initialize feature importance tracking
        self.feature_importance = {}

        self.q2e_cfg = Q2EConfig()
        self.q2e = Q2EController(self.q2e_cfg, rng=self.rng)
        self.beam_search_planner = BeamSearchPlanner(self, beam_width=3, search_depth=3)
        self.use_beam_search = True

        # Initialize strategy performance tracking
        self.strategy_performance = []

        if not hasattr(self, 'strategy_history'):
            self.strategy_history = []  # Initialize if it doesn't exist

        # Initialize verbose attribute to avoid attribute errors
        self.verbose = False

        # Add these attributes
        self.feature_history = []
        self.current_strategy = "explore_new_areas"  # Default strategy
        self.frame_timestamp = 0
        # Add grid_log
        self.grid_log = []
        self.high_level_states = []
        self.prev_state_id = None

    def _init_q2e_fields(self):
        self.Q = QuotientField()
        self.blacklisted_pairs = set()
        self.singularity_events = []
        self.visits = {}
        self.history = []
        self.action_counter = 0
        self.action_effects = {}
        self.state_changes = []
        self.plateau_counter = 0
        self.last_reward = 0
        self.reward_plateau = False

    def _init_rng(self):
        self.rng = np.random.default_rng(seed=42)

    def _init_llm(self):
        self.use_llm = False
        try:
            self.client = OpenAI()
            self.use_llm = True
        except Exception as e:
            logging.warning(f"Could not initialize OpenAI client: {e}")
            logging.info("Running in standalone Q2E mode without LLM reasoning")

    def _init_dynamic_params(self):
        self.dynamic_tau = 0.5
        self.recent_info_gains = []
        self.recent_delta_qs = []

    def _init_thresholds(self):
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.3
        self.loop_threshold = 3
        self.singularity_threshold = 0.2
    
    def _init_hierarchical_control(self):
        """Initialize hierarchical control structures"""
        # High-level cycle parameters
        self.high_level_cycle = 5  # Run high-level reasoning every
        self.cycle_counter = 0
        
        # Memory structures
        self.high_level_memory = {
            'strategy_history': [],       # List of (cycle, strategy) tuples
            'performance_history': [],    # List of performance metrics by cycle
            'interesting_states': {},     # State_id -> reason for interest
            'global_patterns': {},        # Pattern type -> occurrences
            'feature_trends': {}          # Feature -> trend direction
        }
        
        # Add this line to initialize feature_history
        self.feature_history = []
        
        self.low_level_memory = {
            'state_visits': {},           # State_id -> visit count
            'state_transitions': {},      # (state_id, action) -> next_state_id
            'action_effects': {},         # (state_id, action) -> effect measure
            'recent_states': [],          # List of recent state_ids (short-term memory)
            'recent_features': [],        # List of recent feature vectors
            'local_patterns': set()       # Set of detected local patterns
        }
        
        # Memory update parameters
        self.high_level_memory_size = 20  # Max cycles to remember
        self.low_level_memory_size = 50   # Max steps to remember in detail
        self.memory_decay_rate = 0.95     # Memory decay factor for old entries
    
    # ==== Q2E PATCH CHOOSE_ACTION ====

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        # Store frames and increment counter
        self.frames = frames
        self.action_counter += 1
        
        # Handle reset cases
        if latest_frame.full_reset:
            self.history = []
            self._initialize_visit_tracking(latest_frame)
            return GameAction.RESET

        # Initialize visit tracking
        initial_action = self._initialize_visit_tracking(latest_frame)
        if initial_action:
            return initial_action
            
        # Process frame and select action
        return self._process_frame_action(latest_frame)

    def _initialize_visit_tracking(self, latest_frame):
        """Initialize cell visit tracking"""
        # Only initialize the visit tracking data structure
        if not hasattr(self, 'visited_cells'):
            self.visited_cells = {}  # (y, x) -> visit_count
    
        # Track state visit count
        state_id = self._calculate_state_id(latest_frame)
        self.visits[state_id] = self.visits.get(state_id, 0) + 1
    
        # First action special case
        if not self.history:
            action = GameAction.RESET
            self._initial_response(action)
            return action
    
        # This should be all this method does
        return None

    def _select_heuristic_action(self, state_id: str, frame: FrameData) -> GameAction:
        """Select action using improved heuristics"""
        grid = self._get_grid_array(frame)
        available_actions = [
            a for a in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4] 
            if (state_id, a.name) not in self.blacklisted_pairs
        ]
        
        if not available_actions:
            # If all actions are blacklisted, clear blacklist for this state
            available_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
            self.blacklisted_pairs = {pair for pair in self.blacklisted_pairs if pair[0] != state_id}
        
        # Score each action
        action_scores = {}
        player_pos = self._find_player_position(grid)
        
        for action in available_actions:
            # Base score (from past effectiveness)
            base_score = self.action_effects.get(action.name, 0.5)
            
            # Add directional preference based on grid content
            direction_score = self._directional_preference(grid, action, player_pos)
            
            # Adjust exploration vs exploitation based on stagnation
            novelty_factor = min(0.8, 0.2 + (0.05 * self.q2e.stagnation))
            
            # Calculate final score
            action_scores[action] = (
                (1 - novelty_factor) * (base_score + direction_score) + 
                novelty_factor * self.rng.random()
            )

        # Store for dashboard
        self.last_action_scores = action_scores
        
        # Return best action
        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to random action
        return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])

    def _directional_preference(self, grid: np.ndarray, action: GameAction, player_pos) -> float:
        """
        Enhanced heuristic scoring for action preference based on grid and player position.
        """
        # Early return for invalid inputs
        if grid.size == 0 or player_pos is None or player_pos[0] is None:
            return 0.0
        
        y, x = player_pos
        score = 0.0
        
        # Add object-based scores
        score += self._calculate_object_scores(grid, y, x, action)
        
        # Add wall avoidance score
        score += self._calculate_wall_score(grid, y, x, action)
        
        # Add exploration score
        score += self._calculate_exploration_score(y, x, action, grid.shape)
        
        return score

    def _calculate_object_scores(self, grid, y, x, action):
        """Calculate scores for valuable objects in the given direction"""
        score = 0.0
        
        # Look for keys (value 7)
        score += self.q2e._object_in_direction_score(grid, y, x, 7, action) * 3.0
        
        # Look for doors (value 11)
        score += self.q2e._object_in_direction_score(grid, y, x, 11, action) * 2.0
        
        return score

    def _calculate_wall_score(self, grid, y, x, action):
        """Calculate penalty for walls in the given direction"""
        # Get coordinates of the next cell in the direction of action
        next_y, next_x = self._get_next_position(y, x, action)
        
        # Check if position is valid and contains a wall
        if (0 <= next_y < grid.shape[0] and 
            0 <= next_x < grid.shape[1] and
            grid[next_y, next_x] == 10):
            return -1.0
        
        return 0.0

    def _calculate_exploration_score(self, y, x, action, grid_shape):
        """Calculate score for unexplored areas"""
        if not hasattr(self, 'visited_cells'):
            return 0.0
        
        # Get coordinates of the next cell in the direction of action
        next_y, next_x = self._get_next_position(y, x, action)
        
        # Check if position is valid
        if (0 <= next_y < grid_shape[0] and 
            0 <= next_x < grid_shape[1]):
            # Favor less visited cells
            return 0.5 / (self.visited_cells.get((next_y, next_x), 0) + 1)
        
        return 0.0

    def _get_next_position(self, y, x, action):
        """Get the next position based on the action"""
        if action == GameAction.ACTION1:  # UP
            return y-1, x
        elif action == GameAction.ACTION2:  # DOWN
            return y+1, x
        elif action == GameAction.ACTION3:  # LEFT
            return y, x-1
        elif action == GameAction.ACTION4:  # RIGHT
            return y, x+1
    
        return y, x  # Default: stay in place
# =====================================
    
    def _find_player_position(self, grid: np.ndarray):
        """
        Find the player's position in the grid.
        Assumes the player is represented by a unique value, e.g., 1.
        Returns (y, x) tuple or (None, None) if not found.
        """
        if grid.size == 0:
            return (None, None)
        positions = np.argwhere(grid == 1)
        if positions.shape[0] > 0:
            return tuple(positions[0])
        return (None, None)

# =====================================
    
    def _initial_response(self, action):
        """Create initial response for first action"""
        initial_response = ReasoningActionResponse(
            name="RESET",
            reason="Initial action to start the game and observe the environment.",
            short_description="Start game with Q2E tracking enabled",
            hypothesis="The game requires a RESET to begin. Q2E framework will track information gain.",
            aggregated_findings="No findings yet. Q2E metrics initialized.",
        )
        self.history.append(initial_response)
        return action
    
    def _calculate_state_id(self, frame: FrameData) -> str:
        """Generate unique state identifier from frame data"""
        grid_str = str(frame.frame[0] if frame.frame else [])
        return hash(grid_str).__str__()
    
    @staticmethod
    def get_readable_state_id(state_id: str) -> str:
        """Convert hash-based state ID to a more readable format"""
        if not state_id:
            return "empty"
        # Take first 8 chars of the hash
        short_id = state_id[:8] if len(state_id) > 8 else state_id
        return f"state_{short_id}"
    
    def _select_exploration_action(self, frame: FrameData) -> GameAction:
        """Enhanced action selection with dynamic tau and effectiveness tracking"""
        actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        
        state_id = self._calculate_state_id(frame)
        
        # Filter blacklisted actions
        filtered_actions = [a for a in actions 
                           if (state_id, a.name) not in self.blacklisted_pairs]
        
        # If all actions are blacklisted, reset blacklist for this state
        if not filtered_actions:
            filtered_actions = actions
            self.blacklisted_pairs = {pair for pair in self.blacklisted_pairs 
                                     if pair[0] != state_id}
            
        # Calculate dynamic tau based on stability, feature novelty, and trend
        feature_weight = min(1.0, len(self.history) / 30)
        novelty_weight = 1.0 - (self.visits.get(state_id, 0) / max(sum(self.visits.values()), 1.0))
        trend_weight = 0.0
        
        if hasattr(self, 'recent_delta_qs') and len(self.recent_delta_qs) > 5:
            trend = sum(self.recent_delta_qs[-3:]) - sum(self.recent_delta_qs[:3])
            trend_weight = min(0.3, max(-0.3, trend))
        
        # Calculate dynamic tau
        self.dynamic_tau = 0.7 * (self.Q.stability) + 0.2 * feature_weight + 0.1 * novelty_weight + 0.05 * trend_weight
        
        # Detect stagnation
        stagnation_detected = self._detect_stagnation(state_id)
        if stagnation_detected:
            # Increase exploration when stagnated
            self.dynamic_tau = max(0.9, self.dynamic_tau)
        
        # Calculate action scores based on effectiveness and exploration
        action_scores = {}
        for action in filtered_actions:
            # Base exploitation score
            exploit_score = self.action_effects.get(action.name, 0.5)
            
            # Grid-based heuristic score
            grid = self._get_grid_array(frame)
            player_pos = self._find_player_position(grid)
            direction_score = self._directional_preference(grid, action, player_pos)
            
            # Exploration score (with randomness)
            explore_score = self.rng.random()
            
            # Combine scores based on dynamic_tau (higher tau = more exploration)
            action_scores[action] = (1 - self.dynamic_tau) * (exploit_score + direction_score) + self.dynamic_tau * explore_score
        
        # Return best action or fallback to random if no scores
        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        
        return random.choice(filtered_actions) if filtered_actions else GameAction.ACTION1
    
    def _process_frame_action(self, latest_frame):
        grid = self._get_grid_array(latest_frame)
        
        # Extract features
        features = self._extract_features(latest_frame)
        self.current_features = features

        # UPDATE: Add this line to define state_id before using it
        state_id = self._calculate_state_id(latest_frame)
        
        # Update tracking
        player_pos = self._find_player_position(grid)
        self._update_visit_tracking(player_pos)
        
        # Get Q2E metrics
        metrics = self._get_q2e_metrics(grid, latest_frame.score)
        
        # Track feature importance for supervision
        self._track_feature_importance(features, metrics)
        
        # Track metrics for adaptive halting
        if 'info_gain' in metrics:
            self.recent_info_gains.append(metrics['info_gain'])
            if len(self.recent_info_gains) > 10:
                self.recent_info_gains.pop(0)

        # Now state_id is defined when used here
        halt_decision = self._implement_adaptive_halting(state_id, latest_frame, features, metrics)

        # Periodically log grid state (every 10 steps)
        if self.action_counter % 10 == 0:
            grid = self._get_grid_array(self.frames[-1]) if hasattr(self, 'frames') and self.frames else None
            frame = self.frames[-1] if hasattr(self, 'frames') and self.frames else None
            if grid is not None and frame is not None:
                self._log_grid_state(grid, frame, f"Grid at step {self.action_counter}")
    
        # Apply halting decision with enhanced logging
        if halt_decision == 'halt':
            logging.info("🚀 Halting mechanism: Confident in current strategy")
            action = self._apply_strategy(state_id, latest_frame, features)
            if action:
                return action
        elif halt_decision == 'reset':
            logging.info("🔄 Halting mechanism: Forcing strategy reset")
            old_strategy = self.current_strategy
            self._select_new_strategy(latest_frame, features, metrics)
            logging.info(f"Changed strategy from {old_strategy} to {self.current_strategy}")
            return GameAction.RESET
        elif halt_decision == 'ponder':
            logging.info("🤔 Halting mechanism: Increasing exploration to ponder")
            self.dynamic_tau = min(1.0, self.dynamic_tau + 0.2)
        
        # HIGH-LEVEL REASONING (runs periodically)
        if self.action_counter % self.high_level_cycle == 0:
            # Log before high-level reasoning
            logging.info(f"👑 Starting high-level reasoning cycle {self.cycle_counter}")
            
            # Run high-level reasoning
            self._high_level_reasoning(latest_frame, features, metrics)
            self.cycle_counter += 1
            
            # Enhanced logging after high-level reasoning
            self._log_high_level_state(metrics)
    
        # LOW-LEVEL ACTION SELECTION
        action = self._low_level_action_selection(state_id, latest_frame, features, metrics)
        
        # Log action and update history
        self._log_action_metrics(action, state_id, metrics)
        
        return action

    def _get_q2e_metrics(self, grid, score):
        """Get Q2E metrics for the current grid and score"""
        metrics = self.q2e.update(grid, score)
        return metrics

    def _handle_stagnation(self, state_id, metrics):
        """Handle stagnation conditions and return action if needed"""
        stagnation = metrics["stagnation"]
        
        if stagnation < self.q2e_cfg.stagnation_k:
            # No stagnation - adjust dynamic tau and continue
            self.dynamic_tau = max(0.5, self.dynamic_tau - 0.1)
            return None
            
        # Stagnation detected
        logging.info("Q2E: stagnation=%d; forcing strategy switch.", stagnation)
        
        # Clear blacklist for this state
        self.blacklisted_pairs = {pair for pair in self.blacklisted_pairs if pair[0] != state_id}
        
        # Extreme stagnation - reset
        if stagnation % (2 * self.q2e_cfg.stagnation_k) == 0:
            logging.info("Extreme stagnation detected. Triggering RESET.")
            return GameAction.RESET
        
        # Force higher exploration
        self.dynamic_tau = min(1.0, self.dynamic_tau + 0.2)
        
        # Blacklist the most recently taken action
        if self.history:
            last_action = self.history[-1]
            self.blacklisted_pairs.add((state_id, last_action))
        
        return None

    def _update_visit_tracking(self, player_pos):
        """Update cell visit tracking"""
        if not hasattr(self, 'visited_cells'):
            self.visited_cells = {}  # (y, x) -> visit_count
    
        if player_pos and None not in player_pos:
            pos = (int(player_pos[0]), int(player_pos[1]))
            self.visited_cells[pos] = self.visited_cells.get(pos, 0) + 1

    def _select_and_validate_action(self, state_id, frame, metrics):
        """Select an action and validate it's not blacklisted"""
        # First try beam search if enabled
        if self.use_beam_search:
            action = self.beam_search_planner.plan(frame)
        else:
            # Otherwise use heuristic selection
            action = self._select_heuristic_action(state_id, frame)
    
        # Check if action is blacklisted for this state
        if (state_id, action.name) in self.blacklisted_pairs:
            logging.info("Avoiding blacklisted action %s for state %s", action.name, state_id)
            # Try other actions
            alt_actions = [a for a in [GameAction.ACTION1, GameAction.ACTION2, 
                                      GameAction.ACTION3, GameAction.ACTION4] 
                          if (state_id, a.name) not in self.blacklisted_pairs]
            if alt_actions:
                action = random.choice(alt_actions)
    
        # Add to history and check for loops
        if hasattr(self, 'history'):
            self.history.append(action.name)
            self._check_and_blacklist_loops(state_id, action)
    
        return action

    def _log_action_metrics(self, action, state_id, metrics):
        """Log action and Q2E metrics"""
        logging.info(
            f"Action {action} ΔQ={metrics['delta_q']:.4f}, ΔH={metrics['entropy_delta']:.4f}, "
            f"R={metrics['reward_delta']:.2f}, N={metrics['novelty']:.2f}, "
            f"Q={metrics['Q']:.4f}, stag={metrics['stagnation']}"
        )
        
        # Periodically display dashboard
        if self.action_counter % 10 == 0:
            self._print_q2e_dashboard(state_id, action, 
                                    metrics.get('info_gain', 0), 
                                    metrics.get('energy_cost', 0), 
                                    metrics['entropy'], 
                                    metrics['delta_q'], 
                                    metrics['Q'])
    def _print_q2e_dashboard(self, state_id, action, info_gain, energy_cost, entropy, delta_q, q_total):
        """Print a dashboard with Q2E metrics"""
        print("\n┌─────────────── Q2E DASHBOARD ───────────────┐")
        print(f"│ State: {self.get_readable_state_id(state_id)}")
        print(f"│ Action: {action.name}")
        print(f"│ Q-value: {q_total:.4f} (Δ: {delta_q:.4f})")
        print(f"│ Entropy: {entropy:.4f}")
        print(f"│ Info gain: {info_gain:.4f}")
        print(f"│ Energy cost: {energy_cost:.4f}")
        print(f"│ Visits: {self.visits.get(state_id, 0)}")
        print(f"│ Stagnation: {self.q2e.stagnation if hasattr(self, 'q2e') else 0}")
        print("└────────────────────────────────────────────────┘")
    def _check_and_blacklist_loops(self, state_id, action):
        """Check for action loops and blacklist problematic state-action pairs"""
        # Need at least 3 actions to detect a loop
        if len(self.history) < 3:
            return
        
        # Check for immediate loops (e.g., LEFT-RIGHT-LEFT)
        if len(self.history) >= 3:
            if self.history[-1] == self.history[-3] and self.history[-2] != self.history[-1]:
                # Simple oscillation detected (e.g., LEFT-RIGHT-LEFT)
                self.blacklisted_pairs.add((state_id, action.name))
                logging.warning(f"Detected simple oscillation: {self.history[-3:]}, blacklisting {action.name}")
        
        # Check for longer loops if we have enough history
        if len(self.history) >= 8:
            # Check for patterns like (A-B-C-D-A-B-C-D)
            pattern_len = 4
            if self.history[-pattern_len:] == self.history[-(pattern_len*2):-pattern_len]:
                # Pattern repetition detected
                self.blacklisted_pairs.add((state_id, action.name))
                logging.warning(f"Detected pattern repetition, blacklisting {action.name}")
                
                # Log singularity event
                if hasattr(self, 'singularity_events'):
                    self.singularity_events.append({
                        'step': self.action_counter,
                        'state_id': state_id,
                        'action': action.name,
                        'stability': getattr(self.Q, 'stability', 1.0) if hasattr(self, 'Q') else 1.0,
                        'visits': self.visits.get(state_id, 0),
                        'reason': 'Loop detected'
                    })
                    logging.warning("Q2E Singularity detected: %s", self.singularity_events[-1])
    def _get_available_actions(self, state_id):
        """Get available actions (not blacklisted)"""
        all_actions = [GameAction.ACTION1, GameAction.ACTION2, 
                      GameAction.ACTION3, GameAction.ACTION4]
        return [a for a in all_actions 
                if (state_id, a.name) not in self.blacklisted_pairs]
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """
        Determine if the current state is terminal or the agent's task is complete.
        Required implementation of abstract method from Agent base class.
        """
        # Check if we've reached a winning state (positive score)
        if latest_frame.score > 0:
            return True

        # Check for terminal signal from environment
        if getattr(latest_frame, "terminal", False):
            return True

        # Check for extreme stagnation
        if hasattr(self, "q2e") and self.q2e.stagnation > 40:
            logging.info("Agent terminating due to extreme stagnation")
            return True

        # Continue by default
        return False

    def _high_level_reasoning(self, frame, features, metrics):
        """High-level strategic reasoning with enhanced supervision and dimensionality monitoring"""
        # Update cycle length adaptively
        self._update_adaptive_cycle_length(features)
        
        # Update high-level memory
        self._update_high_level_memory(features, metrics)
        
        # Monitor dimensionality of agent behavior
        self._monitor_dimensionality()
        
        # Evaluate current strategy performance
        performance = self._evaluate_strategy_performance(metrics)
        self.strategy_performance.append(performance)
        
        # Log deep statistics (every 5 cycles)
        if self.cycle_counter % 5 == 0:
            self._log_deep_statistics(metrics)
        
        # Determine if we need to change strategy
        strategy_change_needed = self._should_change_strategy(metrics)
        
        # Also change strategy if we detect dimensionality collapse
        if hasattr(self, 'dimensionality_status'):
            high_level = self.dimensionality_status.get('high_level', {})
            if high_level.get('status') in ['severe_collapse', 'moderate_collapse']:
                strategy_change_needed = True
    
        if strategy_change_needed:
            old_strategy = self.current_strategy
            self._select_new_strategy(frame, features, metrics)
            logging.info(f"👑 Changed high-level strategy from {old_strategy} to {self.current_strategy}")
    
        # Store high-level state
        self._store_high_level_state(features, metrics)

    def _update_adaptive_cycle_length(self, features):
        """
        Dynamically adjust the high-level reasoning cycle length based on feature trends.
        This is a placeholder implementation; you can expand it as needed.
        """
        # Example: If state novelty is high, reason more frequently
        if features.get('state_novelty', 0) > 0.7:
            self.high_level_cycle = max(3, self.high_level_cycle - 1)
        elif features.get('state_novelty', 0) < 0.3:
            self.high_level_cycle = min(10, self.high_level_cycle + 1)
        # Otherwise, keep the cycle unchanged

    def _low_level_action_selection(self, state_id, frame, features, metrics):
        """Low-level tactical action selection with memory integration"""
        # First check memory-based overrides
        memory_action = self._use_memory_for_decision(state_id, frame, features, metrics)
        if memory_action is not None:
            return memory_action
        
        # Apply current high-level strategy
        strategy_action = self._apply_strategy(state_id, frame, features)
        if strategy_action is not None:
            return strategy_action
        
        # Update low-level memory
        self._update_low_level_memory(state_id, features)
        
        # Check for local stagnation or loops
        if self._detect_local_stuckness(state_id, metrics):
            return self._try_alternative_action(state_id, frame)
        
        # Use existing selection methods
        if self.use_beam_search:
            return self.beam_search_planner.plan(frame)
        else:
            return self._select_heuristic_action(state_id, frame)

    def _update_high_level_memory(self, features, metrics):
        """Update high-level strategic memory with deeper context tracking"""
        # Store strategy and performance history
        self.high_level_memory['strategy_history'].append((self.cycle_counter, self.current_strategy))
        self.high_level_memory['performance_history'].append({
            'cycle': self.cycle_counter,
            'Q': metrics.get('Q', 0),
            'stagnation': metrics.get('stagnation', 0),
            'score': self.frames[-1].score if self.frames else 0,
            'strategy': self.current_strategy
        })
        
        # Limit history size
        if len(self.high_level_memory['performance_history']) > self.high_level_memory_size:
            self.high_level_memory['performance_history'].pop(0)
        
        # Detect interesting states (high info gain, novel states)
        if self.frames:  # Add this check
            state_id = self._calculate_state_id(self.frames[-1])
            if features.get('state_novelty', 0) > 0.8:
                self.high_level_memory['interesting_states'][state_id] = 'high_novelty'
            if metrics.get('delta_q', 0) > 0.3:
                self.high_level_memory['interesting_states'][state_id] = 'high_info_gain'
            if self.frames[-1].score > 0:
                self.high_level_memory['interesting_states'][state_id] = 'positive_score'
    
        # Track feature trends
        if not hasattr(self, 'feature_history'):
            self.feature_history = []

        if len(self.feature_history) > 2:
            for key in features:
                if key in self.feature_history[-2]:
                    # Calculate trend over last 3 cycles
                    recent_values = [fh.get(key, 0) for fh in self.feature_history[-3:] if key in fh]
                    if len(recent_values) >= 3:
                        if all(x < y for x, y in zip(recent_values, recent_values[1:])):
                            self.high_level_memory['feature_trends'][key] = 'increasing'
                        elif all(x > y for x, y in zip(recent_values, recent_values[1:])):
                            self.high_level_memory['feature_trends'][key] = 'decreasing'
                        else:
                            self.high_level_memory['feature_trends'][key] = 'fluctuating'
        
        # Store current features for trend tracking
        self.feature_history.append(features)
        if len(self.feature_history) > self.high_level_memory_size:
            self.feature_history.pop(0)
        
        # Detect global patterns based on strategy effectiveness
        self._detect_global_patterns()
    
    def _detect_global_patterns(self):
        """
        Detect global patterns in agent's performance or environment.
        This is a placeholder; you can expand it with actual pattern detection logic.
        """
        # Example: Detect if a strategy is used repeatedly with poor results
        strategy_counts = {}
        for cycle, strategy in self.high_level_memory['strategy_history']:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Example: If a strategy is used more than half the cycles, mark as a global pattern
        total_cycles = len(self.high_level_memory['strategy_history'])
        for strategy, count in strategy_counts.items():
            if total_cycles > 0 and count / total_cycles > 0.5:
                self.high_level_memory['global_patterns'][strategy] = "dominant_strategy"

    def _update_low_level_memory(self, state_id, features):
        """Update low-level tactical memory with enhanced pattern detection"""
        # Update state visit count
        self.low_level_memory['state_visits'][state_id] = self.low_level_memory['state_visits'].get(state_id, 0) + 1
    
        # Update recent states
        self.low_level_memory['recent_states'].append(state_id)
        if len(self.low_level_memory['recent_states']) > self.low_level_memory_size:
            self.low_level_memory['recent_states'].pop(0)
    
        # Track recent feature values
        self.low_level_memory['recent_features'].append(features)
        if len(self.low_level_memory['recent_features']) > 5:
            self.low_level_memory['recent_features'].pop(0)
    
        # Track state transitions if we have a previous state and action
        if hasattr(self, 'history') and self.history and hasattr(self, 'prev_state_id'):
            action = self.history[-1]
            transition_key = (self.prev_state_id, action)
            self.low_level_memory['state_transitions'][transition_key] = state_id
            
            # Measure action effect based on feature changes
            if len(self.low_level_memory['recent_features']) >= 2:
                prev_features = self.low_level_memory['recent_features'][-2]
                curr_features = features
                
                # Calculate effect as the magnitude of feature changes
                effect = 0.0
                for key in set(prev_features.keys()) & set(curr_features.keys()):
                    effect += abs(curr_features[key] - prev_features[key])
                
                # Store action effect
                self.low_level_memory['action_effects'][transition_key] = effect
    
        # Detect local patterns (loops, oscillations)
        self._detect_local_patterns(features)
    
        # Save current state as previous state for next update
        self.prev_state_id = state_id

    def _store_high_level_state(self, features, metrics):
        """Store the current high-level state for later analysis"""
        state = {
            'cycle': self.cycle_counter,
            'strategy': self.current_strategy,
            'features': features,
            'Q': metrics.get('Q', 0),
            'stagnation': metrics.get('stagnation', 0)
        }
        
        if not hasattr(self, 'high_level_states'):
            self.high_level_states = []
        
        self.high_level_states.append(state)
        if len(self.high_level_states) > 10:  # Keep last 10 high-level states
            self.high_level_states.pop(0)

    def _extract_features(self, frame: FrameData) -> Dict[str, float]:
        """Extract meaningful features from the current frame for reasoning"""
        grid = self._get_grid_array(frame)
        if grid.size == 0:
            return {}
        
        # Basic statistical features
        features = {
            "mean_value": float(np.mean(grid)),
            "std_value": float(np.std(grid)),
            "unique_values": len(np.unique(grid)),
            "grid_entropy": grid_entropy(grid),
        }
        
        # Grid structure features
        features["grid_height"] = grid.shape[0]
        features["grid_width"] = grid.shape[1]
        
        # Player position
        player_pos = self._find_player_position(grid)
        if player_pos and None not in player_pos:
            features["player_y"] = player_pos[0]
            features["player_x"] = player_pos[1]
            features["player_dist_to_center"] = np.sqrt(
                (player_pos[0] - grid.shape[0]/2)**2 + 
                (player_pos[1] - grid.shape[1]/2)**2
            )
            
            # Distance to key objects
            key_positions = np.argwhere(grid == 7)  # Keys
            if key_positions.shape[0] > 0:
                closest_key = min([np.sum(np.abs(np.array(player_pos) - kp)) for kp in key_positions])
                features["dist_to_closest_key"] = float(closest_key)
                
            door_positions = np.argwhere(grid == 11)  # Doors
            if door_positions.shape[0] > 0:
                closest_door = min([np.sum(np.abs(np.array(player_pos) - dp)) for dp in door_positions])
                features["dist_to_closest_door"] = float(closest_door)
    
        # Object counts
        features["key_count"] = np.sum(grid == 7)
        features["door_count"] = np.sum(grid == 11)
        features["wall_count"] = np.sum(grid == 10)
        
        # Spatial patterns
        features["symmetry_score"] = self._calculate_symmetry(grid)
        features["connectivity"] = self._calculate_connectivity(grid)
        
        # Novelty features
        state_id = self._calculate_state_id(frame)
        features["state_novelty"] = 1.0 - min(1.0, self.visits.get(state_id, 0) / max(1, len(self.visits)))
        
        return features

    def _evaluate_strategy_performance(self, metrics):
        """Evaluate how well the current strategy is performing"""
        # For now, just use Q value as performance indicator
        return metrics.get('Q', 0)

    def _should_change_strategy(self, metrics):
        """Determine if we should change our high-level strategy"""
        # Check for strategy effectiveness
        if not hasattr(self, 'strategy_start_metrics'):
            self.strategy_start_metrics = metrics
            return False
            
        # Calculate improvement since strategy started
        q_improvement = metrics.get('Q', 0) - self.strategy_start_metrics.get('Q', 0)
        stagnation_increase = metrics.get('stagnation', 0) - self.strategy_start_metrics.get('stagnation', 0)
        
        # Q-head calculation (simple version)
        # Higher stagnation or lower Q improvement suggests changing strategy
        q_head = 1.0 / (1.0 + np.exp(-(q_improvement * 5 - stagnation_increase)))
        
        # Decision threshold and stagnation condition
        change_threshold = 0.7
        stagnation_threshold = 5
        
        return q_head > change_threshold or stagnation_increase > stagnation_threshold or metrics['stagnation'] > 10

    def _select_new_strategy(self, frame, features, metrics):
        """Select a new high-level strategy"""
        # Available strategies
        strategies = [
            "explore_new_areas",
            "target_nearest_key",
            "target_nearest_door",
            "follow_walls",
            "return_to_start",
            "random_walk"
        ]
        
        # Don't pick the current strategy
        available = [s for s in strategies if s != self.current_strategy]
        
        # Select new strategy based on features
        if features.get("key_count", 0) > 0 and features.get("dist_to_closest_key", float('inf')) < float('inf'):
            # If keys exist and we can reach them, prioritize getting them
            self.current_strategy = "target_nearest_key"
        elif features.get("door_count", 0) > 0 and features.get("dist_to_closest_door", float('inf')) < float('inf'):
            # If doors exist and we can reach them, prioritize getting to them
            self.current_strategy = "target_nearest_door"
        elif features.get("state_novelty", 0) < 0.3:
            # If we're in a well-explored area, try wall following to find new areas
            self.current_strategy = "follow_walls"
        elif self.visits.get(self._calculate_state_id(frame), 0) > 5:
            # If we've visited this state too many times, explore elsewhere
            self.current_strategy = "explore_new_areas"
        else:
            # Otherwise pick randomly from available strategies
            self.current_strategy = self.rng.choice(available)
        
        # Reset strategy metrics
        self.strategy_start_metrics = metrics
        
        # Log strategy change
        logging.info(f"Changing high-level strategy to: {self.current_strategy}")
        self.strategy_history.append((self.cycle_counter, self.current_strategy))

    def _apply_strategy(self, state_id, frame, features):
        """Apply the current high-level strategy to select actions"""
        grid = self._get_grid_array(frame)
        player_pos = self._find_player_position(grid)
    
        if player_pos is None or None in player_pos:
            return None  # Can't apply strategy without player position
        
        if self.current_strategy == "explore_new_areas":
            return self._strategy_explore_new_areas(state_id, frame, features)
        
        elif self.current_strategy == "target_nearest_key":
            return self._strategy_target_nearest_object(grid, player_pos, 7)  # 7 = key
        
        elif self.current_strategy == "target_nearest_door":
            return self._strategy_target_nearest_object(grid, player_pos, 11)  # 11 = door
        
        elif self.current_strategy == "follow_walls":
            return self._strategy_follow_walls(grid, player_pos)
            
        elif self.current_strategy == "return_to_start":
            return self._strategy_return_to_start(grid, player_pos, features)
            
        elif self.current_strategy == "random_walk":
            # implementation without the return None
            pass
    
        # Now this is reachable when none of the conditions match
        return None  # Fall back to normal action selection
    
    def _strategy_random_walk(self, state_id):
        """Strategy: Take random actions with memory to avoid immediate repetition"""
        available = self._get_available_actions(state_id)
        
        # If all actions are blacklisted, clear the blacklist
        if not available:
            available = [GameAction.ACTION1, GameAction.ACTION2, 
                        GameAction.ACTION3, GameAction.ACTION4]
        
        # Avoid the immediately previous action if possible
        if len(self.history) > 0 and len(available) > 1:
            prev_action_name = self.history[-1]
            prev_action = None
            for action in available:
                if action.name == prev_action_name:
                    prev_action = action
                    break
    
            if prev_action is not None and prev_action in available and len(available) > 1:
                available.remove(prev_action)

        return random.choice(available)
    
    def _strategy_follow_walls(self, grid, player_pos):
        """Strategy: Move along walls to explore the environment"""
        if grid.size == 0 or player_pos is None or None in player_pos:
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
        
        y, x = player_pos
        actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        wall_scores = {}
        for action in actions:
            next_y, next_x = self._get_next_position(y, x, action)
            if 0 <= next_y < grid.shape[0] and 0 <= next_x < grid.shape[1]:
                # Score higher if next to a wall
                score = 0
                if grid[next_y, next_x] == 10:
                    score += 2
                # Score higher if adjacent to a wall (not in the direction of movement)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    adj_y, adj_x = next_y + dy, next_x + dx
                    if 0 <= adj_y < grid.shape[0] and 0 <= adj_x < grid.shape[1]:
                        if grid[adj_y, adj_x] == 10:
                            score += 1
                wall_scores[action] = score
            else:
                wall_scores[action] = -1  # Penalize out of bounds
        max_score = max(wall_scores.values())
        best_actions = [a for a, s in wall_scores.items() if s == max_score]
        return random.choice(best_actions)

    def _strategy_target_nearest_object(self, grid, player_pos, obj_value):
        """
        Strategy: Move towards the nearest object of the given value (e.g., key or door).
        Args:
            grid: numpy array of the game grid
            player_pos: (y, x) tuple of player position
            obj_value: integer value representing the target object
        Returns:
            GameAction: The action to take
        """
        if grid.size == 0 or player_pos is None or None in player_pos:
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])

        obj_positions = np.argwhere(grid == obj_value)
        if obj_positions.shape[0] == 0:
            # No target objects found, fallback to random
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])

        # Find the closest object
        distances = [np.sum(np.abs(np.array(player_pos) - pos)) for pos in obj_positions]
        min_idx = int(np.argmin(distances))
        target_pos = obj_positions[min_idx]

        # Decide which direction to move to get closer
        dy = target_pos[0] - player_pos[0]
        dx = target_pos[1] - player_pos[1]

        # Prioritize vertical movement if farther vertically, else horizontal
        if abs(dy) > abs(dx):
            if dy < 0:
                return GameAction.ACTION1  # UP
            else:
                return GameAction.ACTION2  # DOWN
        elif abs(dx) > 0:
            if dx < 0:
                return GameAction.ACTION3  # LEFT
            else:
                return GameAction.ACTION4  # RIGHT
        else:
            # Already at target, pick random
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])

    def _detect_local_stuckness(self, state_id, metrics):
        """Detect if we're stuck in a local minimum"""
        # Use existing stagnation detection
        return self._detect_stagnation(state_id)

    def _try_alternative_action(self, state_id, frame):
        """Try an alternative action when stuck"""
        # For now, just use a random action not blacklisted
        available = self._get_available_actions(state_id)
        if not available:
            available = [GameAction.ACTION1, GameAction.ACTION2, 
                        GameAction.ACTION3, GameAction.ACTION4]
        return random.choice(available)

    def _log_high_level_state(self, metrics):
        """Log detailed high-level state information with supervision metrics"""
        # Skip logging if we don't have enough data
        if not hasattr(self, 'high_level_states') or not self.high_level_states:
            return
            
        # Get latest high-level state
        latest = self.high_level_states[-1]
        
        # Calculate key metrics
        progress_indicator = self._calculate_progress_indicator()
        feature_importance = self._calculate_feature_importance()
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Print comprehensive dashboard
        print("\n┌───────────────── HIGH-LEVEL CYCLE DASHBOARD ─────────────────┐")
        print(f"│ Cycle: {self.cycle_counter}")
        print(f"│ Strategy: {self.current_strategy}")
        print(f"│ Progress: {progress_indicator:.2f}")
        print(f"│ Q-value: {latest['Q']:.4f}")
        print(f"│ Stagnation: {latest['stagnation']}")
        print(f"│ Strategy Effectiveness: {self._evaluate_strategy_effectiveness():.2f}")
        print("│ Top Features: " + ", ".join(["{}: {:.2f}".format(k, v) for k, v in top_features]))
        print(f"│ Memory Size: {len(self.high_level_memory['performance_history'])}")
        
        # Add detected patterns
        if self.high_level_memory['global_patterns']:
            patterns = ", ".join(self.high_level_memory['global_patterns'].keys())
            print(f"│ Detected Patterns: {patterns}")
        
        print("└──────────────────────────────────────────────────────────────┘")
        
        # Log to standard logger as well
        logging.info(f"HIGH-LEVEL CYCLE {self.cycle_counter}: "
                    f"strategy={self.current_strategy}, "
                    f"progress={progress_indicator:.2f}, "
                    f"Q={latest['Q']:.4f}")
    
    def _evaluate_strategy_effectiveness(self, strategy=None):
        """Evaluate how effective different strategies have been"""
        if not hasattr(self, 'high_level_memory') or 'performance_history' not in self.high_level_memory:
            return 0.5  # Default value
    
        # Get performance history
        history = self.high_level_memory['performance_history']
        
        # Use current strategy if none specified
        if strategy is None and hasattr(self, 'current_strategy'):
            strategy = self.current_strategy
        
        # Calculate strategy effectiveness (simple version)
        relevant_entries = [entry for entry in history 
                           if entry.get('strategy') == strategy]
        
        if not relevant_entries:
            return 0.5  # Default value
            
        # Calculate based on Q improvement and low stagnation
        q_values = [entry.get('Q', 0) for entry in relevant_entries]
        stagnation_values = [entry.get('stagnation', 0) for entry in relevant_entries]
        
        if len(q_values) >= 2:
            q_trend = q_values[-1] - q_values[0]
            stag_penalty = stagnation_values[-1] / 10.0
            effectiveness = (q_trend + 0.5) - stag_penalty
            return min(1.0, max(0.0, effectiveness))
        
        return 0.5  # Default value

    def _calculate_progress_indicator(self):
        """
        Calculate a progress indicator for the agent.
        Returns a float between 0 and 1 based on Q-value improvement and stagnation.
        """
        if not hasattr(self, 'high_level_states') or len(self.high_level_states) < 2:
            return 0.0
        q_values = [state['Q'] for state in self.high_level_states]
        stagnations = [state['stagnation'] for state in self.high_level_states]
        q_trend = q_values[-1] - q_values[0]
        stagnation_trend = stagnations[-1] - stagnations[0]
        # Normalize progress: positive Q trend and low stagnation = high progress
        progress = np.clip(q_trend - 0.1 * stagnation_trend, 0.0, 1.0)
        return float(progress)

    def _log_deep_statistics(self, metrics):
        """Log deep statistics about agent performance and learning"""
        # Skip if we don't have enough data
        if not hasattr(self, 'high_level_states') or len(self.high_level_states) < 5:
            return
        
        # Calculate statistics
        cycles = len(self.high_level_states)
        avg_q = sum(state['Q'] for state in self.high_level_states) / cycles
        q_trend = self.high_level_states[-1]['Q'] - self.high_level_states[0]['Q']
        
        # Strategy effectiveness
        strategy_counts = {}
        for cycle, strategy in self.high_level_memory['strategy_history']:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
        best_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "None"
        
        # Exploration stats
        unique_states = len(self.visits) if hasattr(self, 'visits') else 0
        avg_visits = sum(self.visits.values()) / unique_states if unique_states > 0 else 0
        
        # Log the deep statistics
        logging.info("\n----- DEEP STATISTICS REPORT -----")
        logging.info(f"Total cycles: {cycles}")
        logging.info(f"Average Q-value: {avg_q:.4f}")
        logging.info(f"Q-value trend: {q_trend:.4f}")
        logging.info(f"Most used strategy: {best_strategy} ({strategy_counts.get(best_strategy, 0)} times)")
        logging.info(f"Unique states explored: {unique_states}")
        logging.info(f"Average visits per state: {avg_visits:.2f}")
        logging.info("-------------------------------")
        
        # Every 5 cycles, log deep statistics
        if self.cycle_counter % 5 == 0:
            # Also log top features by importance
            if hasattr(self, 'feature_importance') and self.feature_importance:
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logging.info("Top 5 important features:")
                for feature, importance in top_features:
                    logging.info(f"  - {feature}: {importance:.4f}")
    
    def _calculate_participation_ratio(self, vectors):
        """
        Calculate participation ratio from a set of vectors
        PR = (sum(λ))² / (n * sum(λ²))
        Where λ are eigenvalues of the covariance matrix
        """
        if not vectors or len(vectors) < 2:
            return 1.0  # Default value when we don't have enough data
            
        # Convert to numpy array
        X = np.array(vectors)
        
        # Center the data
        X = X - np.mean(X, axis=0)
        
        # Calculate covariance matrix
        n_samples = X.shape[0]
        if n_samples > 1:
            cov = np.cov(X, rowvar=False)
            
            # Get eigenvalues
            try:
                eigenvalues = np.linalg.eigvals(cov)
                eigenvalues = np.abs(eigenvalues)  # Ensure positive values
                
                # Calculate participation ratio
                sum_eigenvalues = np.sum(eigenvalues)
                if sum_eigenvalues > 0:
                    pr = (sum_eigenvalues**2) / (len(eigenvalues) * np.sum(eigenvalues**2))
                    return float(pr)
            except Exception:
                pass  # Fall back to default value if calculation fails
        
        return 1.0  # Default value
    
    def _update_participation_ratios(self):
        """Update participation ratios for both high and low level states"""
        # Calculate high-level PR from feature history
        if hasattr(self, 'feature_history') and len(self.feature_history) > 3:
            high_level_vectors = []
            feature_keys = sorted(set().union(*self.feature_history[-10:]))  # Get all keys
            
            for feature_dict in self.feature_history[-10:]:
                vector = [feature_dict.get(k, 0.0) for k in feature_keys]
                high_level_vectors.append(vector)
            
            self.high_level_pr = self._calculate_participation_ratio(high_level_vectors)
        else:
            self.high_level_pr = 1.0
    
        # Calculate low-level PR from recent features
        if hasattr(self, 'low_level_memory') and 'recent_features' in self.low_level_memory:
            recent_features = self.low_level_memory['recent_features']
            if len(recent_features) > 3:
                low_level_vectors = []
                feature_keys = sorted(set().union(*recent_features))  # Get all keys
                
                for feature_dict in recent_features:
                    vector = [feature_dict.get(k, 0.0) for k in feature_keys]
                    low_level_vectors.append(vector)
                
                self.low_level_pr = self._calculate_participation_ratio(low_level_vectors)
            else:
                self.low_level_pr = 1.0
        else:
            self.low_level_pr = 1.0
            
        # Log the participation ratios
        logging.info(f"Participation Ratios - High: {self.high_level_pr:.3f}, Low: {self.low_level_pr:.3f}")
    
    def _analyze_eigenvalue_spectrum(self, vectors, level='high'):
        """Analyze eigenvalue spectrum to detect dimensionality issues"""
        if not vectors or len(vectors) < 3:
            return {"status": "insufficient_data"}
            
        # Convert to numpy array
        X = np.array(vectors)
        
        # Calculate covariance matrix
        try:
            cov = np.cov(X, rowvar=False)
            eigenvalues = np.linalg.eigvals(cov)
            eigenvalues = np.abs(eigenvalues)  # Ensure positive values
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Calculate metrics
            total_variance = np.sum(eigenvalues)
            if total_variance <= 0:
                return {"status": "zero_variance"}
                
            # Normalize eigenvalues
            normalized = eigenvalues / total_variance
            
            # Calculate explained variance ratios
            explained_variance = np.cumsum(normalized)
            
            # Count effective dimensions (eigenvalues above threshold)
            threshold = 0.01  # Eigenvalues below 1% are considered noise
            effective_dims = np.sum(normalized > threshold)
            
            # Calculate dimensionality collapse ratio
            top_dims_ratio = np.sum(normalized[:2]) if len(normalized) >= 2 else 1.0
            
            # Determine status
            if top_dims_ratio > 0.9:
                status = "severe_collapse"  # >90% variance in top 2 dims
            elif top_dims_ratio > 0.8:
                status = "moderate_collapse"  # >80% variance in top 2 dims
            elif top_dims_ratio > 0.7:
                status = "mild_collapse"  # >70% variance in top 2 dims
            else:
                status = "healthy"  # Good dimensionality
                
            return {
                "status": status,
                "effective_dimensions": int(effective_dims),
                "total_dimensions": len(eigenvalues),
                "top_dims_ratio": float(top_dims_ratio),
                "eigenvalues": eigenvalues.tolist()[:5],  # First 5 eigenvalues
                "explained_variance": explained_variance.tolist()[:5]  # First 5 cumulative variances
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _monitor_dimensionality(self):
        """Monitor dimensionality of agent behavior and detect collapse"""
        # Check high-level dimensionality
        high_level_vectors = []
        
        if hasattr(self, 'feature_history') and len(self.feature_history) > 3:
            feature_keys = sorted(set().union(*self.feature_history[-10:]))
            for feature_dict in self.feature_history[-10:]:
                vector = [feature_dict.get(k, 0.0) for k in feature_keys]
                high_level_vectors.append(vector)
        
        high_level_analysis = self._analyze_eigenvalue_spectrum(high_level_vectors, 'high')
        
        # Check low-level dimensionality
        low_level_vectors = []
        
        if hasattr(self, 'low_level_memory') and 'recent_features' in self.low_level_memory:
            recent_features = self.low_level_memory['recent_features']
            if len(recent_features) > 3:
                feature_keys = sorted(set().union(*recent_features))
                for feature_dict in recent_features:
                    vector = [feature_dict.get(k, 0.0) for k in feature_keys]
                    low_level_vectors.append(vector)
        
        low_level_analysis = self._analyze_eigenvalue_spectrum(low_level_vectors, 'low')
        
        # Store analyses
        self.dimensionality_status = {
            'high_level': high_level_analysis,
            'low_level': low_level_analysis,
            'timestamp': self.action_counter
        }
        
        # Log detailed analysis periodically
        if self.action_counter % 20 == 0:
            self._log_dimensionality_analysis()
        
        # Check for severe collapse and take action if needed
        self._check_for_dimensionality_collapse()

    def _log_dimensionality_analysis(self):
        """Log detailed dimensionality analysis"""
        if not hasattr(self, 'dimensionality_status'):
            return
            
        analysis = self.dimensionality_status
        
        logging.info("\n===== DIMENSIONALITY ANALYSIS =====")
        
        # High-level analysis
        high_level = analysis.get('high_level', {})
        logging.info(f"HIGH-LEVEL STATUS: {high_level.get('status', 'unknown')}")
        if 'effective_dimensions' in high_level:
            logging.info(f"Effective dimensions: {high_level['effective_dimensions']}/{high_level['total_dimensions']}")
        if 'top_dims_ratio' in high_level:
            logging.info(f"Top 2 dimensions explain: {high_level['top_dims_ratio']:.1%} of variance")
        if 'eigenvalues' in high_level:
            logging.info(f"Top eigenvalues: {', '.join([f'{ev:.3f}' for ev in high_level['eigenvalues']])}")
        
        # Low-level analysis
        low_level = analysis.get('low_level', {})
        logging.info(f"LOW-LEVEL STATUS: {low_level.get('status', 'unknown')}")
        if 'effective_dimensions' in low_level:
            logging.info(f"Effective dimensions: {low_level['effective_dimensions']}/{low_level['total_dimensions']}")
        if 'top_dims_ratio' in low_level:
            logging.info(f"Top 2 dimensions explain: {low_level['top_dims_ratio']:.1%} of variance")
        
        logging.info("===================================")

    def _check_for_dimensionality_collapse(self):
        """Check for dimensionality collapse and respond if necessary"""
        if not hasattr(self, 'dimensionality_status'):
            return
        
        analysis = self.dimensionality_status
        high_level = analysis.get('high_level', {})
        low_level = analysis.get('low_level', {})
        
        # Check for severe collapse in high-level state
        if high_level.get('status') == 'severe_collapse':
            logging.warning("⚠️ DETECTED SEVERE HIGH-LEVEL DIMENSIONALITY COLLAPSE")
            logging.warning("Agent's strategic thinking has collapsed to low dimensionality")
            
            # Force strategy reset
            if hasattr(self, 'current_strategy'):
                logging.info("📊 Forcing strategy diversity to increase dimensionality")
                available_strategies = [
                    "explore_new_areas",
                    "target_nearest_key",
                    "target_nearest_door",
                    "follow_walls",
                    "return_to_start",
                    "random_walk"
                ]
                
                # Choose a completely different strategy
                self.current_strategy = self.rng.choice(available_strategies)
                logging.info(f"Switched to strategy: {self.current_strategy}")
                
                # Increase exploration
                self.dynamic_tau = min(1.0, self.dynamic_tau + 0.3)
        
        # Check for severe collapse in low-level state
        if low_level.get('status') == 'severe_collapse':
            logging.warning("⚠️ DETECTED SEVERE LOW-LEVEL DIMENSIONALITY COLLAPSE")
            logging.warning("Agent's tactical behavior has collapsed to low dimensionality")
            
            # Increase exploration radically
            self.dynamic_tau = 1.0
            logging.info("📊 Setting maximum exploration to break low-level dimensionality collapse")
            
            # Clear some blacklisted actions
            if hasattr(self, 'blacklisted_pairs') and self.blacklisted_pairs:
                # Clear up to 3 random blacklisted pairs
                clear_count = min(3, len(self.blacklisted_pairs))
                for _ in range(clear_count):
                    if self.blacklisted_pairs:
                        self.blacklisted_pairs.pop()
                
    def _calculate_info_gain_trend(self):
        """Calculate the trend of recent information gains."""
        if hasattr(self, 'recent_info_gains') and len(self.recent_info_gains) >= 5:
            recent = self.recent_info_gains[-5:]
            if all(x <= y for x, y in zip(recent, recent[1:])):
                return 0.8  # Consistently increasing
            elif all(x >= y for x, y in zip(recent, recent[1:])):
                return 0.2  # Consistently decreasing
            
            # Calculate slope
            x = list(range(len(recent)))
            y = recent
            n = len(x)
            if n > 1:
                slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                        (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
                return min(1.0, max(0.0, slope + 0.5))
    
        return 0.5  # Neutral if not enough data
    
    def _use_memory_for_decision(self, state_id, frame, features, metrics):
        """Use agent memory to influence decision making"""
        # Your implementation here
        # Example:
        if not hasattr(self, 'memory'):
            self.memory = {}
        
        # Store current features in memory
        self.memory[state_id] = features
        
        # Use memory to influence decision
        # ...
        
        return None  # or appropriate return value

    def _calculate_feature_importance(self):
        """Calculate importance of each feature for decision making"""
        importance = {}
    
        # Use feature history if available
        if hasattr(self, 'feature_history') and len(self.feature_history) >= 2:
            # Calculate changes in features across history
            for i in range(1, len(self.feature_history)):
                prev = self.feature_history[i-1]
                curr = self.feature_history[i]
                
                # Calculate importance based on feature changes
                for key in set(prev.keys()) & set(curr.keys()):
                    change = abs(curr[key] - prev[key])
                    importance[key] = importance.get(key, 0) + change
    
        # Normalize importance values
        if importance:
            max_val = max(importance.values())
            if max_val > 0:
                importance = {k: v/max_val for k, v in importance.items()}
            
        return importance

    def _track_feature_importance(self, features, metrics):
        """Track feature importance based on correlation with Q metrics"""
        if not hasattr(self, 'feature_importance'):
            self.feature_importance = {}
    
        # Update importance based on delta_q (from metrics)
        delta_q = metrics.get('delta_q', 0)
        for key, value in features.items():
            # Use absolute value of delta_q * feature value as importance
            importance = abs(delta_q * value)
            self.feature_importance[key] = self.feature_importance.get(key, 0) + importance

    def _calculate_q_head_confidence(self, metrics):
        """Calculate confidence based on Q-values and other metrics"""
        # Base confidence from Q stability
        q_value = metrics.get('Q', 0)
        stagnation = metrics.get('stagnation', 0)
        # Example: confidence increases with Q and decreases with stagnation
        confidence = 1.0 / (1.0 + np.exp(-q_value + 0.2 * stagnation))
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _detect_local_patterns(self, features=None, metrics=None):
        """Detect local patterns in agent's recent behavior"""
        # Initialize local pattern storage if needed
        if not hasattr(self, 'local_pattern_memory'):
            self.local_pattern_memory = {
                'recent_actions': [],
                'action_effects': {},
                'detected_patterns': []
            }
    
        # Check for local patterns in action history
        if hasattr(self, 'history') and len(self.history) >= 4:
            # Look for short repeating sequences
            for pattern_len in [2, 3]:
                if len(self.history) >= pattern_len * 2:
                    if self.history[-pattern_len:] == self.history[-(pattern_len*2):-pattern_len]:
                        self.local_pattern_memory['detected_patterns'].append({
                            'type': 'action_repetition',
                            'pattern': self.history[-pattern_len:],
                            'step': self.action_counter
                        })
    
        # Return detected patterns
        return self.local_pattern_memory['detected_patterns']
    
    def _implement_adaptive_halting(self, state_id, frame, features, metrics):
        """
        Implements adaptive halting criteria to determine when to stop current strategy.
        
        Args:
            state_id: Unique identifier for the current state
            frame: Current frame data
            features: Extracted features from the current state
            metrics: Performance metrics like Q values, stagnation, etc.
            
        Returns:
            bool: True if should halt current strategy, False otherwise
        """
        # Default thresholds
        stagnation_threshold = 5
        q_threshold = 0.8
        no_improvement_threshold = 3
        
        # Get current metrics
        stagnation = metrics.get('stagnation', 0)
        q_value = metrics.get('Q', 0)
        
        # Track improvement
        if not hasattr(self, 'last_q_values'):
            self.last_q_values = []
        
        self.last_q_values.append(q_value)
        if len(self.last_q_values) > no_improvement_threshold:
            self.last_q_values.pop(0)
        
        # Check for no improvement over last few steps
        no_improvement = False
        if len(self.last_q_values) >= no_improvement_threshold:
            # Check if Q values are flat or decreasing
            if all(x >= y for x, y in zip(self.last_q_values, self.last_q_values[1:])):
                no_improvement = True
        
        # Decide whether to halt
        should_halt = (
            stagnation >= stagnation_threshold or
            q_value >= q_threshold or
            no_improvement
        )
        
        return should_halt

    def _log_grid_state(self, grid, frame, description=""):
        """
        Log the grid state for debugging and analysis.
        
        Args:
            grid: The game grid as numpy array
            frame: The current frame data
            description: Optional description of the state
        """
        # Initialize log storage if not exists
        if not hasattr(self, 'grid_log'):
            self.grid_log = []
        
        # Store grid state with metadata
        self.grid_log.append({
            'step': self.action_counter if hasattr(self, 'action_counter') else 0,
            'grid': grid.copy() if hasattr(grid, 'copy') else grid,
            'description': description,
            'timestamp': getattr(self, 'frame_timestamp', None)
        })
        
        # Optional: Print simplified grid representation if verbose mode
        if hasattr(self, 'verbose') and self.verbose:
            print(f"\n--- {description} ---")
            rows, cols = grid.shape if hasattr(grid, 'shape') else (0, 0)
            for r in range(rows):
                print(" ".join(str(grid[r, c]).rjust(2) for c in range(cols)))

    def _calculate_symmetry(self, grid):
        """
        Calculate horizontal, vertical, and rotational symmetry scores for a grid.
        
        Args:
            grid: numpy array representing the game grid
            
        Returns:
            dict: Symmetry scores (0.0-1.0, higher = more symmetrical)
        """
        if grid is None or not hasattr(grid, 'shape') or grid.size == 0:
            return {'horizontal': 0.0, 'vertical': 0.0, 'rotational': 0.0}
            
        # Calculate horizontal symmetry
        h_sym = 0
        rows, cols = grid.shape
        for i in range(rows):
            matches = sum(grid[i, j] == grid[i, cols-j-1] for j in range(cols//2))
            h_sym += matches / (cols//2) if cols > 1 else 1.0
        h_sym /= rows if rows > 0 else 1.0
            
        # Calculate vertical symmetry
        v_sym = 0
        for j in range(cols):
            matches = sum(grid[i, j] == grid[rows-i-1, j] for i in range(rows//2))
            v_sym += matches / (rows//2) if rows > 1 else 1.0
        v_sym /= cols if cols > 0 else 1.0
        
        # Calculate rotational symmetry (180° rotation)
        r_sym = 0
        total = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == grid[rows-i-1, cols-j-1]:
                    r_sym += 1
                total += 1
        r_sym = r_sym / total if total > 0 else 0.0
            
        return {
            'horizontal': float(h_sym), 
            'vertical': float(v_sym),
            'rotational': float(r_sym)
        }
    
    def _calculate_connectivity(self, grid):
        """
        Calculate grid connectivity - how connected similar elements are.
        Higher connectivity means similar values tend to be adjacent.
        
        Args:
            grid: numpy array of the game grid
            
        Returns:
            float: Connectivity score between 0.0-1.0
        """
        if grid is None or not hasattr(grid, 'shape') or grid.size == 0:
            return 0.0
            
        rows, cols = grid.shape
        if rows <= 1 or cols <= 1:
            return 1.0  # Single row/col is maximally connected
            
        # Count matching neighbors
        matches = 0
        total = 0
        
        # Check horizontal adjacency
        for i in range(rows):
            for j in range(cols-1):
                total += 1
                if grid[i, j] == grid[i, j+1]:
                    matches += 1
                    
        # Check vertical adjacency
        for i in range(rows-1):
            for j in range(cols):
                total += 1
                if grid[i, j] == grid[i+1, j]:
                    matches += 1
                    
        # Return normalized connectivity score
        return float(matches / total if total > 0 else 0.0)
    
    def _strategy_explore_new_areas(self, state_id, frame, features):
        """
        Strategy: Explore areas of the grid that haven't been visited yet.
        
        Args:
            state_id: Current state identifier
            frame: Current frame data
            features: Extracted features
            
        Returns:
            GameAction: The action to take
        """
        # Get player position and grid
        grid = self._get_grid_array(frame)
        player_pos = self._find_player_position(grid)
        
        if not player_pos or None in player_pos:
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
        
        # Calculate "unexploredness" in each direction
        directions = [
            (GameAction.ACTION1, -1, 0),  # Up
            (GameAction.ACTION2, 1, 0),   # Down
            (GameAction.ACTION3, 0, -1),  # Left
            (GameAction.ACTION4, 0, 1),   # Right
        ]
        
        # Initialize scores for each direction
        direction_scores = {}
        
        for action, dy, dx in directions:
            score = 0
            y, x = player_pos
            
            # Look ahead a few steps in this direction
            for steps in range(1, 4):
                # Ensure y and x are not None before arithmetic
                if y is None or x is None:
                    continue  # Skip this direction if position is invalid
                ny, nx = y + dy * steps, x + dx * steps
                
                # Check if position is valid
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    # Higher score for positions visited less often
                    pos = (int(ny), int(nx))
                    visits = self.visited_cells.get(pos, 0) if hasattr(self, 'visited_cells') else 0
                    score += 4 - min(visits, 3)  # 4 for never visited, 1 for visited 3+ times
                else:
                    score -= 5  # Penalty for going out of bounds
            
            direction_scores[action] = score
    
        # Choose direction with highest score, break ties randomly
        max_score = max(direction_scores.values()) if direction_scores else 0
        best_actions = [a for a, s in direction_scores.items() if s == max_score]
        return random.choice(best_actions) if best_actions else GameAction.ACTION1
    
    def _strategy_return_to_start(self, state_id, frame, features):
        """
        Strategy: Navigate back to the starting position.
        
        Args:
            state_id: Current state identifier
            frame: Current frame data
            features: Extracted features
            
        Returns:
            GameAction: The action to take
        """
        # Get grid and positions
        grid = self._get_grid_array(frame)
        player_pos = self._find_player_position(grid)
        
        # Use class attribute for start position, or center if not defined
        if not hasattr(self, 'start_pos') or self.start_pos is None:
            rows, cols = grid.shape if hasattr(grid, 'shape') else (0, 0)
            self.start_pos = (rows // 2, cols // 2)  # Default to center
    
        # If we can't determine positions, resort to random
        if player_pos is None or None in player_pos:
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, 
                                 GameAction.ACTION3, GameAction.ACTION4])
        
        # Calculate direction to move (simple approach)
        if (
            player_pos is not None and
            isinstance(player_pos, tuple) and
            len(player_pos) == 2 and
            player_pos[0] is not None and
            player_pos[1] is not None
        ):
            y_diff = self.start_pos[0] - player_pos[0]
            x_diff = self.start_pos[1] - player_pos[1]
        
            # Prioritize the largest difference
            if abs(y_diff) >= abs(x_diff):
                # Move vertically
                if y_diff > 0:
                    return GameAction.ACTION1  # Up
                elif y_diff < 0:
                    return GameAction.ACTION2  # Down
            else:
                # Move horizontally
                if x_diff < 0:
                    return GameAction.ACTION3  # Left
                elif x_diff > 0:
                    return GameAction.ACTION4  # Right
        
            # We're at the start position
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
        else:
            # If player_pos is invalid, fallback to random action
            return random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])


