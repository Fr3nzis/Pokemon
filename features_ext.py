import pandas as pd
import numpy as np
from tqdm import tqdm
from pk_functions import get_pokemon_types, is_super_effective, momentum_score,damage_features, switch_difference
class FeatureHandler:
    def __init__(self, train_data, test_data=None):
        self.train_data = train_data
        self.test_data = test_data

    def create_advanced_features(self, data):
        STATUS_PENALTIES = {
            "frz": -100, # Congelamento: quasi un K.O.
            "slp": -75,  # Sonno: disabilita completamente
            "par": -40,  # Paralisi: penalità media statica
            "tox": -25,  # Iperavvelenamento: danno crescente
            "brn": -15,  # Scottatura
            "psn": -15,  # Avvelenamento
            "nostatus": 0}
        types = {
            "fire": ["grass", "ice", "bug", "steel"],
            "water": ["fire", "ground", "rock"],
            "grass": ["water", "ground", "rock"],
            "electric": ["water", "flying"],
            "ice": ["grass", "ground", "flying", "dragon"],
            "fighting": ["normal", "ice", "rock", "dark", "steel"],
            "poison": ["grass", "fairy"],
            "ground": ["electric", "fire", "poison", "rock", "steel"],
            "flying": ["fighting", "bug", "grass"],
            "psychic": ["fighting", "poison"],
            "bug": ["grass", "psychic", "dark"],
            "rock": ["fire", "ice", "flying", "bug"],
            "ghost": ["psychic", "ghost"],
            "dragon": ["dragon"],
            "steel": ["ice", "rock", "fairy"],
            "dark": ["psychic", "ghost"],
            "fairy": ["dark", "fighting", "dragon"],
        }

        feature_list = []
        for battle in tqdm(data, desc="Extracting features"):
            features = {}

            # --- Team Player 1 ---
            p1_team = battle.get('p1_team_details', [])
            if p1_team:
                features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
                features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
                features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
                features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
                features['p1_mean_spa'] = np.mean([p.get('base_spa', 0) for p in p1_team])
                features['p1_mean_spd'] = np.mean([p.get('base_spd', 0) for p in p1_team])

            # --- Lead P2 ---
            p2_lead = battle.get('p2_lead_details')
            p2_lead_types = [t.lower() for t in p2_lead.get('types', []) if t and t.lower() != 'notype'] if p2_lead else []
            if p2_lead:
                features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
                features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
                features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
                features['p2_lead_def'] = p2_lead.get('base_def', 0)
                features['p2_lead_spa'] = p2_lead.get('base_spa', 0)
                features['p2_lead_spd'] = p2_lead.get('base_spd', 0)

            # --- Variabili dinamiche ---
            timeline = battle.get('battle_timeline', [])
            ntimeline = len(timeline)

            accuracy_1 = accuracy_2 = 0
            base_power_1 = base_power_2 = 0
            n_null_moves_p1 = n_null_moves_p2 = 0
            p1_super_effective_taken = p2_lead_super_effective_taken = 0
            cumulative_boost_magnitude_diff = 0
            p1_fainted_count = p2_fainted_count = 0
            total_status_advantage = 0

            for turn in timeline:
                p1_state = turn.get("p1_pokemon_state", {})
                p2_state = turn.get("p2_pokemon_state", {})
                p1_boosts = p1_state.get("boosts", {})
                p2_boosts = p2_state.get("boosts", {})
                p1_details = turn.get("p1_move_details", {})
                p2_details = turn.get("p2_move_details", {})
                p1_status = p1_state.get("status", "nostatus")
                p2_status = p2_state.get("status", "nostatus")

            
                # STATUS
                p1_turn_handicap = STATUS_PENALTIES.get(p1_status, 0)
                p2_turn_handicap = STATUS_PENALTIES.get(p2_status, 0)
                net_status_advantage_turn = p1_turn_handicap - p2_turn_handicap
                total_status_advantage += net_status_advantage_turn

                if p1_status == 'fnt':
                    p1_fainted_count += 1
                if p2_status == 'fnt':
                    p2_fainted_count += 1

                # ACCURACY / BASE POWER / NULL MOVES
                if p1_details:
                    accuracy_1 += int(p1_details.get("accuracy", 0))
                    base_power_1 += int(p1_details.get("base_power", 0))
                else:
                    n_null_moves_p1 += 1

                if p2_details:
                    accuracy_2 += int(p2_details.get("accuracy", 0))
                    base_power_2 += int(p2_details.get("base_power", 0))
                else:
                    n_null_moves_p2 += 1

                # SUPER EFFECTIVE CHECK
                if p2_details and p1_state.get('name'):
                    p2_move_type = p2_details.get("type", "").lower()
                    p1_types = get_pokemon_types(battle, p1_state.get("name"))
                    if is_super_effective(p2_move_type, p1_types, types):
                        p1_super_effective_taken += 1

                if p1_details:
                    p1_move_type = p1_details.get("type", "").lower()
                    if is_super_effective(p1_move_type, p2_lead_types, types):
                        p2_lead_super_effective_taken += 1

                # BOOST MAGNITUDE
                p1_score = (p1_boosts.get('atk', 0) - p2_boosts.get('def', 0)) + \
                           (p1_boosts.get('spa', 0) - p2_boosts.get('spd', 0))
                p2_score = (p2_boosts.get('atk', 0) - p1_boosts.get('def', 0)) + \
                           (p2_boosts.get('spa', 0) - p1_boosts.get('spd', 0))
                cumulative_boost_magnitude_diff += (p1_score - p2_score)

            # COVERAGE
            covered_types = set()
            for pokemon in p1_team:
                for p1_type in pokemon.get('types', []):
                    p1_type_lower = p1_type.lower()
                    if p1_type_lower in types:
                        covered_types.update(types[p1_type_lower])
            p1_coverage = len(covered_types)

            # NORMALIZZAZIONI
            diff_accuracy = accuracy_1 - accuracy_2
            diff_base_power = base_power_1 - base_power_2
            norm_null_p1 = n_null_moves_p1 / ntimeline if ntimeline else 0
            norm_null_p2 = n_null_moves_p2 / ntimeline if ntimeline else 0
            diff_null_moves = norm_null_p2 - norm_null_p1
            mean_boost_magnitude_diff = cumulative_boost_magnitude_diff / ntimeline if ntimeline else 0

            # AGGIUNTA FEATURE FINALI
            features.update({
                'total_status_advantage': total_status_advantage,
                'diff_accuracy': diff_accuracy,
                'diff_base_power': diff_base_power,
                'p1_type_coverage': p1_coverage,
                'p1_super_effective_taken': p1_super_effective_taken,
                'p2_lead_super_effective_taken': p2_lead_super_effective_taken,
                'diff_null_moves': diff_null_moves,
                'mean_boost_magnitude_diff': mean_boost_magnitude_diff,
                'fainted_diff': p1_fainted_count - p2_fainted_count,
                'p1_momentum_score': momentum_score(battle),
                'switch_diff' : switch_difference(battle)
                  })

            # DAMAGE FEATURES
            features.update(damage_features(battle))

            # ID e target
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)