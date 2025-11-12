
import pandas as pd
import numpy as np
from tqdm import tqdm
from pk_functions import (
    damage_features, switch_difference, get_effectiveness, 
    diff_coverage_advantage
)
from dicts import status_penalties, pokemon_types


class FeatureHandler:
    def __init__(self, train_data, test_data=None):
        self.train_data = train_data
        self.test_data = test_data

    def create_advanced_features(self, data):
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
            if p2_lead:
                features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
                features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
                features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
                features['p2_lead_def'] = p2_lead.get('base_def', 0)
                features['p2_lead_spa'] = p2_lead.get('base_spa', 0)
                features['p2_lead_spd'] = p2_lead.get('base_spd', 0)
            

            timeline = battle.get('battle_timeline', [])
            ntimeline = len(timeline)
             
            #inizializzazione variabili
            accuracy_1 = accuracy_2 = 0
            base_power_1 = base_power_2 = 0
            p1_null_moves = p2_null_moves = 0
            diff_boosts_score = 0
            diff_status_penalties = 0
            p1_stab = p2_stab= 0
            p1_x4_hits = p2_x4_hits = 0
            p1_x2_hits = p2_x2_hits = 0
            p1_x0_5_hits = p2_x0_5_hits = 0
            p1_x0_25_hits = p2_x0_25_hits = 0
            p2_known_names = set()
            p1_first_ko_turn = None
            p2_first_ko_turn = None
            p1_team_state = {}
            p2_team_state = {}
            turn_counter = 1

            #esploro i turni per battaglia e creo features
            for turn in timeline:
                p1_state = turn.get("p1_pokemon_state", {})
                p2_state = turn.get("p2_pokemon_state", {})
                p1_boosts = p1_state.get("boosts", {})
                p2_boosts = p2_state.get("boosts", {})
                p1_details = turn.get("p1_move_details", {})
                p2_details = turn.get("p2_move_details", {})
                p1_status = p1_state.get("status", "nostatus")
                p2_status = p2_state.get("status", "nostatus")
                p1_name = p1_state.get("name")
                p2_name = p2_state.get("name")
                p1_types = pokemon_types.get(p1_name.lower() if p1_name else None, [])
                p2_types = pokemon_types.get(p2_name.lower() if p2_name else None, [])
                if p2_name:
                    p2_known_names.add(p2_name)

                # inizializza hp/status
                if p1_name and p1_name not in p1_team_state:
                    p1_team_state[p1_name] = {'hp_pct': 1.0, 'status': 'nostatus'}
                if p2_name and p2_name not in p2_team_state:
                    p2_team_state[p2_name] = {'hp_pct': 1.0, 'status': 'nostatus'}
                if p1_name and p1_state.get('hp_pct') is not None:
                    p1_team_state[p1_name]['hp_pct'] = p1_state.get('hp_pct')
                if p2_name and p2_state.get('hp_pct') is not None:
                    p2_team_state[p2_name]['hp_pct'] = p2_state.get('hp_pct')
                if p1_name:
                    p1_team_state[p1_name]['status'] = p1_status
                if p2_name:
                    p2_team_state[p2_name]['status'] = p2_status

                #FIRST KO
                if p1_status == 'fnt' or p1_state.get('hp_pct') == 0:
                    if p1_first_ko_turn is None:
                        p1_first_ko_turn = turn_counter
                if p2_status == 'fnt' or p2_state.get('hp_pct') == 0:
                    if p2_first_ko_turn is None:
                        p2_first_ko_turn = turn_counter
                turn_counter += 1

                # DIFF STATUS PENALTIES
                p1_turn_status_penalties = status_penalties.get(p1_status, 0)
                p2_turn_status_penalties = status_penalties.get(p2_status, 0)
                diff_status_penalties_turn = p1_turn_status_penalties - p2_turn_status_penalties
                diff_status_penalties += diff_status_penalties_turn


                # ACCURACY / BASE POWER / NULL MOVES
                if p1_details:
                    accuracy_1 += int(p1_details.get("accuracy", 0))
                    base_power_1 += int(p1_details.get("base_power", 0))
                else:
                    p1_null_moves += 1

                if p2_details:
                    accuracy_2 += int(p2_details.get("accuracy", 0))
                    base_power_2 += int(p2_details.get("base_power", 0))
                else:
                    p2_null_moves += 1

                # P1 STAB E EFFECTIVENESS
                if p1_details and p1_details.get("accuracy") is not None:
                    p1_move_type = p1_details.get("type", "").lower()
                    if p1_move_type in p1_types:
                        p1_stab += 1
                    p1_effectiveness = get_effectiveness(p1_move_type, p2_types)
                    if p1_effectiveness == 4:
                        p1_x4_hits += 1
                    elif p1_effectiveness == 2:
                        p1_x2_hits += 1
                    elif p1_effectiveness == 0.5:
                        p1_x0_5_hits += 1
                    elif p1_effectiveness == 0.25:
                        p1_x0_25_hits += 1

                # P2 STAB E EFFECTIVENESS
                if p2_details and p2_details.get("accuracy") is not None:
                    p2_move_type = p2_details.get("type", "").lower()
                    if p2_move_type in p2_types:
                        p2_stab += 1
                    p2_effectiveness = get_effectiveness(p2_move_type, p1_types)
                    if p2_effectiveness == 4:
                        p2_x4_hits += 1
                    elif p2_effectiveness == 2:
                        p2_x2_hits += 1
                    elif p2_effectiveness == 0.5:
                        p2_x0_5_hits += 1
                    elif p2_effectiveness == 0.25:
                        p2_x0_25_hits += 1


                # DIFF BOOSTS SCORE
                p1_boosts_score = (p1_boosts.get('atk', 0) - p2_boosts.get('def', 0)) + (p1_boosts.get('spa', 0) - p2_boosts.get('spd', 0))
                p2_boosts_score = (p2_boosts.get('atk', 0) - p1_boosts.get('def', 0)) + (p2_boosts.get('spa', 0) - p1_boosts.get('spd', 0))
                diff_boosts_score += (p1_boosts_score - p2_boosts_score)

            #features
            diff_accuracy = accuracy_1 - accuracy_2
            diff_base_power = base_power_1 - base_power_2
            p1_norm_null_moves = p1_null_moves / ntimeline if ntimeline else 0
            p2_norm_null_moves = p2_null_moves / ntimeline if ntimeline else 0
            diff_null_moves = p2_norm_null_moves - p1_norm_null_moves
            diff_boosts_score = diff_boosts_score / ntimeline if ntimeline else 0
            diff_stab = (p1_stab - p2_stab) / ntimeline if ntimeline else 0
            diff_x4_eff = (p1_x4_hits - p2_x4_hits) / ntimeline if ntimeline else 0
            diff_x2_eff = (p1_x2_hits - p2_x2_hits) / ntimeline if ntimeline else 0
            diff_x0_5_eff = (p1_x0_5_hits - p2_x0_5_hits) / ntimeline if ntimeline else 0
            diff_x0_25_eff = (p1_x0_25_hits - p2_x0_25_hits) / ntimeline if ntimeline else 0
            p1_first_ko = p1_first_ko_turn if p1_first_ko_turn is not None else ntimeline + 1
            p2_first_ko = p2_first_ko_turn if p2_first_ko_turn is not None else ntimeline + 1
            p1_alive = sum(1 for i in p1_team_state.values() if i['status'] != 'fnt')
            p2_alive = sum(1 for i in p2_team_state.values() if i['status'] != 'fnt')
            p1_total_hp = sum(max(0, i.get('hp_pct', 0)) for i in p1_team_state.values())
            p2_total_hp = sum(max(0, i.get('hp_pct', 0)) for i in p2_team_state.values())
            p1_fainted_final = len(p1_team_state) - p1_alive
            p2_fainted_final = len(p2_team_state) - p2_alive
            


            #aggiunta features
            features.update({
                'diff_coverage_advantage': diff_coverage_advantage(p1_team, p2_known_names), #da controllare
                'diff_status_penalties': diff_status_penalties, #
                'diff_accuracy': diff_accuracy,#
                'diff_base_power': diff_base_power,#
                'diff_null_moves': diff_null_moves,#
                'diff_boosts_score': diff_boosts_score,#
                'switch_diff': switch_difference(battle),#
                'diff_stab': diff_stab,#
                'diff_x4_eff': diff_x4_eff,#
                'diff_x2_eff': diff_x2_eff,#
                'diff_x0_5_eff': diff_x0_5_eff,#
                'diff_x0_25_eff': diff_x0_25_eff,#
                'p1_first_ko': p1_first_ko,
                'p2_first_ko': p2_first_ko,
                #'diff_ko':p1_first_ko-p2_first_ko,#
                'p1_final_alive': p1_alive,
                'p2_final_alive': p2_alive,
                #'diff_alive':p1_alive-p2_alive,
                'p1_final_fainted': p1_fainted_final,
                'p2_final_fainted': p2_fainted_final,
                #'diff_fainted':p1_fainted_final-p2_fainted_final,
                'p1_final_hp_sum': p1_total_hp,
                'p2_final_hp_sum': p2_total_hp,
                #'diff_final_hp':p1_total_hp-p2_total_hp,
                **damage_features(battle)
            })


            # ID e target
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)