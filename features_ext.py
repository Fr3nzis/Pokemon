import pandas as pd
import numpy as np
from tqdm import tqdm
from pk_functions import get_pokemon_types, is_super_effective, momentum_score,damage_features, switch_difference,get_effectiveness, calculate_net_coverage
from dicts import types, STATUS_PENALTIES, pokemon_types

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
            priority_1 = priority_2 = 0
            n_null_moves_p1 = n_null_moves_p2 = 0
            p1_super_effective_taken = p2_lead_super_effective_taken = 0
            cumulative_boost_magnitude_diff = 0
            p1_fainted_count = p2_fainted_count = 0
            total_status_advantage = 0
            p1_cumulative_effectiveness = p2_cumulative_effectiveness = 0
            diff_speed = 0
            p1_stab_hits = p2_stab_hits = 0
            p1_x4_hits = p2_x4_hits = 0
            p1_x2_hits = p2_x2_hits = 0
            p1_x0_5_hits = p2_x0_5_hits = 0 
            p1_x0_25_hits = p2_x0_25_hits = 0 
            p2_known_names = set()
            
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
                    priority_1 += int(p1_details.get("priority", 0))
                else:
                    n_null_moves_p1 += 1

                if p2_details:
                    accuracy_2 += int(p2_details.get("accuracy", 0))
                    base_power_2 += int(p2_details.get("base_power", 0))
                    priority_2 += int(p2_details.get("priority", 0))
                else:
                    n_null_moves_p2 += 1
                '''
                # EFFECTIVENESS SCORE (CUMULATIVE)
                if p1_details:
                     p1_move_type = p1_details.get("type", "").lower()
                     
                     # 2. Stabiliamo se la mossa fa danno
                     p1_category = p1_details.get("category", "").upper()
                     is_damaging_p1 = (p1_category == "PHYSICAL" or p1_category == "SPECIAL")

                     if p1_move_type and p2_name:
                        # 3. Chiamata corretta alla funzione
                        eff_p1 = get_effectiveness(
                            p1_move_type,
                            p2_types,
                            attacker_types=p1_types,  # Passa i tipi di P1 per lo STAB
                            is_damaging=is_damaging_p1 # Passa il flag di danno
                        )
                        p1_cumulative_effectiveness += eff_p1

                if p2_details:
                     p2_move_type = p2_details.get("type", "").lower()

                     # 2. Stabiliamo se la mossa fa danno
                     p2_category = p2_details.get("category", "").upper()
                     is_damaging_p2 = (p2_category == "PHYSICAL" or p2_category == "SPECIAL")
                     
                     if p2_move_type and p1_name:
                        # 3. Chiamata corretta alla funzione
                        eff_p2 = get_effectiveness(
                            p2_move_type,
                            p1_types,
                            attacker_types=p2_types,  # Passa i tipi di P2 per lo STAB
                            is_damaging=is_damaging_p2 # Passa il flag di danno
                        )
                        p2_cumulative_effectiveness += eff_p2'''
                 
                 # --- CONTEGGI P1 (MOSSE, EFFICACIA, STAB, ACCURACY, BASE POWER) ---
                if p1_details and p1_details.get("accuracy") is not None:
                    move_type_p1 = p1_details.get("type", "").lower()
                    
                    # STAB
                    if move_type_p1 in p1_types: p1_stab_hits += 1
                    
                    # Efficacia Dettagliata
                    eff_p1 = get_effectiveness(move_type_p1, p2_types)
                    if eff_p1 == 4: p1_x4_hits += 1
                    elif eff_p1 == 2: p1_x2_hits += 1
                    elif eff_p1 == 0.5: p1_x0_5_hits += 1
                    elif eff_p1 == 0.25: p1_x0_25_hits += 1
                
                # --- CONTEGGI P2 (MOSSE, EFFICACIA, STAB, ACCURACY, BASE POWER) ---
                if p2_details and p2_details.get("accuracy") is not None:
                    move_type_p2 = p2_details.get("type", "").lower()
                    
                    # STAB
                    if move_type_p2 in p2_types: p2_stab_hits += 1
                        
                    # Efficacia Dettagliata
                    eff_p2 = get_effectiveness(move_type_p2, p1_types)
                    if eff_p2 == 4: p2_x4_hits += 1
                    elif eff_p2 == 2: p2_x2_hits += 1
                    elif eff_p2 == 0.5: p2_x0_5_hits += 1
                    elif eff_p2 == 0.25: p2_x0_25_hits += 1
                        
                
                # --- FINE BLOCCO CORRETTO ---
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
                diff_speed +=  (p1_boosts.get('spe', 0) - p2_boosts.get('spe', 0)) 

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
            diff_prio = priority_1 - priority_2
            norm_null_p1 = n_null_moves_p1 / ntimeline if ntimeline else 0
            norm_null_p2 = n_null_moves_p2 / ntimeline if ntimeline else 0
            diff_null_moves = norm_null_p2 - norm_null_p1
            mean_boost_magnitude_diff = cumulative_boost_magnitude_diff / ntimeline if ntimeline else 0
            diff_eff = p1_cumulative_effectiveness - p2_cumulative_effectiveness
            stab_diff = (p1_stab_hits - p2_stab_hits) / ntimeline if ntimeline else 0
            x4_eff_diff = (p1_x4_hits - p2_x4_hits) / ntimeline if ntimeline else 0
            x2_eff_diff = (p1_x2_hits - p2_x2_hits) / ntimeline if ntimeline else 0
            x0_5_eff_diff = (p1_x0_5_hits - p2_x0_5_hits) / ntimeline if ntimeline else 0
            x0_25_eff_diff = (p1_x0_25_hits - p2_x0_25_hits) / ntimeline if ntimeline else 0

            net_coverage_advantage = calculate_net_coverage(p1_team, p2_known_names)
            
            
            # AGGIUNTA FEATURE FINALI
            features.update({
                'diff_eff' : diff_eff,
                'total_status_advantage': total_status_advantage,
                'diff_accuracy': diff_accuracy,
                'diff_base_power': diff_base_power,
                #'p1_type_coverage': p1_coverage,
                'net_coverage_advantage': net_coverage_advantage,
                #'p1_super_effective_taken': p1_super_effective_taken,
                #'p2_lead_super_effective_taken': p2_lead_super_effective_taken,
                'diff_null_moves': diff_null_moves,
                'mean_boost_magnitude_diff': mean_boost_magnitude_diff,
                'fainted_diff': p1_fainted_count - p2_fainted_count,
                'p1_momentum_score': momentum_score(battle),
                'switch_diff' : switch_difference(battle),
                'diff_speed_boost': diff_speed,
                #'diff_prio':diff_prio,
                #'stab_diff': stab_diff,
                'x4_eff_diff': x4_eff_diff,
                'x2_eff_diff': x2_eff_diff,
                'x0_5_eff_diff': x0_5_eff_diff,
                'x0_25_eff_diff': x0_25_eff_diff,
                #'p1_accuracy_p2_super_effective_taken':accuracy_1*p2_lead_super_effective_taken,
                'p2_accuracy_p1_super_effective_taken':accuracy_2*p1_super_effective_taken,
                #'fainted/switch': (p1_fainted_count - p2_fainted_count)/(switch_difference(battle)+1e-7)
                #'status_adv_e_diff_base_power':(total_status_advantage)*diff_base_power,
                #'p1_base_power_p2_super_effective_taken':base_power_1*p2_lead_super_effective_taken,
                #'p2_base_power_p1_super_effective_taken':base_power_2*p1_super_effective_taken,
                'tot_stat_adv_p2_lead_sup_eff_taken':total_status_advantage*p2_lead_super_effective_taken
             })
                  

            # DAMAGE FEATURES
            features.update(damage_features(battle))

            # ID e target
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)