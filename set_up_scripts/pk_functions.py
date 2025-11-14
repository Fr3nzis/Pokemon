from dicts import gen1_type


def damage_features(battle: dict) -> dict: 
    """
    Estrae feature legate al danno netto e al rapporto danni inflitti/subiti per P1.
    """
    timeline = battle.get('battle_timeline', [])
    p1_net_damage = 0
    p1_last_hp_pct = 1
    p2_last_hp_pct = 1
    p1_total_damage_received = 0
    p1_total_damage_inflicted = 0

    p1_active_pokemon = battle.get('p1_team_details', [{}])[0].get('name') if battle.get('p1_team_details') else ''
    p2_active_pokemon = battle.get('p2_lead_details', {}).get('name', '')

    for turn_data in timeline:
        p1_state = turn_data.get("p1_pokemon_state", {})
        p2_state = turn_data.get("p2_pokemon_state", {})

        p1_current_hp_pct = p1_state.get('hp_pct', 0)
        p2_current_hp_pct = p2_state.get('hp_pct', 0)
        p1_current_name = p1_state.get('name')
        p2_current_name = p2_state.get('name')

        if p1_current_name != p1_active_pokemon:
            p1_last_hp_pct = p1_current_hp_pct
            p1_active_pokemon = p1_current_name
        if p2_current_name != p2_active_pokemon:
            p2_last_hp_pct = p2_current_hp_pct
            p2_active_pokemon = p2_current_name

        p1_damage_inflicted = max(0, p2_last_hp_pct - p2_current_hp_pct)
        p1_damage_received = max(0, p1_last_hp_pct - p1_current_hp_pct)

        p1_net_damage += (p1_damage_inflicted - p1_damage_received)
        p1_total_damage_inflicted += p1_damage_inflicted
        p1_total_damage_received += p1_damage_received

        p1_last_hp_pct = p1_current_hp_pct
        p2_last_hp_pct = p2_current_hp_pct


    if p1_total_damage_received < 1e-7:
        p1_damage_ratio = p1_total_damage_inflicted
    else:
        p1_damage_ratio = p1_total_damage_inflicted / p1_total_damage_received

    return {
        'p1_net_damage': p1_net_damage,
        'p1_damage_ratio': p1_damage_ratio
    }

def switch_difference(battle: dict) -> int:
    """
    Restituisce la differenza (p1_switches - p2_switches) contando solo i switch volontari
    (cioè cambi non dovuti a faint).
    """
    timeline = battle.get('battle_timeline', [])

    p1_active_pokemon = battle.get('p1_team_details', [{}])[0].get('name') if battle.get('p1_team_details') else None
    p2_active_pokemon = battle.get('p2_lead_details', {}).get('name', None)

    p1_last_pokemon = {'name': p1_active_pokemon, 'hp_pct': 1.0, 'status': None}
    p2_last_pokemon = {'name': p2_active_pokemon, 'hp_pct': 1.0, 'status': None}

    p1_switches = 0
    p2_switches = 0

    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})

        p1_current_name = p1_state.get('name')
        p2_current_name= p2_state.get('name')

        p1_hp_pct = p1_state.get('hp_pct', p1_last_pokemon.get('hp_pct', 1))
        p2_hp_pct = p2_state.get('hp_pct', p2_last_pokemon.get('hp_pct', 1))
        p1_status = p1_state.get('status', None)
        p2_status = p2_state.get('status', None)

        if p1_current_name and p1_current_name != p1_last_pokemon.get('name'):
            involuntary_switch = (p1_last_pokemon.get('status') == 'fnt') or (p1_last_pokemon.get('hp_pct', 1) == 0)
            if not involuntary_switch:
                p1_switches += 1 
            p1_last_pokemon['name'] = p1_current_name
            p1_last_pokemon['hp_pct'] = p1_hp_pct
            p1_last_pokemon['status'] = p1_status
        else:
            p1_last_pokemon['hp_pct'] = p1_hp_pct
            p1_last_pokemon['status'] = p1_status

        if p2_current_name and p2_current_name != p2_last_pokemon.get('name'):
            involuntary_switch = (p2_last_pokemon.get('status') == 'fnt') or (p2_last_pokemon.get('hp_pct', 1.0) == 0)
            if not involuntary_switch:
                p2_switches += 1
            p2_last_pokemon['name'] = p2_current_name
            p2_last_pokemon['hp_pct'] = p2_hp_pct
            p2_last_pokemon['status'] = p2_status
        else:
            p2_last_pokemon['hp_pct'] = p2_hp_pct
            p2_last_pokemon['status'] = p2_status

    return p1_switches - p2_switches

def get_effectiveness(move_type, opponent_types):
    """Calcola il moltiplicatore di efficacia (x0, x0.25, x0.5, x1, x2, x4)"""
    if not opponent_types or not move_type:
        return 1.0
        
    effectiveness = 1.0
    
    if move_type in gen1_type:
        attack_map = gen1_type[move_type]
        
        for opp_type in opponent_types:
            if opp_type in attack_map:
                effectiveness *= attack_map[opp_type]

    else: 
        return 1
                
    return effectiveness


