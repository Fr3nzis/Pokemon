import math


def get_pokemon_types(battle_details, pokemon_name):
    """
    Cerca i tipi di un Pokémon nei dettagli statici della squadra P1 o del Lead di P2.
    Restituisce una lista di tipi in minuscolo, escludendo 'notype'.
    """
    for p in battle_details.get('p1_team_details', []):
        if p.get('name') == pokemon_name:
            return [t.lower() for t in p.get('types', []) if t and t.lower() != 'notype']

    p2_lead = battle_details.get('p2_lead_details', {})
    if p2_lead.get('name') == pokemon_name:
        return [t.lower() for t in p2_lead.get('types', []) if t and t.lower() != 'notype']

    return []


def is_super_effective(move_type: str, defender_types: list[str], type_chart: dict) -> bool:
    """
    Restituisce True se la mossa è super efficace contro almeno un tipo difensivo.
    """
    move_type = (move_type or "").lower()
    if move_type not in type_chart:
        return False
    return any(def_type in type_chart[move_type] for def_type in defender_types)


def momentum_score(battle):
    """
    Calcola un punteggio di momentum:
    - Conta i turni dove P1 infligge più danno di quanto ne subisca.
    - Valorizza le serie consecutive di turni favorevoli (streaks).
    """
    timeline = battle.get("battle_timeline", [])
    p1_last_hp_pct = 1
    p2_last_hp_pct = 1
    p1_momentum_turns = 0
    p1_current_streak = 0
    p1_max_streak = 0

    for turn in timeline:
        p1_hp_pct = turn.get("p1_pokemon_state", {}).get("hp_pct", p1_last_hp_pct)
        p2_hp_pct = turn.get("p2_pokemon_state", {}).get("hp_pct", p2_last_hp_pct)

        p1_damage_inflicted = max(0, p2_last_hp_pct - p2_hp_pct)
        p1_damage_received = max(0, p1_last_hp_pct - p1_hp_pct)

        if p1_damage_inflicted > p1_damage_received:
            p1_momentum_turns += 1
            p1_current_streak += 1
            p1_max_streak = max(p1_max_streak, p1_current_streak)
        else:
            p1_current_streak = 0

        p1_last_hp_pct, p2_last_hp_pct = p1_hp_pct, p2_hp_pct

    return p1_momentum_turns + (0.5 * p1_max_streak)


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

        p1_current_hp = p1_state.get('hp_pct', 0)
        p2_current_hp = p2_state.get('hp_pct', 0)
        p1_current_name = p1_state.get('name')
        p2_current_name = p2_state.get('name')

        if p1_current_name != p1_active_pokemon:
            p1_last_hp_pct = p1_current_hp
            p1_active_pokemon = p1_current_name
        if p2_current_name != p2_active_pokemon:
            p2_last_hp_pct = p2_current_hp
            p2_active_pokemon = p2_current_name

        p1_damage_inflicted = max(0, p2_last_hp_pct - p2_current_hp)
        p1_damage_received = max(0, p1_last_hp_pct - p1_current_hp)

        p1_net_damage += (p1_damage_inflicted - p1_damage_received)
        p1_total_damage_inflicted += p1_damage_inflicted
        p1_total_damage_received += p1_damage_received

        p1_last_hp_pct = p1_current_hp
        p2_last_hp_pct = p2_current_hp

    epsilon = 1e-7
    if p1_total_damage_received < epsilon:
        p1_damage_ratio = p1_total_damage_inflicted
    else:
        p1_damage_ratio = p1_total_damage_inflicted / p1_total_damage_received

    return {
        'p1_net_damage': p1_net_damage,
        'p1_damage_ratio': p1_damage_ratio
    }
