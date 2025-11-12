from dicts import GEN1_TYPE_CHART,pokemon_types,types


def damage_features(battle: dict) -> dict: #verificata
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

def switch_difference(battle: dict) -> int: #verificata
    """
    Restituisce la differenza (p1_switches - p2_switches) contando solo i switch volontari
    (cioè cambi non dovuti a faint).
    """
    timeline = battle.get('battle_timeline', [])

    # Pokémon inizialmente attivi (fallback se mancanti)
    p1_active_pokemon = battle.get('p1_team_details', [{}])[0].get('name') if battle.get('p1_team_details') else None
    p2_active_pokemon = battle.get('p2_lead_details', {}).get('name', None)

    # Stato "precedente" per rilevare faint o hp == 0
    # Inizializziamo hp_prev a 1 (100%) come fai nel damage_features
    p1_last_pokemon = {'name': p1_active_pokemon, 'hp_pct': 1.0, 'status': None}
    p2_last_pokemon = {'name': p2_active_pokemon, 'hp_pct': 1.0, 'status': None}

    p1_switches = 0
    p2_switches = 0

    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})

        p1_current_name = p1_state.get('name')
        p2_current_name= p2_state.get('name')

        # Prendiamo hp_pct e status (fallback a valori sensati se mancanti)
        p1_hp_pct = p1_state.get('hp_pct', p1_last_pokemon.get('hp_pct', 1))
        p2_hp_pct = p2_state.get('hp_pct', p2_last_pokemon.get('hp_pct', 1))
        p1_status = p1_state.get('status', None)
        p2_status = p2_state.get('status', None)

#se il nome del pokemon corrente è diverso da quello precedente c'è stato un cambio
        if p1_current_name and p1_current_name != p1_last_pokemon.get('name'):
            #voglio controllare se il cambio è volontario o no(pokemon faint o ucciso)
            #creo una variabile vera se si verifica lo status di fainted o se il pokemon non ha più punti
            involuntary_switch = (p1_last_pokemon.get('status') == 'fnt') or (p1_last_pokemon.get('hp_pct', 1) == 0)
            if not involuntary_switch:
                p1_switches += 1 #conto solo gli switch volontari
            # aggiorna 
            p1_last_pokemon['name'] = p1_current_name
            p1_last_pokemon['hp_pct'] = p1_hp_pct
            p1_last_pokemon['status'] = p1_status
        else:
            # aggiorna solo hp/status se stesso rimane attivo (per il prossimo loop)
            p1_last_pokemon['hp_pct'] = p1_hp_pct
            p1_last_pokemon['status'] = p1_status

        # --- P2: stesso ragionamento
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

    # Ritorna solo la differenza come richiesto
    return p1_switches - p2_switches

def get_effectiveness(move_type, opponent_types):
    """Calcola il moltiplicatore di efficacia (x0, x0.25, x0.5, x1, x2, x4)"""
    if not opponent_types or not move_type:
        return 1.0
        
    effectiveness = 1.0
    
    # Controlla il dizionario per il tipo di mossa
    if move_type in GEN1_TYPE_CHART:
        attack_map = GEN1_TYPE_CHART[move_type]
        
        # Controlla entrambi i tipi dell'avversario
        for opp_type in opponent_types:
            if opp_type in attack_map:
                effectiveness *= attack_map[opp_type]

    else: 
        return 1
                
    return effectiveness


def diff_coverage_advantage(p1_team_details, p2_known_names): #DA CONTROLLARE
    """
    Viene calcolato il vantaggio di copertura (P1 vs P2).
    Quantifica:
    - P1 Score: Quanti dei Pokémon noti di P2 la squadra di P1 può colpire con mosse super efficaci
    - P2 Score: Quanti dei Pokémon di P1 i Pokémon noti di P2 possono colpire con mosse super efficaci
    """
    
    #Prendiamo i pokemon di tipo attacco di P1 (squadra completa)
    p1_attack_types = set()
    for pokemon in p1_team_details:
        p1_types = pokemon.get('types', [])
        for t in p1_types: 
            if t and t.lower()!= 'notype':
                p1_attack_types.add(t.lower())

    # 2. Ottieni i tipi difensivi di P1 (dalla squadra completa)
    p1_defensive_types = p1_attack_types # In Gen 1, Tipi Off = Tipi Def
    
    # 3. Ottieni i tipi offensivi e difensivi di P2 (SOLO dai Pokémon noti)
    p2_known_types = set()
    for name in p2_known_names:
        if name in pokemon_types:
            for t in pokemon_types[name]:
                if t and t.lower() != 'notype':
                    p2_known_types.add(t.lower())
    
    p2_att_types = p2_known_types
    p2_def_types = p2_known_types

    # 4. Calcola Punteggio P1 (P1 Offesa vs P2 Difesa)
    # Quanti dei tipi difensivi noti di P2 sono coperti da P1?
    p1_coverage_score = 0
    for p1_att in p1_attack_types:
        if p1_att in types: # 'types' è il dizionario di efficacia offensiva
            for p2_def in p2_def_types:
                if p2_def in types[p1_att]:
                    p1_coverage_score += 1
                    break # Trovato, passa al prossimo tipo offensivo di P1

    # 5. Calcola Punteggio P2 (P2 Offesa vs P1 Difesa)
    # Quanti dei tipi difensivi di P1 sono coperti dai tipi noti di P2?
    p2_coverage_score = 0
    for p2_att in p2_att_types:
        if p2_att in types:
            for p1_def in p1_defensive_types:
                if p1_def in types[p2_att]:
                    p2_coverage_score += 1
                    break # Trovato, passa al prossimo tipo offensivo di P2

    # 6. Ritorna la differenza netta
    return p1_coverage_score - p2_coverage_score