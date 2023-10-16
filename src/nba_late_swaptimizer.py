import json
import csv
import os
import datetime
import re
import numpy as np
import pulp as plp
from random import shuffle, choice
import itertools

class NBA_Late_Swaptimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    output_lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    ids_to_gametime = {}
    global_team_limit = None
    projection_minimum = None
    randomness_amount = 0
    min_salary = None

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)
        
        late_swap_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['late_swap_path']))
        self.load_player_lineups(late_swap_path)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json'), encoding='utf-8-sig') as json_file:
            self.config = json.load(json_file)
            
    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                team = row['TeamAbbrev'] if self.site == 'dk' else row['Team']
                position = row['Position']
                
                if (player_name, position, team) in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[(player_name, position, team)]['ID'] = int(row['ID'])
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game Info'].split(' ')[0]
                        self.player_dict[(player_name, position, team)]['GameTime'] = ' '.join(row['Game Info'].split()[1:])
                        self.player_dict[(player_name, position, team)]['GameTime'] = datetime.datetime.strptime(self.player_dict[(player_name, position, team)]['GameTime'][:-3], '%m/%d/%Y %I:%M%p')
                        self.ids_to_gametime[int(row['ID'])] = self.player_dict[(player_name, position, team)]['GameTime']
                    else:
                        self.player_dict[(player_name, position, team)]['ID'] = row['Id'].replace('-', '#')
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game']
        
    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]
        self.min_salary = int(self.config["min_lineup_salary"])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row['name'].replace('-', '#')
                if float(row['fpts']) < self.projection_minimum:
                    continue
                
                position = row['position']
                team = row['team']
                    
                self.player_dict[(player_name, position, team)] = {
                    'Fpts': float(row['fpts']),
                    'Salary': int(row['salary'].replace(',', '')),
                    'Minutes': float(row['minutes']),
                    'Name': row['name'],
                    'Team': row['team'],
                    'Ownership': float(row['own%']),
                    'StdDev': float(row['stddev']),
                    'Position': [pos for pos in row['position'].split('/')],
                }
                if row['team'] not in self.team_list:
                    self.team_list.append(row['team'])
       
    # Load user lineups for late swap
    def load_player_lineups(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if row['entry id'] != '' and self.site == 'dk':
                    # current_time = datetime.datetime.now()  # get the current time
                    current_time = datetime.datetime(2023, 10, 24, 20, 0)
                    PG_id = re.search(r"\((\d+)\)", row['pg']).group(1)
                    SG_id = re.search(r"\((\d+)\)", row['sg']).group(1)
                    SF_id = re.search(r"\((\d+)\)", row['sf']).group(1)
                    PF_id = re.search(r"\((\d+)\)", row['pf']).group(1)
                    C_id = re.search(r"\((\d+)\)", row['c']).group(1)
                    G_id = re.search(r"\((\d+)\)", row['g']).group(1)
                    F_id = re.search(r"\((\d+)\)", row['f']).group(1)
                    UTIL_id = re.search(r"\((\d+)\)", row['util']).group(1)
                    self.lineups.append(
                        {
                            'entry_id': row['entry id'],
                            'contest_id': row['contest id'],
                            'PG': row['pg'].replace('-', '#'),
                            'SG': row['sg'].replace('-', '#'),
                            'SF': row['sf'].replace('-', '#'),
                            'PF': row['pf'].replace('-', '#'),
                            'C': row['c'].replace('-', '#'),
                            'G': row['g'].replace('-', '#'),
                            'F': row['f'].replace('-', '#'),
                            'UTIL': row['util'].replace('-', '#'),
                            'PG_is_locked': current_time > self.ids_to_gametime[int(PG_id)],
                            'SG_is_locked': current_time > self.ids_to_gametime[int(SG_id)],
                            'SF_is_locked': current_time > self.ids_to_gametime[int(SF_id)],
                            'PF_is_locked': current_time > self.ids_to_gametime[int(PF_id)],
                            'C_is_locked': current_time > self.ids_to_gametime[int(C_id)],
                            'G_is_locked': current_time > self.ids_to_gametime[int(G_id)],
                            'F_is_locked': current_time > self.ids_to_gametime[int(F_id)],
                            'UTIL_is_locked': current_time > self.ids_to_gametime[int(UTIL_id)],
                        }
                    )
       
                
    def swaptimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        
        
        for lineup_obj in self.lineups:
            # print(lineup_obj)
            already_used_players = [lineup_obj['PG'], lineup_obj['SG'], lineup_obj['SF'], lineup_obj['PF'], lineup_obj['C'], lineup_obj['G'], lineup_obj['F'], lineup_obj['UTIL']]
            self.problem = plp.LpProblem('NBA', plp.LpMaximize)
            
            # for (player, pos_str, team) in self.player_dict:
            #     print((player, pos_str, team))
            #     print(self.player_dict[(player, pos_str, team)]["ID"])

            lp_variables = {}
            for player, attributes in self.player_dict.items():
                player_id = attributes['ID']
                for pos in attributes['Position']:
                    lp_variables[(player, pos, player_id)] = plp.LpVariable(name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary)
        
            
            
            # set the objective - maximize fpts & set randomness amount from config
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[player]["Fpts"],
                            (
                                self.player_dict[player]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position']
                    ),
                    "Objective",
                )
            else:
                self.problem += (
                    plp.lpSum(
                        self.player_dict[player]["Fpts"]
                        * lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position']
                    ),
                    "Objective",
                )
            
            
            # Set the salary constraints
            max_salary = 50000 if self.site == 'dk' else 60000
            min_salary = 49000 if self.site == 'dk' else 59000
            
            if self.projection_minimum is not None:
                min_salary = self.min_salary
            
            # Maximum Salary Constraint
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position']
                )
                <= max_salary,
                "Max Salary",
            )

            # Minimum Salary Constraint
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position']
                )
                >= min_salary,
                "Min Salary",
            )

            # Address limit rules if any
            for limit, groups in self.at_least.items():
                for group in groups:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes['ID'])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes['Position'] if attributes['Name'] in group
                        )
                        >= int(limit),
                        f"At least {limit} players {group}",
                    )

            for limit, groups in self.at_most.items():
                for group in groups:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes['ID'])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes['Position'] if attributes['Name'] in group
                        )
                        <= int(limit),
                        f"At most {limit} players {group}",
                    )

            for matchup, limit in self.matchup_limits.items():
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position'] if attributes['Matchup'] == matchup
                    )
                    <= int(limit),
                    "At most {} players from {}".format(limit, matchup)
                )

            for matchup, limit in self.matchup_at_least.items():
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position'] if attributes['Matchup'] == matchup
                    )
                    >= int(limit),
                    "At least {} players from {}".format(limit, matchup)
                )

            # Address team limits
            for teamIdent, limit in self.team_limits.items():
                self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                        for (player, pos_str, team) in self.player_dict if team == teamIdent) <= int(limit), "At most {} players from {}".format(limit, teamIdent)
                
            if self.global_team_limit is not None:
                for teamIdent in self.team_list:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes['ID'])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes['Position']
                            if attributes["Team"] == teamIdent
                        )
                        <= int(self.global_team_limit),
                        f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                    )

            if self.site == 'dk':
                # Constraints for specific positions
                for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 1, f"Must have at least 1 {pos}"

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]['ID']
                    self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

                # Handle the G, F, and UTIL spots
                guards = ['PG', 'SG']
                forwards = ['SF', 'PF']

                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in guards if pos in attributes['Position']) >= 3, f"Must have at least 3 guards"
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in forwards if pos in attributes['Position']) >= 3, f"Must have at least 3 forwards"

                # UTIL can be from any position. But you don't really need a separate constraint for UTIL.
                # It's automatically handled because you're ensuring every other position is filled and each player is selected at most once. 
                # If you've correctly handled the other positions, UTIL will be the remaining player.

                # Total players constraint
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in attributes['Position']) == 8, f"Must have 8 players"
            else:
            # Constraints for specific positions
                for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    if pos == 'C':
                        self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 1, f"Must have at least 1 {pos}"
                    else:
                        self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 2, f"Must have at least 2 {pos}"


                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]['ID']
                    self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

                
                # Total players constraint
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in attributes['Position']) == 9, f"Must have 9 players"

            
           
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.output_lineups), self.lineups))
                break
                
            # Check for infeasibility
            if plp.LpStatus[self.problem.status] != 'Optimal':
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.output_lineups), self.lineups))
                break

            # Get the lineup and add it to our list
            selected_vars = [player for player in lp_variables if lp_variables[player].varValue != 0]
            # print(selected_vars)

            selected_players_info = [var[0] for var in selected_vars]

            players = []
            for key in self.player_dict.keys():
                if key in selected_players_info:
                    players.append(key)
                      
            # fpts_used = self.problem.objective.value()
            # print(fpts_used, players)
            self.output_lineups.append((players, lineup_obj))


    def output(self):
        print('Lineups done generating. Outputting.')
       
        sorted_lineups = []
        for lineup, old_lineup in self.output_lineups:
            sorted_lineup = self.sort_lineup_dk(lineup, old_lineup)
            quit()
            if self.site == 'dk':
                sorted_lineup = self.adjust_roster_for_late_swap_dk(sorted_lineup)
            sorted_lineups.append((sorted_lineup, old_lineup))

        print(sorted_lineup)
        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_optimal_lineups_{}.csv'.format(self.site, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'PG,SG,SF,PF,C,G,F,UTIL,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for x in sorted_lineups:
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    own_s = sum(self.player_dict[player]['Ownership'] for player in x)
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    stddev = sum(
                        [self.player_dict[player]['StdDev'] for player in x])
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}'.format(
                        self.player_dict[x[0][0]]['Name'], self.player_dict[x[0]]['ID'],
                        self.player_dict[x[1]]['Name'], self.player_dict[x[1]]['ID'],
                        self.player_dict[x[2]]['Name'], self.player_dict[x[2]]['ID'],
                        self.player_dict[x[3]]['Name'], self.player_dict[x[3]]['ID'],
                        self.player_dict[x[4]]['Name'], self.player_dict[x[4]]['ID'],
                        self.player_dict[x[5]]['Name'], self.player_dict[x[5]]['ID'],
                        self.player_dict[x[6]]['Name'], self.player_dict[x[6]]['ID'],
                        self.player_dict[x[7]]['Name'], self.player_dict[x[7]]['ID'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write(
                    'PG,PG,SG,SG,SF,SF,PF,PF,C,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for x in sorted_lineups:
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    own_s = sum(self.player_dict[player]['Ownership'] for player in x)
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    stddev = sum(
                        [self.player_dict[player]['StdDev'] for player in x])
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['ID'].replace('#', '-'), self.player_dict[x[0]]['Name'],
                        self.player_dict[x[1]]['ID'].replace('#', '-'), self.player_dict[x[1]]['Name'],
                        self.player_dict[x[2]]['ID'].replace('#', '-'), self.player_dict[x[2]]['Name'],
                        self.player_dict[x[3]]['ID'].replace('#', '-'), self.player_dict[x[3]]['Name'],
                        self.player_dict[x[4]]['ID'].replace('#', '-'), self.player_dict[x[4]]['Name'],
                        self.player_dict[x[5]]['ID'].replace('#', '-'), self.player_dict[x[5]]['Name'],
                        self.player_dict[x[6]]['ID'].replace('#', '-'), self.player_dict[x[6]]['Name'],
                        self.player_dict[x[7]]['ID'].replace('#', '-'), self.player_dict[x[7]]['Name'],
                        self.player_dict[x[8]]['ID'].replace('#', '-'), self.player_dict[x[8]]['Name'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')


    def sort_lineup_dk(self, lineup, old_lineup):
        print(old_lineup)
        print(lineup)
        # Step 1: Create a mapping from player's name to their positional info
        player_to_info = {player[0]: player for player in lineup}
        
        # Defined order for positions
        order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        # Resultant lineup based on order
        sorted_lineup = [None] * 8
        
        # Step 2: Check the old_lineup and fill in locked players
        for position in order:
            player_name = old_lineup[position].split(" (")[0]  # Extract player's name from old lineup entry
            if old_lineup[f"{position}_is_locked"] and player_name in player_to_info:
                sorted_lineup[order.index(position)] = player_to_info[player_name]
        
        # Step 3: Fill in non-locked players
        for player, position, team in lineup:
            if not any(item and item[0] == player for item in sorted_lineup):  # Check if player is already in sorted_lineup
                available_slots = [pos for pos in position.split('/') if not sorted_lineup[order.index(pos)]]
                
                # For players with flexible positions like SG/SF, they can also fit in G
                if any(pos in ['PG', 'SG'] for pos in position.split('/')) and not sorted_lineup[order.index('G')]:
                    available_slots.append('G')
                
                # Similarly, SF/PF can fit in F
                if any(pos in ['SF', 'PF'] for pos in position.split('/')) and not sorted_lineup[order.index('F')]:
                    available_slots.append('F')

                # If UTIL position is available
                if 'UTIL' in order and not sorted_lineup[order.index('UTIL')]:
                    available_slots.append('UTIL')
                
                for slot in available_slots:
                    if not sorted_lineup[order.index(slot)]:  # If slot is unoccupied
                        sorted_lineup[order.index(slot)] = (player, position, team)
                        break
        print(sorted_lineup)
        return sorted_lineup

        
    def sort_lineup_fd(self, lineup, old_lineup=None):
        order = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
        sorted_lineup = [None] * 9  # Total 9 players

        def find_and_remove(positions, player_id=None):
            for pos in positions:
                for player in lineup:
                    if (not player_id or player[0] == player_id) and player[1] == pos:
                        lineup.remove(player)
                        return player
            return None
        
        # Fill in locked positions from old_lineup
        if old_lineup:
            for pos in order:
                locked = getattr(old_lineup, f"{pos}_is_locked", False)
                if locked:
                    player_id = getattr(old_lineup, pos)
                    # Get player from lineup based on ID and position
                    player = find_and_remove([pos], player_id)
                    sorted_lineup[order.index(pos)] = player
            
        # Fill the first occurrences of each position
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            if sorted_lineup[order.index(pos)] is None:  # Skip if already filled from old_lineup
                player = find_and_remove([pos])
                if player:
                    sorted_lineup[order.index(pos)] = player

        # Now, fill the second occurrences or use dual positions if available
        for pos in ['PG', 'SG', 'SF', 'PF']:
            index = order.index(pos) + 1  # The next position after the first occurrence
            if sorted_lineup[index] is None:  # Skip if already filled from old_lineup
                player = find_and_remove([pos]) or find_and_remove([f"{pos}/{order[index+1]}"])
                if player:
                    sorted_lineup[index] = player

        # Fill any remaining spots with remaining players
        for i, player in enumerate(sorted_lineup):
            if not player and lineup:
                sorted_lineup[i] = lineup.pop(0)

        return sorted_lineup

    def adjust_roster_for_late_swap_dk(self, lineup):
        print(lineup)
        sorted_lineup = [None] * 8
            
        # Swap PG and G spots if PG is later than G and G spot is PG eligible
        if self.player_dict[lineup[0]]['GameTime'] > self.player_dict[lineup[5]]['GameTime'] and 'PG' in self.player_dict[lineup[5]]['Position']:
            sorted_lineup[0] = lineup[5]
            sorted_lineup[5] = lineup[0]
        else:
            sorted_lineup[0] = lineup[0]
            sorted_lineup[5] = lineup[5]
        
        # Swap SG and G spots if SG is later than G and G spot is SG eligible
        if self.player_dict[lineup[1]]['GameTime'] > self.player_dict[lineup[5]]['GameTime'] and 'SG' in self.player_dict[lineup[5]]['Position']:
            sorted_lineup[1] = lineup[5]
            sorted_lineup[5] = lineup[1]
        else:
            sorted_lineup[1] = lineup[1]
            sorted_lineup[5] = lineup[5]
        
        # Swap SF and F spots if SF is later than F and F spot is SF eligible
        if self.player_dict[lineup[2]]['GameTime'] > self.player_dict[lineup[6]]['GameTime'] and 'SF' in self.player_dict[lineup[6]]['Position']:
            sorted_lineup[2] = lineup[6]
            sorted_lineup[6] = lineup[2]
        else:
            sorted_lineup[2] = lineup[2]
            sorted_lineup[6] = lineup[6]
            
        # Swap PF and F spots if PF is later than F and F spot is PF eligible
        if self.player_dict[lineup[3]]['GameTime'] > self.player_dict[lineup[6]]['GameTime'] and 'PF' in self.player_dict[lineup[6]]['Position']:
            sorted_lineup[3] = lineup[6]
            sorted_lineup[6] = lineup[3]
        else:
            sorted_lineup[3] = lineup[3]
            sorted_lineup[6] = lineup[6]
            
        # Swap C and UTIL spots if C is later than UTIL and UTIL spot is C eligible
        if self.player_dict[lineup[4]]['GameTime'] > self.player_dict[lineup[7]]['GameTime'] and 'C' in self.player_dict[lineup[7]]['Position']:
            sorted_lineup[4] = lineup[7]
            sorted_lineup[7] = lineup[4]
        else:
            sorted_lineup[4] = lineup[4]
            sorted_lineup[7] = lineup[7]
        
        # Swap any roster position with UTIL if UTIL is earlier than position and UTIL is eligible for position
        for i, player in enumerate(lineup):
            if self.player_dict[player]['GameTime'] < self.player_dict[lineup[7]]['GameTime']:
                for position in self.player_dict[lineup[7]]['Position']:
                    if position in self.player_dict[player]['Position']:
                        sorted_lineup[i] = lineup[7]
                        sorted_lineup[7] = lineup[i]
                        break
                    
        return sorted_lineup