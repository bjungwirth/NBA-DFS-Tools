import json
import csv
import os
import datetime
import numpy as np
import pulp as plp
import random
import itertools
import requests 
from thefuzz import fuzz 


class NBA_Optimizer_IKB:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    player_dict = {}
    team_replacement_dict = {
        "PHO": "PHX",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        #"NY": "NYK",
    }
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    matchup_list = []
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

        self.problem = plp.LpProblem("NBA", plp.LpMaximize)

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )

        salaries_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(salaries_path)
        self.load_projections(projection_path)

        

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for p in reader:
                # Remove punctuation, replace certain characters, and convert to lowercase
                fixed_name = p["name"].replace("-", "#").replace(".", "").replace("'", "").lower()
                #remove punctuation and lowercase
                player_name = p['name']
                player_id = p['id']
                team = p['teamabbrev']
                salary = int(p['salary'])
                self.player_dict[(fixed_name, team)] = {
                    "Fpts": 0.0,
                    "Salary": salary,
                    "Minutes": 0.0,
                    "Name": player_name,
                    "Team": team,
                    "Ownership": 0.0,
                    "StdDev": 0.0,
                    "Position": 'UTIL',
                    "ID": player_id,
                    "fixedName": fixed_name
                }            

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
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                # Remove punctuation, replace certain characters, and convert to lowercase
                player_name = row["name"].replace("-", "#").replace(".", "").replace("'", "").lower()
                if float(row["fpts"]) < self.projection_minimum:
                    continue  # Skip players below the projection minimum

                position = 'UTIL'  # Default position
                team = row["team"]
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]

                key = (player_name, team)
                if key in self.player_dict:
                    self.update_player_stats(key, row)
                else:
                    # Try fuzzy matching
                    matched = False
                    for player, attributes in self.player_dict.items():
                        if fuzz.partial_ratio(player[0], player_name) > 90:
                            print(f'Found fuzzy matching player {player_name} to {player[0]}')
                            self.update_player_stats(player, row)
                            matched = True
                            break
                    if not matched:
                        print(f"Player {player_name} not found in player_dict")

                if team not in self.team_list:
                    self.team_list.append(team)

    def update_player_stats(self, player_key, row):
        # Update player stats in the dictionary
        self.player_dict[player_key]["Fpts"] = float(row["fpts"])
        self.player_dict[player_key]["Minutes"] = float(row["minutes"])
        self.player_dict[player_key]["Ownership"] = float(row["ownership"])
        self.player_dict[player_key]["StdDev"] = float(row["stddev"])

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # Create a binary decision variable for each player for each of their positions
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            if "ID" in attributes:
                player_id = attributes["ID"]
            else:
                print(
                    f"Player in player_dict does not have an ID: {player}. Check for mis-matches between names, teams or positions in projections.csv and player_ids.csv"
                )
            for pos in attributes["Position"]:
                lp_variables[(player, player_id)] = plp.LpVariable(
                    name=f"{player}_{player_id}", cat=plp.LpBinary
                )

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
                    * lp_variables[(player, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Fpts"]
                    * lp_variables[(player, self.player_dict[player]["ID"])]
                    for player in self.player_dict
                ),
                "Objective",
            )

        # Set the salary constraints
        max_salary = 50000 
        min_salary = 45000 

        if self.projection_minimum is not None:
            min_salary = self.min_salary

        # Maximum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, attributes["ID"])]
                for player, attributes in self.player_dict.items()
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, attributes["ID"])]
                for player, attributes in self.player_dict.items()
            )
            >= min_salary,
            "Min Salary",
        )

        # Must not play all 7 players from the same match (8 if dk, 9 if fd)
        matchup_limit = 6
        for matchupIdent in self.matchup_list:
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    if attributes["Matchup"] in matchupIdent
                )
                <= matchup_limit,
                f"Must not play all {matchup_limit} players from match {matchupIdent}",
            )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        if attributes["Name"] in group
                    )
                    >= int(limit),
                    f"At least {limit} players {group}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        if attributes["Name"] in group
                    )
                    <= int(limit),
                    f"At most {limit} players {group}",
                )

        for matchup, limit in self.matchup_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    if attributes["Matchup"] == matchup
                )
                <= int(limit),
                "At most {} players from {}".format(limit, matchup),
            )

        for matchup, limit in self.matchup_at_least.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    if attributes["Matchup"] == matchup
                )
                >= int(limit),
                "At least {} players from {}".format(limit, matchup),
            )

        # Address team limits
        for teamIdent, limit in self.team_limits.items():
            self.problem += plp.lpSum(
                lp_variables[self.player_dict[(player, team)]["ID"]]
                for (player, team) in self.player_dict
                if team == teamIdent
            ) <= int(limit), "At most {} players from {}".format(limit, teamIdent)

        if self.global_team_limit is not None:
            if not (self.site == "fd" and self.global_team_limit >= 4):
                for teamIdent in self.team_list:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if attributes["Team"] == teamIdent
                        )
                        <= int(self.global_team_limit),
                        f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                    )
        
        util_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items()]
        self.problem += plp.lpSum(lp_variables[player] for player in util_players) == 7, "Must have 7 UTIL"
        # Constraints for specific positions

        # Constraint to ensure each player is only selected once
        for player in self.player_dict:
            player_id = self.player_dict[player]["ID"]
            self.problem += (
                plp.lpSum(
                    lp_variables[(player,player_id)]
                )
                <= 1,
                f"Can only select {player} once",
            )

        self.problem.writeLP("problem.lp")
        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.lineups), self.num_lineups
                    )
                )
                break

            # Check for infeasibility
            if plp.LpStatus[self.problem.status] != "Optimal":
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.lineups), self.num_lineups
                    )
                )
                break

            # Get the lineup and add it to our list
            selected_vars = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            self.lineups.append(selected_vars)

            if i % 100 == 0:
                print(i)

            #print(selected_vars)

            # Ensure this lineup isn't picked again
            player_ids = [tpl[1] for tpl in selected_vars]
            player_keys_to_exlude = []
            for key, attr in self.player_dict.items():
                if attr["ID"] in player_ids:
                    player_keys_to_exlude.append((key, attr["ID"]))

            self.problem += (
                plp.lpSum(lp_variables[x] for x in player_keys_to_exlude)
                <= len(selected_vars) - self.num_uniques,
                f"Lineup {i}",
            )

            # self.problem.writeLP("problem.lp")

            # Set a new random fpts projection within their distribution
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
                        * lp_variables[(player, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    ),
                    "Objective",
                )

    def output(self):
        print("Lineups done generating. Outputting.")
        # Construct the output file path
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_optimal_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        )
        with open(out_path, "w", newline='') as f:
            # Setup the CSV writer
            import csv
            writer = csv.writer(f)
            # Write headers
            headers = ["Player1_ID", "Player2_ID",  "Player3_ID", "Player4_ID",  "Player5_ID",  "Player6_ID", "Player7_ID",
                       "Player1",  "Player2", "Player3", "Player4", "Player5", "Player6", "Player7",   
                     "Salary", "Fpts Proj", "Own. Prod.", "Own. Sum.", "Minutes", "StdDev"]
            writer.writerow(headers)

            # Process each lineup
            for lineup in self.lineups:
                row = []
                # Append player name and ID for each player in the lineup
                names = []
                ids = []
                for player in lineup:
                    p = player[0]
                    player_data = self.player_dict[p]
                    names.append(player_data["Name"])
                    ids.append(player_data['ID'])
                row.extend(ids)
                row.extend(names)
                # Calculate lineup stats
                salary = sum(self.player_dict[player[0]]["Salary"] for player in lineup)
                fpts_p = sum(self.player_dict[player[0]]["Fpts"] for player in lineup)
                own_p = np.prod([self.player_dict[player[0]]["Ownership"] / 100 for player in lineup])
                own_s = sum(self.player_dict[player[0]]["Ownership"] for player in lineup)
                mins = sum(self.player_dict[player[0]]["Minutes"] for player in lineup)
                stddev = sum(self.player_dict[player[0]]["StdDev"] for player in lineup)
                # Append calculated stats
                row.extend([salary, round(fpts_p, 2), own_p, own_s, mins, stddev])
                # Write the lineup to the file
                writer.writerow(row)
        print("Output done.")