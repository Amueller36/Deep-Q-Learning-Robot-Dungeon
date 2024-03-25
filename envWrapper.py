import copy
import functools
import json
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set

import gymnasium.spaces
import requests
import numpy as np
import torch.nn.functional
from pydantic import BaseModel
import logging as log

from Config import Config
from rest.create_game_request import CreateGameRequest
from rest.create_game_response import CreateGameResponse
from rest.join_player_request import JoinPlayerRequest
from rest.join_player_response import JoinPlayerResponse
from util import log_scale, normalize, requests_retry_session


class ActionsEnum(Enum):
    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3
    ATTACK_LOWEST_HP_ENEMY_ROBOT = 4  # HP > 0
    ATTACK_HIGHEST_VALUE_ENEMY_ROBOT = 5
    UPGRADE_DAMAGE_1 = 6
    UPGRADE_DAMAGE_2 = 7
    UPGRADE_DAMAGE_3 = 8
    UPGRADE_DAMAGE_4 = 9
    UPGRADE_DAMAGE_5 = 10
    UPGRADE_HEALTH_1 = 11
    UPGRADE_HEALTH_2 = 12
    UPGRADE_HEALTH_3 = 13
    UPGRADE_HEALTH_4 = 14
    UPGRADE_HEALTH_5 = 15
    UPGRADE_MINING_1 = 16
    UPGRADE_MINING_2 = 17
    UPGRADE_MINING_3 = 18
    UPGRADE_MINING_4 = 19
    UPGRADE_MINING_5 = 20
    UPGRADE_STORAGE_1 = 21
    UPGRADE_STORAGE_2 = 22
    UPGRADE_STORAGE_3 = 23
    UPGRADE_STORAGE_4 = 24
    UPGRADE_STORAGE_5 = 25
    UPGRADE_MINING_SPEED_1 = 26
    UPGRADE_MINING_SPEED_2 = 27
    UPGRADE_MINING_SPEED_3 = 28
    UPGRADE_MINING_SPEED_4 = 29
    UPGRADE_MINING_SPEED_5 = 30
    UPGRADE_ENERGY_1 = 31
    UPGRADE_ENERGY_2 = 32
    UPGRADE_ENERGY_3 = 33
    UPGRADE_ENERGY_4 = 34
    UPGRADE_ENERGY_5 = 35
    UPGRADE_ENERGY_REGEN_1 = 36
    UPGRADE_ENERGY_REGEN_2 = 37
    UPGRADE_ENERGY_REGEN_3 = 38
    UPGRADE_ENERGY_REGEN_4 = 39
    UPGRADE_ENERGY_REGEN_5 = 40
    BUY_ENERGY_RESTORE = 41
    BUY_HEALTH_RESTORE = 42
    MINE_RESOURCE = 43
    REGENERATE = 44
    SELL_RESOURCE = 45

    def get_item_name(self):
        if 5 < self.value < 41:
            return self.name.split("UPGRADE_")[1]
        elif 40 < self.value < 43:
            return self.name.split("BUY_")[1]
        else:
            raise ValueError(
                f"Action {self.name} does not have an item name. It has to be a Buying or Upgrading action.")


class CommandType(Enum):
    MOVEMENT = "MOVEMENT"
    BATTLE = "BATTLE"
    MINING = "MINING"
    REGENERATE = "REGENERATE"
    BUYING = "BUYING"
    SELLING = "SELLING"


class CommandObject(BaseModel):
    robot_id: Optional[str] = None
    planet_id: Optional[str] = None
    target_id: Optional[str] = None
    item_name: Optional[str] = None
    item_quantity: Optional[int] = None


class Command(BaseModel):
    player_name: str
    game_id: str
    command_type: str
    command_object: CommandObject


class PlayersActions(BaseModel):
    player_name: str
    game_id: str
    buy_new_robots: int = 0,
    actions_for_robots: dict[str, ActionsEnum] = {}  # {robot_id: action}


class Neighbours(BaseModel):
    NORTH: Optional[str] = None
    EAST: Optional[str] = None
    WEST: Optional[str] = None
    SOUTH: Optional[str] = None


class ResourceType(Enum):
    COAL = "COAL"
    IRON = "IRON"
    GEM = "GEM"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"


class ObservationPlanet(BaseModel):
    x: int
    y: int
    movement_difficulty: int
    resource: Optional[ResourceType]
    resource_amount: int
    amount_of_friendly_robots: int
    fighting_score_friendly_robots: float
    amount_of_enemy_robots: int
    fighting_score_enemy_robots: float
    neighbours: Neighbours


class ObservationRobot(BaseModel):
    x: int
    y: int
    planet_id: str
    robot_id: str
    health: int
    max_health: int
    energy: int
    max_energy: int
    energy_regen: int
    storage: int
    max_storage: int
    mining_speed: int
    mineable_resources: List[ResourceType]
    damage: int
    fighting_score: float
    money_value: int  # The value of the robot in money, determined by the Upgrade levels.
    money_made: int  # The money the robot has made in total.


def one_hot_encode_resources(mineable_resources: List[ResourceType] | None):
    """
    Method needs to convert the mineable resources of a robot to a one-hot encoded vector format, probably with numpy.
    for example if the Robot can mine only Coal [1,0,0,0,0]
    """
    if mineable_resources is None:
        return np.zeros(5, dtype=np.int8)
    available_resources = [resource for resource in ResourceType]
    encoding = np.zeros(len(available_resources), dtype=np.int8)

    for resource in mineable_resources:
        encoding[available_resources.index(resource)] = 1

    return encoding


def initialize_planet_state_vector(vector_size: int):
    """Initialisiert einen Zustandsvektor für einen Planeten mit -1."""
    return np.full(vector_size, -1.0)


def planet_state_to_vector(planet: ObservationPlanet):
    """
    Method needs to convert the state of a planet to a vector format, probably with numpy.
    Current Vector Size count : 13  (8 + 5(One-Hot Encoded Resources))
    Should Include:
    - X, Y Position of the planet : int
    - MovementDifficulty: MovementDifficulty of the planet : int
    - Resource: Resource of the planet : int
    - ResourceAmount: ResourceAmount of the planet : int
    - AmountOfFriendlyRobots: AmountOfFriendlyRobots on the planet : int
    - FightingScoreFriendlyRobots: FightingScoreFriendlyRobots on the planet : float
    - AmountOfEnemyRobots: AmountOfEnemyRobots on the planet : int
    - FightingScoreEnemyRobots: FightingScoreEnemyRobots on the planet : float
    - Neighbours: Neighbours of the planet : int
    """

    # Apply logarithmic scaling to fighting scores for dynamic range adjustment
    # log.warning("FIGHTING SCORE FRIENDLY ROBOTS " + str(planet.fighting_score_friendly_robots))
    # log.warning("FIGHTING SCORE ENEMY ROBOTS" + str(planet.fighting_score_enemy_robots))
    fighting_score_friendly_robots_scaled = log_scale(planet.fighting_score_friendly_robots)
    fighting_score_enemy_robots_scaled = log_scale(planet.fighting_score_enemy_robots)

    # Normalize scalar values
    movement_difficulty_normalized = normalize(planet.movement_difficulty, 1, 3)
    resource_amount_normalized = normalize(planet.resource_amount, 0.0, 10000.0)

    # One-hot encode the resource
    resource_encoded = one_hot_encode_resources(
        [planet.resource]) if planet.resource is not None else one_hot_encode_resources(None)

    # Create a vector for the planet state
    planet_state = np.array([
        planet.x,
        planet.y,
        movement_difficulty_normalized,
        resource_amount_normalized,
        planet.amount_of_friendly_robots,
        fighting_score_friendly_robots_scaled,
        planet.amount_of_enemy_robots,
        fighting_score_enemy_robots_scaled
    ])
    log.debug(f"\nPLANET X {planet.x}\n"
              f"PLANET Y {planet.y}\n"
              f"MOVEMENT DIFFICULTY {movement_difficulty_normalized}\n"
              f"RESOURCE AMOUNT {resource_amount_normalized}\n"
              f"AMOUNT OF FRIENDLY ROBOTS {planet.amount_of_friendly_robots}\n"
              f"FIGHTING SCORE FRIENDLY ROBOTS {fighting_score_friendly_robots_scaled}\n"
              f"AMOUNT OF ENEMY ROBOTS {planet.amount_of_friendly_robots}\n"
              f"FIGHTING SCORE ENEMY ROBOTS {fighting_score_enemy_robots_scaled}\n"
              f"RESOURCE ENCODED {resource_encoded}\n")

    # Append the one-hot encoded resource to the planet state

    planet_state = np.concatenate((planet_state, resource_encoded))

    return planet_state


class PlayerState(BaseModel):
    current_round: int
    player_name: str
    money: int
    total_money_made: int
    map: Dict[str, ObservationPlanet]  # uuid of planet -> planet
    visited_planets: Set[str]  # uuid of planet
    alive_robots: Dict[str, ObservationRobot]  # uuid of our robot -> our robot
    alive_enemy_robots: Dict[str, ObservationRobot]  # uuid of enemy robot -> enemy robot
    dead_robots: Dict[str, ObservationRobot]  # uuid of our robot -> robot
    killed_robots: Dict[
        str, List[Tuple[str, ObservationRobot]]]  # our uuid of attacker robot -> (Enemy Player name, Enemy robot)
    robot_rewards: Dict[str, float] = {}  # uuid of robot -> reward

    def get_own_robots(self):
        return self.alive_robots.values()

    def get_enemy_robots(self):
        return self.alive_enemy_robots.values()

    def get_enemy_robots_on_planet(self, planet_id: str):
        return [robot for robot in self.alive_enemy_robots.values() if robot.planet_id == planet_id]

    def get_own_robots_on_planet(self, planet_id: str):
        return [robot for robot in self.alive_robots.values() if robot.planet_id == planet_id]

    def to_observation_for_robot(self, robot_id: str, enemy_total_money_made: int = 0):


        # Ensure the robot_id exists in alive_robots
        if robot_id not in self.alive_robots:
            robot = self.dead_robots[robot_id]
        else:
            robot = self.alive_robots[robot_id]

        # Normalize scalar values
        health_normalized = normalize(robot.health, 0, robot.max_health)
        energy_normalized = normalize(robot.energy, 0, robot.max_energy)
        storage_normalized = normalize(robot.storage, 0, robot.max_storage)

        # One-hot encode the minable resources
        minable_resources_encoded = one_hot_encode_resources(robot.mineable_resources)

        # Create a vector for the robot-specific state
        robot_state = np.array([
            self.money,
            self.total_money_made,
            enemy_total_money_made,
            robot.x,
            robot.y,
            health_normalized,
            energy_normalized,
            robot.energy_regen,
            storage_normalized,
            robot.damage,
            robot.mining_speed,
            robot.fighting_score
        ])

        log.debug(f"MONEY {self.money}"
                  f"TOTAL MONEY MADE {self.total_money_made}"
                  f"ENEMY TOTAL MONEY MADE {enemy_total_money_made}"
                  f"ROBOT X {robot.x}"
                  f"ROBOT Y {robot.y}"
                  f"HEALTH {health_normalized}"
                  f"ENERGY {energy_normalized}"
                  f"ENERGY REGEN {robot.energy_regen}"
                  f"STORAGE {storage_normalized}"
                  f"DAMAGE {robot.damage}"
                  f"MINING SPEED {robot.mining_speed}"
                  f"FIGHTING SCORE {robot.fighting_score}")

        # Append the one-hot encoded resources to the robot state
        robot_state = np.concatenate((robot_state, minable_resources_encoded))

        map_state = self._convert_map_to_observation_vector()
        # Combine robot state and map state into one observation vector
        observation = np.concatenate((robot_state, map_state))

        return observation

    def to_observation_for_robot_old(self, robot_id: str, enemy_total_money_made: int = 0):
        """
        Method needs to convert the state without enemy robots, without dead robots and without killed robots to an
        observation format, probably with numpy.

        Should Include:
        - Money
        - Total Money Made
        - X, Y Position of the robot : int
        - Health: Health of the robot : int
        - MaxHealth: MaxHealth of the robot : int
        - Energy: Energy of the robot : int
        - MaxEnergy: MaxEnergy of the robot : int
        - EnergyRegeneration: EnergyRegeneration of the robot : int
        - Storage: Storage of the robot : int
        - MaxStorage: MaxStorage of the robot : int
        - Damage: Damage of the robot : int
        - MinableResources: Resource the robot can mine [0: Coal, 1: Coal,Iron, 2: Coal,Iron, Gem, 3: Coal,Iron,Gem,Gold, 4: Coal,Iron,Gem,Gold,Platin] (For example if the Robot can mine only Coal [1,0,0,0,0] (Binary Representation)
        - FightingScore: FightingScore of the robot : float
        - Map: All Planets with their Resources and Neighbours


        """

        # Ensure the robot_id exists in alive_robots
        if robot_id not in self.alive_robots:
            robot = self.dead_robots[robot_id]
        else:
            robot = self.alive_robots[robot_id]

        # Normalize scalar values
        health_normalized = normalize(robot.health, 0, robot.max_health)
        energy_normalized = normalize(robot.energy, 0, robot.max_energy)
        storage_normalized = normalize(robot.storage, 0, robot.max_storage)

        # One-hot encode the minable resources
        minable_resources_encoded = one_hot_encode_resources(robot.mineable_resources)

        # Create a vector for the robot-specific state
        robot_state = np.array([
            self.money,
            self.total_money_made,
            robot.x,
            robot.y,
            health_normalized,
            energy_normalized,
            robot.energy_regen,
            storage_normalized,
            robot.damage,
            robot.mining_speed,
            robot.fighting_score
        ])

        log.debug(f"MONEY {self.money}"
                  f"TOTAL MONEY MADE {self.total_money_made}"
                  f"ROBOT X {robot.x}"
                  f"ROBOT Y {robot.y}"
                  f"HEALTH {health_normalized}"
                  f"ENERGY {energy_normalized}"
                  f"ENERGY REGEN {robot.energy_regen}"
                  f"STORAGE {storage_normalized}"
                  f"DAMAGE {robot.damage}"
                  f"MINING SPEED {robot.mining_speed}"
                  f"FIGHTING SCORE {robot.fighting_score}")

        # Append the one-hot encoded resources to the robot state
        robot_state = np.concatenate((robot_state, minable_resources_encoded))

        map_state = self._convert_map_to_observation_vector()
        # Combine robot state and map state into one observation vector
        observation = np.concatenate((robot_state, map_state))

        return observation

    def _convert_map_to_observation_vector(self):
        planet_state_vector_dic = {}
        for planet_id, planet in self.map.items():
            empty_planet_state_vector = initialize_planet_state_vector(13)
            planet_state_vector_dic[planet_id] = empty_planet_state_vector

        for planet_id, planet in self.map.items():
            if planet_id in self.visited_planets:
                planet_state_vector_dic[planet_id] = planet_state_to_vector(planet)
                for direction, neighbour_id in planet.neighbours.dict().items():
                    if neighbour_id is not None and neighbour_id not in self.visited_planets:
                        neighbour_planet_vector = planet_state_vector_dic[neighbour_id]
                        # Modify X, Y Position of the planet, because we know these then, and the robot knows them too.
                        neighbour_planet: ObservationPlanet = self.map[neighbour_id]
                        neighbour_planet_vector[0] = neighbour_planet.x
                        neighbour_planet_vector[1] = neighbour_planet.y
                        planet_state_vector_dic[neighbour_id] = neighbour_planet_vector
        map_state = np.concatenate([planet_state_vector for planet_state_vector in planet_state_vector_dic.values()])
        return map_state

    def __str__(self):
        return f"Current Round: {self.current_round}\n" \
               f"Player Name: {self.player_name}\n" \
               f"Money: {self.money}\n" \
               f"Total Money Made: {self.total_money_made}\n" \
               f"Map: {self.map}\n" \
               f"Alive Robots: {self.alive_robots}\n" \
               f"Alive Enemy Robots: {self.alive_enemy_robots}\n" \
               f"Dead Robots: {self.dead_robots}\n" \
               f"Killed Robots: {self.killed_robots}\n"


class RobotGameEnvironment:
    game_id: Optional[str]
    game_host: str
    action_space: gymnasium.spaces.Discrete

    players: List[str]
    player_states: Dict[str, PlayerState]  # player_name -> state
    action_queue_for_round: Dict[
        str, List[Command]]  # player_name -> List of Commands for the current round



    def __init__(self):
        self.game_id = None
        self.game_host = f'{Config.game_host}:{Config.game_port}'
        self.action_space = gymnasium.spaces.Discrete(46)
        self.players = [f"Player{i}" for i in range(1, Config.NUM_PLAYERS + 1)]
        self.player_states = {}  # Beinhaltet Geld, Position/Leben/etc aller seiner Roboter
        self.action_queue_for_round = {}  # Beinhaltet die Aktionen die ein Spieler für die aktuelle Runde ausführen will
        self.global_current_round_state = {}
        self.current_round = 0
        self.max_rounds = Config.NUM_ROUNDS

        # Größe des Beobachtungsraums definieren
        num_robot_features = 17  # Anzahl der direkten Robotermerkmale inkl. Minenressourcen
        old_num_robot_features = 16
        num_planet_features = 13  # Anzahl der Merkmale pro Planet
        num_planets = Config.MAP_SIZE ** 2 - Config.MAP_SIZE  # Gesamtanzahl der Planeten
        observation_length = (num_robot_features + (num_planet_features * num_planets) )* 2 # Wegen "Frame" stacking * 2
        old_observation_length = (old_num_robot_features + (num_planet_features * num_planets))
        self.old_observation_space = gymnasium.spaces.Box(low=-1, high=np.inf, shape=(old_observation_length,),
                                                      dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-1, high=np.inf, shape=(observation_length,),
                                                      dtype=np.float32)

        # Weitere Initialisierungen hier

    def add_robot_action_to_queue(self, player_name: str, robot_id: str, action: ActionsEnum | np.ndarray | np.integer):

        if player_name not in self.action_queue_for_round:
            log.info(f"Player {player_name} not in action queue for round. Adding him.")
            self.action_queue_for_round[player_name] = []

        if isinstance(action, np.ndarray) or isinstance(action, np.integer):
            if np.array_equal(action.shape, self.action_space.shape):  # Verbesserte Formprüfung
                action_index = np.argmax(action)
                try:
                    action = ActionsEnum(action)  # Konvertiere den Index in eine Aktion
                    log.debug(f"Action for robot {robot_id} is {action} CALCULATED BY INDEX {action_index}")
                except ValueError:
                    log.error(f"Invalid action index {action_index} for robot {robot_id}")
                    return  # Optional: Fehlerbehandlung oder Rückkehr
            else:
                log.error("Action array shape does not match action space shape.")
                raise ValueError("Action array shape does not match action space shape.")
                return  # Optional: Fehlerbehandlung oder Rückkehr

        if isinstance(action, ActionsEnum):
            log.debug(f"Action for robot {robot_id} is {action}")
            # Die Überprüfung auf Aktionstyp ist bereits durch isinstance gegeben
            # Füge die Aktion zur Warteschlange hinzu
        else:
            log.error(f"Action {action} is neither an ndarray nor an ActionsEnum instance.")

        if player_name not in self.action_queue_for_round:
            log.debug(f"Player {player_name} not in action queue for round. Adding him.")
            self.action_queue_for_round[player_name] = []

        command = self.convert_to_command_json_format(
            PlayersActions(player_name=player_name, game_id=self.game_id, actions_for_robots={robot_id: action}))[0]
        log.debug(f"Command for player {player_name} is {command}")
        self.action_queue_for_round[player_name].append(command)

    def add_buy_robots_action_to_queue(self, player_name: str, buy_new_robots_amount: int):
        """
        Adds a buying command to the action queue for the current round.
        :param player_name:
        :param buy_new_robots_amount:
        :return:
        """
        if player_name not in self.action_queue_for_round:
            log.debug(f"Player {player_name} not in action queue for round. Adding him.")
            self.action_queue_for_round[player_name] = []

        command: Command = Command(player_name=player_name, game_id=self.game_id, command_type=CommandType.BUYING.name,
                                   command_object=CommandObject(item_quantity=buy_new_robots_amount, item_name="ROBOT"))
        self.action_queue_for_round[player_name].append(command)


    def get_observation_for_robot_old(self, player_name: str, robot_id: str):
        """
        Returns the observation for a robot in the environment.
        With Actions of premade robots already being represented in the state.
        :param player_name: The name of the player who owns the robot.
        :param robot_id: The id of the robot.
        :return: The observation for the robot.
        """
        if len(self.action_queue_for_round[player_name]) == 0:
            return self.player_states[player_name].to_observation_for_robot_old(robot_id)

        latest_command = [self.action_queue_for_round[player_name][-1].dict()]

        try:
            response = requests_retry_session().post(
                f"{self.game_host}/games/{self.game_id}/commands/hypothetically",
                json=latest_command,
                timeout=10
            )
            response.raise_for_status()  # This will raise an exception for HTTP error codes
            player_state_with_premade_robots_actions_included = response.json()
        except requests.exceptions.HTTPError as err:
            print(f'HTTPError: {err}')
            # Handle HTTP error (e.g., return an error, retry, or default behavior)
            # Consider what to do in case of failure. This might depend on your application's needs.
            return None
        except requests.exceptions.ConnectionError as err:
            print(f'ConnectionError: {err}')
            return None
        except requests.exceptions.Timeout as err:
            print(f'Timeout: {err}')
            return None
        except requests.exceptions.RequestException as err:
            print(f'RequestException: {err}')
            return None

        new_simulated_player_state: PlayerState = PlayerState.parse_obj(
            player_state_with_premade_robots_actions_included)
        self.player_states[player_name] = new_simulated_player_state
        return self.player_states[player_name].to_observation_for_robot_old(robot_id)

    def reset(self, amount_of_robots_to_buy: int = 5):
        """
        Erstellt ein neues Spiel auf dem Server, registriert die Spieler und startet das Spiel,
        kauft amount_of_robots_to_buy Roboter für jeden Spieler und gibt den Anfangszustand zurück.
        """
        # 0. Lösche das alte Spiel, falls vorhanden
        if self.game_id:
            delete_response = requests.delete(f'{self.game_host}/games/{self.game_id}').text
            log.info(f"Delete Game Response {delete_response}")
        self.__init__()  # Reset all variables

        # 1. Erstelle ein neues Spiel auf dem Server
        create_game_response = requests.post(self.game_host + '/games', json=CreateGameRequest().dict())
        game_id = CreateGameResponse.parse_obj(create_game_response.json()).game_id
        self.game_id = game_id

        # 2. Registriere die Spieler für das Spiel
        for player in self.players:
            join_response = requests.put(self.game_host + '/games/' + game_id,
                                         json=JoinPlayerRequest(player_name=player).dict())
            join_data = JoinPlayerResponse.parse_obj(join_response.json())
            log.debug(join_data)

        # 3. Starte das Spiel
        game_start_response = requests.post(self.game_host + '/games/' + game_id + '/gameCommands/start')
        if game_start_response.status_code != 200:
            log.error("Error while starting game")
            return
        log.info(f"Game {game_id} started successfully.")

        # 4. Kaufe 5 Roboter für jeden Spieler
        log.info(f"Buying {amount_of_robots_to_buy} Robots for each player")
        for player in self.players:
            self.add_buy_robots_action_to_queue(player, amount_of_robots_to_buy)
        self.step()

        # 5. Erhalte den Zustand für Runde 1 für jeden Spieler

        self._get_state_from_server_for_players_for_current_round()

        # self._get_round_state_from_server_for_current_round()

        # Rückgabe des Anfangszustandes
        return self.player_states

    def step(self):
        winner: str = None  # Player Name of the Winner
        terminated, truncated = False, False

        current_round_state = copy.deepcopy(self.player_states)

        # Überprüfen der Aktionen für Gültigkeit
        for idx, player_name in enumerate(self.players):
            log.info(f"About to send Commands to Server for Player {player_name}...")
            response = self._send_actions_to_server(
                [self.action_queue_for_round[player_name]])  # Send all actions/commands for the player
            if idx == len(self.players) - 1:
                # If the last player submitted his commands, the server should not include a message
                assert response.status_code == 200, (f"Sent all Player commands to the Server but it still waits for "
                                                     f"other players to send commands which shouldnt be the case...\n Server response: {response.status_code} {response.text} ")

        self._get_state_from_server_for_players_for_current_round()

        new_round_state = self.player_states

        # Überprüfen, ob das Spiel zu Ende ist und wer gewonnen hat
        # Das Spiel ist zu Ende, wenn ein Spieler keine lebenden Roboter mehr hat und kein Geld mehr um neue zu kaufen, oder die maximale Anzahl an Runden erreicht wurde.
        # Der Gewinner ist der mit dem meisten Geld. Bzw. der Spieler, der noch lebende Roboter hat, wohingegen kein anderer Spieler mehr lebende Roboter hat.
        if self.current_round == self.max_rounds:
            terminated = True
            truncated = True
            log.info("Max Rounds reached. Game terminated.")
            # if all players have some robots alive, the winner is the one with the most money
            if all([len(player_state.alive_robots) > 0 for player_state in new_round_state.values()]):
                winner = max(new_round_state, key=lambda player: new_round_state[player].total_money_made)
            else:
                # if not, the winner is the one with alive robots & most money
                winner = max(new_round_state, key=lambda player: new_round_state[player].total_money_made if len(
                    new_round_state[player].alive_robots) > 0 else 0)
            log.info(f"Winner is {winner} with {new_round_state[winner].total_money_made} money made.")

        if self.current_round > 1:
            for player_name in self.players:
                rewards_for_player = self.calculate_rewards_for_players_robots_based_on_old_state_commands_and_new_state(
                    old_state=current_round_state[player_name], new_state=new_round_state[player_name],
                    commands=self.action_queue_for_round[player_name], game_over=terminated,
                    won=(player_name == winner))
                new_round_state[player_name].robot_rewards = rewards_for_player

        for player_name in self.players:
            self.action_queue_for_round[player_name] = []

        player_rewards: Dict[str, Dict[str, float]] = {player_name: player_state.robot_rewards for
                                                       player_name, player_state in
                                                       new_round_state.items()}  # PlayerName -> {RobotID -> Reward}
        # Rückgabe des neuen Zustandes, ob das Spiel zu Ende ist und die Belohnungen für die Roboter der Spieler
        return new_round_state, truncated, terminated, player_rewards

    def get_action_mask_for_robot(self, player_name: str, robot_id: str):
        """Provides an action mask for the robot based on the current state of the environment"""
        money_available = self.player_states[player_name].money

        UPGRADE_1_COST = 50
        UPGRADE_2_COST = 300
        UPGRADE_3_COST = 1500
        UPGRADE_4_COST = 4000
        UPGRADE_5_COST = 15000
        HEALTH_RESTORE_COST = 50
        ENERGY_RESTORE_COST = 75

        DAMAGE_0_VALUE = 1
        DAMAGE_1_VALUE = 2
        DAMAGE_2_VALUE = 5
        DAMAGE_3_VALUE = 10
        DAMAGE_4_VALUE = 20
        DAMAGE_5_VALUE = 50

        HEALTH_0_VALUE = 10
        HEALTH_1_VALUE = 25
        HEALTH_2_VALUE = 50
        HEALTH_3_VALUE = 100
        HEALTH_4_VALUE = 200
        HEALTH_5_VALUE = 500

        MINING_0_VALUE = [ResourceType.COAL]
        MINING_1_VALUE = [ResourceType.COAL, ResourceType.IRON]
        MINING_2_VALUE = [ResourceType.COAL, ResourceType.IRON, ResourceType.GEM]
        MINING_3_VALUE = [ResourceType.COAL, ResourceType.IRON, ResourceType.GEM, ResourceType.GOLD]
        MINING_4_VALUE = [ResourceType.COAL, ResourceType.IRON, ResourceType.GEM, ResourceType.GOLD,
                          ResourceType.PLATINUM]
        MINING_5_VALUE = [ResourceType.COAL, ResourceType.IRON, ResourceType.GEM, ResourceType.GOLD,
                          ResourceType.PLATINUM]

        STORAGE_0_VALUE = 20
        STORAGE_1_VALUE = 50
        STORAGE_2_VALUE = 100
        STORAGE_3_VALUE = 200
        STORAGE_4_VALUE = 400
        STORAGE_5_VALUE = 1000

        MINING_SPEED_0_VALUE = 2
        MINING_SPEED_1_VALUE = 5
        MINING_SPEED_2_VALUE = 10
        MINING_SPEED_3_VALUE = 15
        MINING_SPEED_4_VALUE = 20
        MINING_SPEED_5_VALUE = 40

        ENERGY_0_VALUE = 20
        ENERGY_1_VALUE = 30
        ENERGY_2_VALUE = 40
        ENERGY_3_VALUE = 60
        ENERGY_4_VALUE = 100
        ENERGY_5_VALUE = 200

        ENERGY_REGEN_0_VALUE = 4
        ENERGY_REGEN_1_VALUE = 6
        ENERGY_REGEN_2_VALUE = 8
        ENERGY_REGEN_3_VALUE = 10
        ENERGY_REGEN_4_VALUE = 15
        ENERGY_REGEN_5_VALUE = 20

        robot: ObservationRobot = self.player_states[player_name].alive_robots[robot_id]
        robot_current_planet: ObservationPlanet = self.player_states[player_name].map[robot.planet_id]
        action_mask = np.zeros(shape=self.action_space.n, dtype=np.int8)

        log.debug(f"ROBOT {robot.robot_id} CURRENT PLANET " + str(robot_current_planet))
        log.debug(f"ROBOTs Planet {robot.planet_id} NEIGHBOURS " + str(robot_current_planet.neighbours))

        # Movement
        if robot.energy < robot_current_planet.movement_difficulty:
            action_mask[:4] = 0
        else:
            if robot_current_planet.neighbours.NORTH is not None and robot_current_planet.neighbours.NORTH.lower() != "none":
                action_mask[0] = 1
            if robot_current_planet.neighbours.EAST is not None and robot_current_planet.neighbours.EAST.lower() != "none":
                action_mask[1] = 1
            if robot_current_planet.neighbours.SOUTH is not None and robot_current_planet.neighbours.SOUTH.lower() != "none":
                action_mask[2] = 1
            if robot_current_planet.neighbours.WEST is not None and robot_current_planet.neighbours.WEST.lower() != "none":
                action_mask[3] = 1
        log.debug("ACTION MASK MOVEMENT " + str(action_mask[:4]))

        # Battle
        enemy_robots_on_planet = self.player_states[player_name].get_enemy_robots_on_planet(robot.planet_id)
        damage_levels = [1, 2, 5, 10, 20, 50]  # Level0, Level1, Level2, Level3, Level4, Level5
        # robots energy_needed for an attack is the damage level + 1, the first element in the list is level 0, so the energy needed is 1, so I need the index of the damage level
        energy_needed_for_attack = damage_levels.index(robot.damage) + 1
        if len(enemy_robots_on_planet) > 0 and robot.energy > energy_needed_for_attack:
            action_mask[4] = 1
            action_mask[5] = 1

        # Upgrades
        if money_available >= UPGRADE_1_COST:
            # order is always the same
            # dmg
            # health
            # mining
            # storage
            # mining speed
            # energy
            # energy regen
            if robot.damage < DAMAGE_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_DAMAGE_1.value] = 1
            if robot.max_health < HEALTH_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_HEALTH_1.value] = 1
            if len(robot.mineable_resources) < len(MINING_1_VALUE):
                action_mask[ActionsEnum.UPGRADE_MINING_1.value] = 1
            if robot.max_storage < STORAGE_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_STORAGE_1.value] = 1
            if robot.mining_speed < MINING_SPEED_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_MINING_SPEED_1.value] = 1
            if robot.max_energy < ENERGY_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_1.value] = 1
            if robot.energy_regen < ENERGY_REGEN_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_REGEN_1.value] = 1
        if money_available >= UPGRADE_2_COST:
            if robot.damage == DAMAGE_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_DAMAGE_2.value] = 1
            if robot.max_health == HEALTH_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_HEALTH_2.value] = 1
            if len(robot.mineable_resources) < len(MINING_2_VALUE):
                action_mask[ActionsEnum.UPGRADE_MINING_2.value] = 1
            if robot.max_storage == STORAGE_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_STORAGE_2.value] = 1
            if robot.mining_speed == MINING_SPEED_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_MINING_SPEED_2.value] = 1
            if robot.max_energy == ENERGY_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_2.value] = 1
            if robot.energy_regen == ENERGY_REGEN_1_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_REGEN_2.value] = 1
        if money_available >= UPGRADE_3_COST:
            if robot.damage == DAMAGE_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_DAMAGE_3.value] = 1
            if robot.max_health == HEALTH_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_HEALTH_3.value] = 1
            if len(robot.mineable_resources) < len(MINING_3_VALUE):
                action_mask[ActionsEnum.UPGRADE_MINING_3.value] = 1
            if robot.max_storage == STORAGE_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_STORAGE_3.value] = 1
            if robot.mining_speed == MINING_SPEED_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_MINING_SPEED_3.value] = 1
            if robot.max_energy == ENERGY_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_3.value] = 1
            if robot.energy_regen == ENERGY_REGEN_2_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_REGEN_3.value] = 1
        if money_available >= UPGRADE_4_COST:
            if robot.damage == DAMAGE_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_DAMAGE_4.value] = 1
            if robot.max_health == HEALTH_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_HEALTH_4.value] = 1
            if len(robot.mineable_resources) < len(MINING_4_VALUE):
                action_mask[ActionsEnum.UPGRADE_MINING_4.value] = 1
            if robot.max_storage == STORAGE_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_STORAGE_4.value] = 1
            if robot.mining_speed == MINING_SPEED_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_MINING_SPEED_4.value] = 1
            if robot.max_energy == ENERGY_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_4.value] = 1
            if robot.energy_regen == ENERGY_REGEN_3_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_REGEN_4.value] = 1
        if money_available >= UPGRADE_5_COST:
            if robot.damage == DAMAGE_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_DAMAGE_5.value] = 1
            if robot.max_health == HEALTH_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_HEALTH_5.value] = 1
            # if len(robot.mineable_resources) < len(MINING_5_VALUE):
            action_mask[ActionsEnum.UPGRADE_MINING_5.value] = 0  # Lohnt sich nie!
            if robot.max_storage == STORAGE_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_STORAGE_5.value] = 1
            if robot.mining_speed == MINING_SPEED_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_MINING_SPEED_5.value] = 1
            if robot.max_energy == ENERGY_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_5.value] = 1
            if robot.energy_regen == ENERGY_REGEN_4_VALUE:
                action_mask[ActionsEnum.UPGRADE_ENERGY_REGEN_5.value] = 1
        if money_available >= HEALTH_RESTORE_COST and robot.health < robot.max_health:
            action_mask[ActionsEnum.BUY_HEALTH_RESTORE.value] = 1
        if money_available >= ENERGY_RESTORE_COST and robot.energy < robot.max_energy:
            action_mask[ActionsEnum.BUY_ENERGY_RESTORE.value] = 1

        # Mining
        if robot_current_planet.resource is not None and robot_current_planet.resource_amount > 0 and robot.storage < robot.max_storage and robot_current_planet.resource in robot.mineable_resources:
            action_mask[ActionsEnum.MINE_RESOURCE.value] = 1
        # Regenerate
        if robot.energy < robot.max_energy:
            action_mask[ActionsEnum.REGENERATE.value] = 1

        if robot.storage > 0:
            action_mask[ActionsEnum.SELL_RESOURCE.value] = 1

        return action_mask

    def _send_actions_to_server(self, commands: List[List[Command]]):
        """ When all players have submitted their commands, send them to the server. """
        for commandsList in commands:
            commands = [command.dict() for command in commandsList]
            log.info(f"Sending commands to server: {commands}")
            return requests_retry_session().post(self.game_host + '/games/' + self.game_id + '/commands', json=commands)

    def convert_to_command_json_format(self, player_actions: PlayersActions) -> List[Command]:
        # Konvertiere die Aktionen in das Format, das der Game Server versteht
        current_player_state = self.player_states[player_actions.player_name]

        commands = []
        for robot_id, action in player_actions.actions_for_robots.items():
            command_object = CommandObject(robot_id=robot_id)
            current_robot: ObservationRobot = current_player_state.alive_robots[robot_id]
            current_planet: ObservationPlanet = current_player_state.map[current_robot.planet_id]
            (x, y) = (current_robot.x, current_robot.y)

            if action.value <= ActionsEnum.MOVE_WEST.value:
                command_type = CommandType.MOVEMENT
                move_direction = action.name.split("MOVE_")[1]
                if move_direction == "NORTH" and current_planet.neighbours.NORTH is not None:
                    command_object.target_id = current_planet.neighbours.NORTH
                elif move_direction == "EAST" and current_planet.neighbours.EAST is not None:
                    command_object.target_id = current_planet.neighbours.EAST
                elif move_direction == "SOUTH" and current_planet.neighbours.SOUTH is not None:
                    command_object.target_id = current_planet.neighbours.SOUTH
                elif move_direction == "WEST" and current_planet.neighbours.WEST is not None:
                    command_object.target_id = current_planet.neighbours.WEST

            elif action.value <= ActionsEnum.ATTACK_HIGHEST_VALUE_ENEMY_ROBOT.value:
                command_type = CommandType.BATTLE
                robots_on_same_planet = current_player_state.get_enemy_robots_on_planet(current_robot.planet_id)
                if len(robots_on_same_planet) == 0:
                    log.critical(
                        "=====No robots on the same planet to attack. ACTION MASKING IS NOT WORKING PROPERLY=====")
                if action.value == ActionsEnum.ATTACK_LOWEST_HP_ENEMY_ROBOT.value:
                    # ATTACK THE ROBOT WITH THE LOWEST HEALTH above 0
                    command_object.target_id = min(robots_on_same_planet, key=lambda robot: robot.health).robot_id
                if action.value == ActionsEnum.ATTACK_HIGHEST_VALUE_ENEMY_ROBOT.value:
                    command_object.target_id = max(robots_on_same_planet, key=lambda robot: robot.money_value).robot_id
                # CHECK IF THE TARGET ID IS NOT OUR OWN ROBOT
                if command_object.target_id in current_player_state.alive_robots:
                    log.critical(
                        f"ATTACKING OUR OWN ROBOT {command_object.target_id} WITH ACTION {action} FOR ROBOT {robot_id}")
                    command_object.target_id = None
            elif action.value == ActionsEnum.MINE_RESOURCE.value:
                command_type = CommandType.MINING
                command_object.target_id = current_robot.planet_id
            elif action.value == ActionsEnum.REGENERATE.value:
                command_type = CommandType.REGENERATE
            elif action.value == ActionsEnum.SELL_RESOURCE.value:
                command_type = CommandType.SELLING
            else:
                # Buying Commands have an item name
                command_object.item_name = action.get_item_name()
                command_object.item_quantity = 1
                command_type = CommandType.BUYING

            command = Command(player_name=player_actions.player_name, game_id=self.game_id,
                              command_type=command_type.name, command_object=command_object)
            commands.append(command)
        return commands

    def _get_state_from_server_for_players_for_current_round(self):
        for player in self.players:
            state = requests_retry_session().get(f'{self.game_host}/games/{self.game_id}/currentRound/players/{player}/new').json()
            self.player_states[player] = PlayerState.parse_obj(state)
            log.debug(f"{player}'s State\n{self.player_states[player]}")
            self.current_round = self.player_states[player].current_round
            self.global_current_round_state[player] = self.player_states[player]

        log.info(f"Current Round: {self.current_round}")

    def calculate_rewards_for_players_robots_based_on_old_state_commands_and_new_state(self, old_state: PlayerState,
                                                                                       new_state: PlayerState,
                                                                                       commands: List[Command],
                                                                                       game_over: bool = False,
                                                                                       won: bool = False) -> Dict[
        str, float]:
        """
        Supposed to calculate rewards for the player's robots based on the previous round state, the actions that
        were chosen and the new state.
        """

        robot_rewards: Dict[str, float] = defaultdict(float)  # robot_id -> reward

        # Discovering planets
        newly_discovered_planets = old_state.visited_planets ^ new_state.visited_planets
        for command in filter(lambda cmd: cmd.command_type == CommandType.MOVEMENT.name, commands):
            if command.command_object.target_id in newly_discovered_planets:
                robot_rewards[command.command_object.robot_id] += 10

        # Mining rewards
        planet_mine_counter = Counter(
            cmd.command_object.target_id for cmd in commands if cmd.command_type == CommandType.MINING.name)
        for command in filter(lambda cmd: cmd.command_type == CommandType.MINING.name, commands):
            resource_type = new_state.map[command.command_object.target_id].resource
            resource_value_map = {ResourceType.COAL: 5, ResourceType.IRON: 15, ResourceType.GEM: 18,
                                  ResourceType.GOLD: 21,
                                  ResourceType.PLATINUM: 24}
            if resource_type:
                reward = resource_value_map[resource_type] * (
                            1 + planet_mine_counter[command.command_object.target_id] / 25)
            else:
                reward = 0  # Punishment for mining on a planet without resources
            robot_rewards[command.command_object.robot_id] += reward

        # Selling rewards
        for command in filter(lambda cmd: cmd.command_type == CommandType.SELLING.name, commands):
            robot_id = command.command_object.robot_id
            if robot_id not in new_state.alive_robots:
                robot = new_state.dead_robots[robot_id]
            else:
                robot = new_state.alive_robots[robot_id]
            money_made = robot.money_made - old_state.alive_robots[robot_id].money_made
            storage_usage = float(robot.storage) / float(
                robot.max_storage)
            robot_rewards[robot_id] += money_made * storage_usage  # Ensures that inefficient selling is punished

        # ATTACKING REWARDS
        attacking_commands: List[Command] = [command for command in commands if
                                             command.command_type == CommandType.BATTLE.name]
        coordinated_attacks: Dict[str, int] = Counter(
            command.command_object.target_id for command in commands if command.command_type == CommandType.BATTLE.name)

        for command in attacking_commands:
            our_robot_id = command.command_object.robot_id
            our_robot = old_state.alive_robots[our_robot_id]
            # Check if enemy robot was killed/ or got damaged
            enemy_killed_robots_ids: Set[str] = set(
                kill_info[1].robot_id for kill_infos in new_state.killed_robots.values() for kill_info in kill_infos)

            if command.command_object.target_id in enemy_killed_robots_ids:
                # Get Enemy Robot Money Value, by checking the killed_robots dictionary. The key is our robot id, the value is a tuple with the enemy player name and the enemy robot
                enemy_robot_money_value = None
                for kill_info in new_state.killed_robots.values():
                    for enemy_player_name, enemy_robot in kill_info:
                        if enemy_robot.robot_id == command.command_object.target_id:
                            enemy_robot_money_value = enemy_robot.money_value
                            break
                enemy_robots_killed = len(new_state.killed_robots)  # Number of killed enemy robots
                num_enemy_robots_total = len(old_state.alive_enemy_robots)

                if num_enemy_robots_total > 0:
                    percent_enemies_killed = enemy_robots_killed / num_enemy_robots_total
                else:
                    percent_enemies_killed = 0

                kill_reward = 0.2 * enemy_robot_money_value * (
                        1 + coordinated_attacks[command.command_object.target_id] / 25)
                kill_impact_factor = 1 + (percent_enemies_killed * 2)  # Scale up to 3x reward
                adjusted_kill_reward = kill_reward * kill_impact_factor

                robot_rewards[our_robot_id] += adjusted_kill_reward
            else:
                log.debug(f"OLD STATE ALIVE ENEMY ROBOTS {old_state.alive_enemy_robots}")
                log.debug(f"NEW STATE ALIVE ENEMY ROBOTS {new_state.alive_enemy_robots}")
                reward = (10 + our_robot.damage) * (1 + coordinated_attacks[command.command_object.target_id] / 25)
                robot_rewards[our_robot_id] += reward
        our_freshly_died_robots: Set[
            str] = new_state.dead_robots.keys() - old_state.dead_robots.keys()
        for robot_id in our_freshly_died_robots:
            robot = old_state.alive_robots[robot_id]
            base_death_penalty = robot.money_value * 2

            # # Adjustment based on our robot count
            # if new_state.alive_robots and len(new_state.alive_robots) > 0:
            #     team_size_factor = 1 + (2.0 / len(old_state.alive_robots))  # Increased influence
            # else:
            #     team_size_factor = 3  # Avoid division by zero, and set a harsh penalty if the last robot dies
            robot_rewards[robot_id] -= base_death_penalty * 2
        return robot_rewards

    def render(self):
        # Optional: Visualisierung des aktuellen Zustands der Umgebung
        pass