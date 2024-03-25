from unittest import TestCase

import numpy as np
import requests

import util
import logging as log

from Config import Config
from envWrapper import RobotGameEnvironment, ActionsEnum, ResourceType


class TestRobotGameEnvironment(TestCase):

    def setUp(self):
        util.configure_logger()
        self.env = RobotGameEnvironment()


    def delete_active_game(self):
        game_id = self.env.game_id
        response = requests.delete(f"{Config.game_host}:{Config.game_port}/games/{game_id}")
        if response.status_code == 200:
            log.info(f"Game {game_id} deleted")



    def test_simulated_actions(self):
        """
        Test if the actions are being simulated correctly.
        4 Robots are bought, then one robot buys an upgrade, the simulated state should show that the money has reduced.
        :return:
        """
        env = self.env
        env.reset(4)
        randomRobotId = list(env.player_states["Player1"].alive_robots.keys())[0]
        randomRobotId2 = list(env.player_states["Player1"].alive_robots.keys())[1]
        randomRobotId3 = list(env.player_states["Player1"].alive_robots.keys())[2]
        obs = env.get_observation_for_robot(player_name="Player1", robot_id=randomRobotId)
        env.add_robot_action_to_queue(player_name="Player1", robot_id=randomRobotId,
                                      action=ActionsEnum.UPGRADE_DAMAGE_1)
        obs2 = env.get_observation_for_robot(player_name="Player1", robot_id=randomRobotId2)
        env.add_robot_action_to_queue(player_name="Player1", robot_id=randomRobotId2,
                                      action=ActionsEnum.UPGRADE_ENERGY_1)
        obs3 = env.get_observation_for_robot(player_name="Player1", robot_id=randomRobotId3)
        index_of_current_money_amount = 0
        log.info(f"OBSERVATION 1 {obs[0:3]}")
        log.info(f"OBSERVATION 2 {obs2[0:3]}")
        log.info(f"OBSERVATION 3 {obs3[0:3]}")
        assert obs[index_of_current_money_amount] > obs2[index_of_current_money_amount] > obs3[
            index_of_current_money_amount]

    def test_reset_game_buys_5_robots_ends_up_round_1(self):
        env = self.env
        env.reset(5)
        assert len(env.player_states["Player1"].alive_robots) == 5
        assert len(env.player_states["Player2"].alive_robots) == 5
        assert env.player_states["Player1"].current_round == 1
        assert env.player_states["Player2"].current_round == 1
        assert env.player_states["Player1"].money == 0
        assert env.player_states["Player2"].money == 0


    def test_active_game_with_two_players_taking_random_actions(self):
        env = self.env
        env.reset(5)
        terminated = False

        for i in range(Config.NUM_ROUNDS):
            if terminated:
                break
            for player_name in env.players:
                for robot_id in env.player_states[player_name].alive_robots:
                    obs = env.get_observation_for_robot(player_name=player_name, robot_id=robot_id) # Has to be done so the simulated state is safed and action masking works properly.
                    action_mask = env.get_action_mask_for_robot(player_name=player_name, robot_id=robot_id)
                    print(f"Player {player_name} robot {robot_id} action mask {action_mask}")
                    action = env.action_space.sample(mask=action_mask)
                    print(f"Player {player_name} robot {robot_id} action {action}")
                    env.add_robot_action_to_queue(player_name=player_name, robot_id=robot_id, action=action)
            new_round_state, done, terminated, player_rewards = env.step()


