import csv
import logging
from typing import Dict, NamedTuple, Optional

import numpy as np
import envWrapper
from DQN.dqn_agent import DQNAgent
from DQN.util import plot_learning_curve
from util import configure_logger
import requests

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""


def write_data_to_csv(filename, scores, eps_history, steps_array, win_ctr, money_delta_player1_to_player2):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Schreiben der Kopfzeile
        writer.writerow(['Score', 'Epsilon', 'Steps', 'Money Delta Player1 to Player2', 'Win %'])
        # Schreiben der Daten
        for i in range(len(scores)):
            writer.writerow(
                [scores[i], eps_history[i], steps_array[i], money_delta_player1_to_player2[i], win_ctr / (i + 1) * 100])


def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
    res = requests.post(url)
    if res.status_code != 200:
        print("Error while sending message to telegram")
        print(res.json())


class Transition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: Optional[float]
    new_observation: Optional[np.ndarray]
    terminal: Optional[bool]


TransitionDict = Dict[str, Dict[str, Transition]]  # player_name -> robot_id -> transition


def complete_transitions(transitions: TransitionDict, rewards: Dict[str, Dict[str, float]], terminated: bool):
    for (player_name, robot_transitions) in transitions.items():
        for (robot_id, transition) in robot_transitions.items():
            transition: Transition

            logging.debug(
                f"==========BEFORE COMPLETED TRANSITION {robot_id}========\n {transition.reward} {transition.new_observation} {transition.terminal} <<<<<<< Should be None, None, None")
            transition = transition._replace(reward=rewards[player_name][robot_id],
                                             new_observation=env.get_observation_for_robot(player_name=player_name,
                                                                                           robot_id=robot_id),
                                             terminal=terminated)
            robot_transitions[robot_id] = transition
            logging.debug(
                f"==========COMPLETED TRANSITION {robot_id}========\n {transition.reward} {transition.terminal} <<<<<<< Should have values.")
        transitions[player_name] = robot_transitions


if __name__ == '__main__':
    configure_logger()
    if TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != "":
        send_telegram_message("Starting Training...")
    env = envWrapper.RobotGameEnvironment()

    best_score = -np.inf
    print(env.observation_space.shape)
    load_checkpoint = True
    load_checkpoint_agent_2 = True
    should_learn = False
    n_games = 1  # Number of games to play

    agent = DQNAgent(gamma=0.99, epsilon=0.99, lr=0.0001, input_dims=env.old_observation_space.shape,
                     n_actions=env.action_space.n, mem_size=125_000, batch_size=128, eps_min=0.05, replace=500,
                     eps_dec=0.0000176, algo='DQNAgent', env_name='MSD', chkpt_dir='modelsAgent1/')

    agent2 = DQNAgent(gamma=0.99, epsilon=0.0, lr=0.0001, input_dims=env.old_observation_space.shape,
                      n_actions=env.action_space.n, mem_size=0, batch_size=128, eps_min=0.00, replace=500,
                      eps_dec=0.0000176, algo='DQNAgent', env_name='MSD',
                      chkpt_dir='modelsAgent2/') if load_checkpoint_agent_2 else None
    if agent2:
        agent2.load_models()
    if load_checkpoint:
        agent.load_models()
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(
        n_games) + '_games' + '_batch_size' + str(agent.batch_size) + '_eps_min' + str(
        agent.eps_min) + '_eps_dec' + str(agent.eps_dec) + '_replace' + str(agent.replace_target_cnt)
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array, win_ctr, money_delta_player1_to_player2 = [], [], [], 0, []

    for i in range(n_games):
        terminated = False
        score = 0
        truncated = False

        player_states: Dict[str, envWrapper.PlayerState] = env.reset()

        # Transition dict, because all player robots choose actions before the env step is taken, so we only then get the next observation and reward for each robot
        # This is a bit of a hack, but it works for now
        transitions: TransitionDict = {}
        while not (terminated or truncated):
            for (player_name, player_state) in player_states.items():
                player_name: str
                player_state: envWrapper.PlayerState
                transitions[player_name] = {}
                # Player 1 is the agent, he chooses actions based on his Q Network

                players_available_money = player_state.money
                if player_state.alive_robots.__len__() == 0 and players_available_money >= 100:
                    env.add_buy_robots_action_to_queue(player_name=player_name,
                                                       buy_new_robots_amount=players_available_money // 100)
                if players_available_money >= 1000:
                    env.add_buy_robots_action_to_queue(player_name=player_name, buy_new_robots_amount=3)
                elif players_available_money >= 500:
                    env.add_buy_robots_action_to_queue(player_name=player_name, buy_new_robots_amount=2)

                if player_name == "Player1":
                    player_1_robots: Dict[str, envWrapper.ObservationRobot] = player_state.alive_robots
                    for (robot_id, robot) in player_1_robots.items():
                        observation: np.ndarray = env.get_observation_for_robot_old(player_name, robot_id)
                        action_mask: np.ndarray = env.get_action_mask_for_robot(player_name, robot_id)
                        action = agent.choose_action(observation, action_mask)
                        env.add_robot_action_to_queue(player_name=player_name, robot_id=robot_id, action=action)
                        if should_learn:
                            transitions[player_name][robot_id] = Transition(observation, action, None, None, None)

                else:
                    player_2_robots: Dict[str, envWrapper.ObservationRobot] = player_state.alive_robots
                    for (robot_id, robot) in player_2_robots.items():
                        observation: np.ndarray = env.get_observation_for_robot_old(player_name, robot_id)
                        action_mask: np.ndarray = env.get_action_mask_for_robot(player_name, robot_id)
                        action = agent2.choose_action(observation, action_mask) if agent2 else env.action_space.sample(
                            action_mask)
                        env.add_robot_action_to_queue(player_name=player_name, robot_id=robot_id, action=action)

            new_player_states, terminated, truncated, rewards = env.step()
            logging.info(f"CURRENT ROUND {env.current_round} TERMINATED: {terminated} TRUNCATED: {truncated} ")
            if should_learn:
                complete_transitions(transitions, rewards, terminated)
                for (player_name, robot_transitions) in transitions.items():
                    if player_name == "Player1":
                        for (robot_id, transition) in robot_transitions.items():
                            agent.store_transition(transition.observation, transition.action, transition.reward,
                                                   transition.new_observation,
                                                   int(transition.terminal))
                            score += transition.reward
                agent.learn()

            player_states = new_player_states

            n_steps += 1
        money_delta = player_states["Player1"].total_money_made - player_states["Player2"].total_money_made

        if len(player_states["Player1"].alive_robots) > 0 and len(player_states["Player2"].alive_robots) > 0:
            if money_delta > 0:
                win_ctr += 1
        elif len(player_states["Player1"].alive_robots) > 0 and len(player_states["Player2"].alive_robots) == 0:
            win_ctr += 1
        scores.append(score)
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        money_delta_player1_to_player2.append(money_delta)

        avg_money_delta = np.mean(money_delta_player1_to_player2[-100:])

        if avg_score > best_score:
            if should_learn:
                agent.save_models()
            best_score = avg_score
        print(
            f"Episode: {i}, Score: {score}, Avg Score: {avg_score}, Best Score(Rewards): {best_score}, Steps: {n_steps}, Avg Money Delta (Last 25): {avg_money_delta}, Epsilon: {agent.epsilon} Wins: {win_ctr / (i + 1) * 100}%")
        logging.critical(
            f"Episode: {i}, Score: {score}, Avg Score: {avg_score}, Best Score(Rewards): {best_score}, Steps: {n_steps}, Avg Money Delta (Last 25): {avg_money_delta}  Epsilon: {agent.epsilon} Wins: {win_ctr / (i + 1) * 100}%")
        if i % 25 == 0 and TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != "":
            send_telegram_message(
                f"Episode: {i}, Score: {score}, Avg Score: {avg_score}, Best Score(Rewards): {best_score}, Steps: {n_steps}, Avg Money Delta (Last 25): {avg_money_delta}, Epsilon: {agent.epsilon} Wins: {win_ctr / (i + 1) * 100}%")

        if i != 0 and (i % 100 == 0 or i == (n_games - 1)):
            fname_new = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(i + 1) + "von" + str(
                n_games) + '_games' + '_batch_size' + str(agent.batch_size) + '_eps_min' + str(
                agent.eps_min) + '_eps_dec' + str(agent.eps_dec) + '_replace' + str(agent.replace_target_cnt)
            figure_file2 = './plots/' + fname_new + f'_episode_{i}.png'
            plot_learning_curve(steps_array, scores, eps_history, figure_file2)
            if TELEGRAM_BOT_TOKEN != "" and TELEGRAM_CHAT_ID != "":
                with open(figure_file2, 'rb') as image:
                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                    files = {'photo': image}
                    data = {'chat_id': TELEGRAM_CHAT_ID}
                    response = requests.post(url, files=files, data=data)
        env.reset()
    csv_name = 'training_data_' + agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(
        n_games) + '_games' + '_batch_size' + str(agent.batch_size) + '_eps_min' + str(
        agent.eps_min) + '_eps_dec' + str(agent.eps_dec) + '_replace' + str(agent.replace_target_cnt)

    write_data_to_csv(f"{csv_name}.csv", scores, eps_history, steps_array, win_ctr,
                      money_delta_player1_to_player2)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
