import tensorflow as tf
import trainer
import gym
import numpy as np


def choose_action(state:np.ndarray, agent:trainer.Agent):
    feed = {
        agent.network.states: np.reshape(state, [-1, *agent.input_shape])
    }
    action = agent.sess.run(agent.network.action_prob, feed)
    action = np.squeeze(action)

    return action.argmax() + 1



def play_episode(env: gym.Env, agent: trainer.Agent, pipeline, render=False):

    s = env.reset()

    state_diff = s = pipeline(s)

    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()

        action = choose_action(state_diff, agent)
        s2, r, done, _ = env.step(action)

        total_reward += r

        s2 = pipeline(s2)

        state_diff = s2 - s
        s = s2

    return total_reward


def main(_):
    env = gym.make("Pong-v0")
    input_shape = [80, 80, 1]
    output_dim = 3
    network = trainer.A3CNetwork("name", input_shape, output_dim)
    agent = trainer.Agent("name", env, network, input_shape, output_dim)

    with tf.Session() as sess:
        agent.sess = sess
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("train_logs"))
        print("Model Loaded")

        for i in range(FLAGS.n_episode):
            reward = play_episode(env, agent, trainer.pipeline)
            print(i, reward)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--n_episode",
        type=int,
        default=1,
        help="Number of episode to run"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="Render a game play"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
