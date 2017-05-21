import tensorflow as tf
import train
import numpy as np
import gym


def get_action(network, state):
    sess = tf.get_default_session()

    feed = {
        network.states: state
    }
    action_prob = sess.run(network.action_prob, feed)

    action_prob = np.squeeze(action_prob)

    return np.random.choice(len(action_prob), p=action_prob)


def preprocess_state(state, input_dim):

    return np.array(state).reshape(-1, input_dim)


def play_episode(env, network):
    input_dim = env.observation_space.shape[0]
    state = env.reset()

    done = False
    total_reward = 0
    while not done:

        state = preprocess_state(state, input_dim)
        action = get_action(network, state)

        state, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


def main():
    sess = tf.InteractiveSession()
    global_network = train.A3CNetwork(name="global", input_dim=4, output_dim=2, hidden_dims=[16, 32])
    env = gym.make("CartPole-v0")

    save_path = "checkpoint/model.ckpt"
    saver = tf.train.Saver()

    saver.restore(sess, save_path)

    for _ in range(10):
        r = play_episode(env, global_network)
        print(r)


if __name__ == '__main__':
    main()