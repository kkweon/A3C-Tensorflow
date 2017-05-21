import argparse
import sys

import tensorflow as tf
import gym
import numpy as np
import logging
from scipy.misc import imresize

FLAGS = None

logger = logging.getLogger(__name__)


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image

    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image

    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)

    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)

    return image


def resize_image(image, new_HW):
    """Returns a resized image

    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)

    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    return imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0

    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)

    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image


def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.

        However, when the reward is either -1 or 1,
        it means the game has been reset.

        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim):
        """Network structure is defined here

        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

        action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
        net = self.states

        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net,
                                   filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net,
                                   filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("fc1"):
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 256, name='dense')
            net = tf.nn.relu(net, name='relu')

        # actor network
        action_values = tf.layers.dense(net, output_dim, name="final_fc")
        self.action_prob = tf.nn.softmax(action_values, name="action_prob")
        single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

        entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
        entropy = tf.reduce_sum(entropy, axis=1)

        log_action_prob = tf.log(single_action_prob + 1e-7)
        maximize_objective = log_action_prob * self.advantage + entropy * 0.005
        self.actor_loss = - tf.reduce_sum(maximize_objective)

        # value network
        self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
        self.value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        self.total_loss = self.actor_loss + self.value_loss * .5
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=.99)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)

        loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        value_summary = tf.summary.histogram("values", self.values)

        self.summary_op = tf.summary.merge([loss_summary, value_summary])


class Agent(object):

    def __init__(self, name, env, network, input_shape, output_dim):
        """Agent worker thread

        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.env): Gym environment
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork (H, W, C)
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary
                TODO: Add summary
        """
        self.name = name
        self.env = env
        self.network = network
        self.input_shape = input_shape
        self.output_dim = output_dim

        self.summary_writer = tf.summary.FileWriter("train_logs/{}".format(name))

    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        logger.info(message)
        print(message)

    def play_episode(self, session):
        self.sess = session
        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = pipeline(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0

        episode_length = 0

        while not done:

            a = self.choose_action(state_diff)
            s2, r, done, _ = self.env.step(a)

            s2 = pipeline(s2)
            total_reward += r

            states.append(state_diff)
            actions.append(a)
            rewards.append(r)

            state_diff = s2 - s
            s = s2

            episode_length += 1

            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    self.train(states, actions, rewards)
                    states, actions, rewards = [], [], []
                    time_step = 0

        self.print(total_reward)
        summary = tf.Summary()
        summary.value.add(tag="episode_reward", simple_value=total_reward)
        summary.value.add(tag="episode_length", simple_value=episode_length)

        self.summary_writer.add_summary(summary, global_step=self.sess.run(tf.train.get_or_create_global_step()))
        self.summary_writer.flush()

        return total_reward

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.network.states: states
        }

        action = self.sess.run(self.network.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim) + 1, p=action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - 1
        rewards = np.array(rewards)

        feed = {
            self.network.states: states
        }

        values = self.sess.run(self.network.values, feed)

        rewards = discount_reward(rewards, gamma=0.99)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-7

        feed = {
            self.network.states: states,
            self.network.actions: actions,
            self.network.rewards: rewards,
            self.network.advantage: advantage
        }

        self.sess.run(self.network.train_op, feed_dict=feed)


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(FLAGS.task_index),
                cluster=cluster)):
            network = A3CNetwork("task_{}".format(FLAGS.task_index), input_shape=[80, 80, 1], output_dim=3)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000), ]

        env = gym.make("Pong-v0")

        if FLAGS.task_index == 0:
            env = gym.wrappers.Monitor(env, "monitor", force=True)

        agent = Agent("task_{}".format(FLAGS.task_index), env, network, [80, 80, 1], output_dim=3)
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="train_logs",
                                               save_summaries_secs=None,
                                               save_summaries_steps=None,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                reward = agent.play_episode(mon_sess)
                agent.print(reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
