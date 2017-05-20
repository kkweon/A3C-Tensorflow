import tensorflow as tf
import numpy as np
import threading
import gym


def copy_src_to_dst(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def pipeline(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_dim, output_dim, hidden_dims=[16, 32], logdir=None):
        """

        Assumes input_dim is flat
        (N, D)
        """
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, input_dim], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            for idx, h_dim in enumerate(hidden_dims):
                with tf.variable_scope("layer{}".format(idx)):
                    net = tf.layers.dense(net, h_dim, name="fc")
                    net = tf.nn.relu(net)

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)


class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_dim, output_dim, hidden_dims, logdir=None):
        super(Agent, self).__init__()
        self.local = A3CNetwork(name, input_dim, output_dim, hidden_dims, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir

    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        print(message)

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = pipeline(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0
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

            if r == -1 or r == 1:
                time_step += 1

            if time_step >= 5 or done:
                self.train(states, actions, rewards)
                self.sess.run(self.global_to_local)
                states, actions, rewards = [], [], []
                time_step = 0

        self.print(total_reward)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, input_dim)
        """
        states = np.reshape(states, [-1, self.input_dim])
        feed = {
            self.local.states: states
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim) + 1, p=action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - 1
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_rewards(rewards, gamma=0.99)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-7

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)


def main():
    try:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()

        save_path = "checkpoint/model.ckpt"
        n_threads = 8
        input_dim = 80 * 80
        output_dim = 3  # {1, 2, 3}
        hidden_dims = [256, 256]
        global_network = A3CNetwork(name="global",
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    hidden_dims=hidden_dims)

        thread_list = []
        env_list = []

        for id in range(n_threads):
            env = gym.make("Pong-v0")

            if id == 0:
                env = gym.wrappers.Monitor(env, "monitors", force=True)

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name="thread_{}".format(id),
                                 global_network=global_network,
                                 input_dim=input_dim,
                                 output_dim=output_dim,
                                 hidden_dims=hidden_dims)
            thread_list.append(single_agent)
            env_list.append(env)

        init = tf.global_variables_initializer()
        sess.run(init)

        for t in thread_list:
            t.start()

        print("Ctrl + C to close")
        coord.wait_for_stop()

    except KeyboardInterrupt:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path)
        print()
        print("=" * 10)
        print('Checkpoint Saved to {}'.format(save_path))
        print("=" * 10)

        print("Closing threads")
        coord.request_stop()
        coord.join(thread_list)

        print("Closing environments")
        for env in env_list:
            env.close()

        sess.close()


if __name__ == '__main__':
    main()
