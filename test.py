from StockSimEnv import Env
from ActorNetwork import ActorNetwork
import tensorflow as tf
import glo
#glo.init()
#env = Env()
#env.get_state()
sess = tf.Session()
actor = ActorNetwork(sess)
