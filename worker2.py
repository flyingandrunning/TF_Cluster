import tensorflow as tf

worker1 = "192.168.199.167:33333"
worker2 = "192.168.199.133:33333"
worker_hosts = [worker1, worker2]

ps1 = "192.168.199.205:33333"
ps_hosts = [ps1]

cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts, "ps": ps_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
server.join()
