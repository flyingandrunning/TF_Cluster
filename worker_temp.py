import tensorflow as tf

worker1 = "192.168.199.167:22223"
worker_hosts = [worker1]
cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
server.join()
