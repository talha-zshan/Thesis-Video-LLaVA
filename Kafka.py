from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic  
from confluent_kafka import KafkaError
import logging

# Setting up logging for visibility of operations
logging.basicConfig(level=logging.INFO)

class KafkaManager:
    def __init__(self, config):
        self.producer_config = config.copy()
        self.consumer_config = config.copy()
        self.consumer_config.update({
            'group.id': 'my-consumer-group',
            'auto.offset.reset': 'earliest'
        })
        self.admin_config = {'bootstrap.servers': config['bootstrap.servers']}
        self.admin_client = AdminClient(self.admin_config)
        self.producer = Producer(self.producer_config)
        self.consumer = Consumer(self.consumer_config)

    def create_topic(self, all_topics, num_partitions=1, replication_factor=1):
        """
        Create a topic if it does not exist.
        Args:
        topic_name (str): Name of the topic to be created.
        num_partitions (int): Number of partitions for the topic.
        replication_factor (int): Replication factor for the topic.
        """
        topic_list = [NewTopic(kafka_topic, num_partitions, replication_factor) for kafka_topic in all_topics]
        try:
            # Call create_topics to asynchronously create topics, a dict of <topic,future> is returned.
            fs = self.admin_client.create_topics(topic_list)
            # Wait for each operation to finish.
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    logging.info(f"Topic {topic} created")
                except Exception as e:
                    if e.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                        logging.error(f"Failed to create topic {topic}: {e}")
        except Exception as e:
            logging.error(f"Failed to create topic: {e}")

    def produce_message(self, topic, message):
        """
        Produce a message to a specified Kafka topic.
        Args:
        topic (str): The topic to which the message will be sent.
        message (str): The message to be sent.
        """
        def acked(err, msg):
            if err is not None:
                logging.error(f"Failed to deliver message: {err.str()}")
            else:
                logging.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

        # Trigger the asynchronous send of a message
        self.producer.produce(topic, message.encode('utf-8'), callback=acked)
        # Wait up to 1 second for events. Callbacks will be invoked during
        # this method call if the message is acknowledged.
        self.producer.poll(1)
        self.producer.flush()

    def consume_messages(self, topic):
        """
        Consume a single message from a specified Kafka topic and then shut down.
        Args:
        topic (str): The topic to subscribe to and consume a message from.
        """
        self.consumer.subscribe([topic])
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue  # Wait for the next poll if no message is received
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event, unlikely to happen right at startup but included for completeness
                        logging.info('End of partition reached {0}/{1}'
                                    .format(msg.topic(), msg.partition()))
                    else:
                        logging.error('Error occurred: {0}'.format(msg.error().str()))
                    break  # Exit after handling the error
                else:
                    # Successful message consumption
                    logging.info('Received message: {0}'.format(msg.value().decode('utf-8')))
                    break  # Exit after receiving the first message
        finally:
            self.consumer.close()  # Ensure the consumer is properly closed after reading the message
            return msg.value().decode('utf-8')



# Uncomment the following lines to initialize KafkaManager and perform operations
all_topics = ["camera-1", "camera-2", "camera-3", "camera-4"]
kafka_config = {'bootstrap.servers': 'localhost:9092'}
kafka_manager = KafkaManager(kafka_config)
# kafka_manager.create_topic(all_topics, 3, 1)
kafka_manager.produce_message('camera-1', 'demo_videos/talking_dog.mp4')
kafka_manager.produce_message('camera-1', 'demo_videos/talking_dog.mp4')
kafka_manager.produce_message('camera-1', 'demo_videos/talking_dog.mp4')

# kafka_manager.consume_messages('camera-1')
