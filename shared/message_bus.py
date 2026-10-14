"""
shared/message_bus.py

Redis Pub/Sub message bus for the Warehouse Intelligence System.

STATUS: SKELETON — channel names are fixed.
Publishers and subscribers are implemented per function.

Channel registry:
    f1.inventory    — F1 publishes, F4 subscribes
    f3.ergonomics   — F3 publishes, F4 subscribes
    f4.routes       — F4 publishes, frontend subscribes
"""

import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Channel names — agreed across all functions, do not change unilaterally
CHANNEL_F1_INVENTORY  = "f1.inventory"
CHANNEL_F3_ERGONOMICS = "f3.ergonomics"
CHANNEL_F4_ROUTES     = "f4.routes"


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )


def publish(channel: str, message: dict):
    """Publish a message dict to a named channel."""
    client = get_redis_client()
    client.publish(channel, json.dumps(message))


def subscribe(channel: str):
    """Return a pubsub subscriber for a named channel."""
    client = get_redis_client()
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    return pubsub