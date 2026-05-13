import redis
import os


def create_redis_engine(logger):
    # Use localhost when running outside Docker, redis when inside Docker
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    
    try:
        engine = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
        # Test connection
        engine.ping()
        logger.info(f"Redis engine created successfully at {redis_host}:{redis_port}")
        return engine
    except redis.ConnectionError:
        # If localhost fails, try redis (Docker hostname)
        if redis_host == "localhost":
            logger.warning(f"Failed to connect to localhost:6379, trying redis:6379...")
            try:
                engine = redis.StrictRedis(host='redis', port=6379, db=0)
                engine.ping()
                logger.info("Redis engine created successfully at redis:6379")
                return engine
            except Exception as e:
                logger.error(f"Could not create redis engine: {e}")
                raise
        else:
            raise
    except Exception as e:
        logger.error(f"Could not create redis engine due to {e}")
        raise
