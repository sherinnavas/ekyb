import random

WEB_ANALYSIS_DATA = {
    "traffic": [
        {
            "date": "2023-01-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-02-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-03-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-04-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-05-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-06-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-07-01",
            "traffic": random.uniform(1000, 1500)
        },
        {
            "date": "2023-08-01",
            "traffic": random.uniform(1000, 1500)
        }
    ]
}

WEB_AVERAGE_DURATION_DATA = { "duration": [
    {
        "date": "2023-01-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-02-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-03-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-04-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-05-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-06-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-07-01",
        "average_visit_duration": random.uniform(200, 1000)
    },
    {
        "date": "2023-08-01",
        "average_visit_duration": random.uniform(200, 1000)
    }
]
}

device_id = str(random.randint(10000000, 99999999))

THM_RESPONSE = {
  "identity_verification": {
    "user_behavior_match": True,  # Generate a random boolean value
    "identity_verified": True  # Generate a random boolean value
  },
  "device_fingerprint": {
    "device_id": device_id  # Include the generated device ID
  },
  "geolocation": {
    "country": "Kingdom of Saudi Arabia",
    "latitude": round(random.uniform(-90, 90), 4),  # Generate a random latitude
    "longitude": round(random.uniform(-180, 180), 4)  # Generate a random longitude
  },
  "bot_detection": {
    "is_bot": False  # Generate a random boolean value
  }
}
