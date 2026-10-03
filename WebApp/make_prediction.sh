#!/bin/bash

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "age": 22,
    "gender": "Male",
    "occupation_type": "Student",
    "chronotype": "Evening",
    "bedtime_phone_minutes": 80,
    "primary_bedtime_app": "TikTok / Reels",
    "screen_brightness_pct": 70,
    "blue_light_filter_active": 0,
    "caffeine_post_5pm_mg": 120,
    "physical_activity_min": 30,
    "morning_alarm_snoozes": 2
}'